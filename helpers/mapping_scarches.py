import warnings

warnings.filterwarnings("ignore")

import os
import argparse

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import scipy
from scipy.sparse import csr_matrix
import scvi
from helpers.log import get_logger

_log = get_logger('mapping_scarches')

def is_lognorm(mat, max_row_num=10000, max_col_num=None):
    """
    Detect if matrix is log-transformed using robust hybrid approach.
    Combines range-based, minimum value, and statistical heuristics.
    Works efficiently with sparse matrices without materializing to dense.
    
    Log-normalized data (typically log1p):
    - Max values typically < 100 (often < 20)
    - Min nonzero value typically < 1 (e.g., log(2) ≈ 0.693)
    - Variance/mean ratio differs from Poisson
    
    Raw counts:
    - Max values often > 100
    - Min nonzero value is typically 1 (discrete counts)
    - Variance approximately equals mean (Poisson-like)
    """
    is_sparse = scipy.sparse.issparse(mat)
    
    # Heuristic 1: Range check (works on sparse matrices efficiently)
    # Log-transformed: bounded max value
    if is_sparse:
        max_val = mat.max()
    else:
        max_val = np.max(mat)
    range_suggests_log = max_val < 100
    
    # Heuristic 2: Minimum nonzero value check
    # Raw counts: min nonzero ≈ 1; log1p: min nonzero ≈ log(2) ≈ 0.693
    if is_sparse:
        nonzero_data = mat.data
        if len(nonzero_data) > 0:
            min_nonzero = np.min(nonzero_data)
            min_suggests_log = min_nonzero < 1.0
        else:
            min_suggests_log = False
    else:
        nonzero_vals = mat[mat > 0]
        if len(nonzero_vals) > 0:
            min_nonzero = np.min(nonzero_vals)
            min_suggests_log = min_nonzero < 1.0
        else:
            min_suggests_log = False
    
    # Heuristic 3: Variance-to-mean ratio (efficient for sparse)
    # Raw counts (Poisson): var ≈ mean; Log-transformed: var/mean deviates
    # Only consider genes with at least 1% non-zero values for robustness
    if is_sparse:
        means_per_gene = np.array(mat.mean(axis=0)).flatten()
        # For variance: var = E[X^2] - E[X]^2
        sq_means = np.array(mat.power(2).mean(axis=0)).flatten()
        vars_per_gene = sq_means - means_per_gene ** 2
        
        # Filter genes by sparsity: keep only genes with >= 1% non-zero values
        n_cells = mat.shape[0]
        nnz_per_gene = np.array(mat.getnnz(axis=0))
        sparsity_per_gene = nnz_per_gene / n_cells
        gene_mask = sparsity_per_gene >= 0.01
        
        if np.sum(gene_mask) > 0:
            vars_filtered = vars_per_gene[gene_mask]
            means_filtered = means_per_gene[gene_mask]
            var_mean_ratio = vars_filtered / (np.abs(means_filtered) + 1e-10)
        else:
            var_mean_ratio = np.array([])
    else:
        means_per_gene = np.mean(mat, axis=0)
        vars_per_gene = np.var(mat, axis=0)
        
        # Filter genes by sparsity: keep only genes with >= 1% non-zero values
        n_cells = mat.shape[0]
        nnz_per_gene = np.sum(mat > 0, axis=0)
        sparsity_per_gene = nnz_per_gene / n_cells
        gene_mask = sparsity_per_gene >= 0.01
        
        if np.sum(gene_mask) > 0:
            vars_filtered = vars_per_gene[gene_mask]
            means_filtered = means_per_gene[gene_mask]
            var_mean_ratio = vars_filtered / (np.abs(means_filtered) + 1e-10)
        else:
            var_mean_ratio = np.array([])
    
    # Avoid division by zero
    var_mean_ratio_clean = var_mean_ratio[~np.isnan(var_mean_ratio)]
    
    if len(var_mean_ratio_clean) > 0:
        vm_median = np.median(var_mean_ratio_clean)
        # Poisson: ratio ≈ 1; log-transformed: ratio typically < 0.5 or > 3
        vm_suggests_log = (vm_median < 0.5) or (vm_median > 3)
    else:
        vm_suggests_log = False
    
    # Vote: at least 2 heuristics should agree
    votes_for_log = sum([range_suggests_log, min_suggests_log, vm_suggests_log])
    return votes_for_log >= 2
    
_SCPOLI_DEFAULT_EPOCHS = 100
_SCVI_DEFAULT_EPOCHS = 200


def _is_scvi(vae):
    """True for scvi-tools SCVI/SCANVI models (SCANVI subclasses neither, so check both)."""
    return isinstance(vae, (scvi.model.SCANVI, scvi.model.SCVI))


def _is_scpoli(vae):
    """True for scPoli models of the scpolivi package (scvi-tools reimplementation)."""
    from scpolivi import SCPOLI
    return isinstance(vae, SCPOLI)


def _check_supported(vae):
    if not (_is_scvi(vae) or _is_scpoli(vae)):
        raise RuntimeError('This VAE model is not yet supported')


def _model_class(vae):
    if isinstance(vae, scvi.model.SCANVI):
        return scvi.model.SCANVI
    if isinstance(vae, scvi.model.SCVI):
        return scvi.model.SCVI
    if _is_scpoli(vae):
        from scpolivi import SCPOLI
        return SCPOLI
    raise RuntimeError('This VAE model is not yet supported')


def _model_layer(vae):
    """Layer the model reads expression from (``None`` means ``.X``)."""
    return vae.adata_manager._registry['setup_args'].get('layer')


def load_scpoli_reference(ref_dir, adata_ref):
    """Load a pretrained scPoli reference model with scpolivi.

    Two on-disk formats are supported:

    * scvi-tools format (``model.pt``), as written by ``SCPOLI.save``;
    * scArches format (``attr.pkl``, ``model_params.pt``, ``var_names.csv``), as written by
      scArches ``scPoli.save``. It is converted on the fly with ``SCPOLI.from_scarches``
      (scArches itself is not needed).

    ``adata_ref`` is registered with the model and reordered/subset in place to the model's
    genes if necessary.
    """
    from scpolivi import SCPOLI

    if os.path.exists(os.path.join(ref_dir, 'model.pt')):
        return SCPOLI.load(ref_dir, adata_ref)

    if os.path.exists(os.path.join(ref_dir, 'attr.pkl')):
        var_names = np.atleast_1d(
            np.genfromtxt(os.path.join(ref_dir, 'var_names.csv'), delimiter=',', dtype=str)
        )
        if list(adata_ref.var_names) != list(var_names):
            missing = pd.Index(var_names).difference(adata_ref.var_names)
            if len(missing) > 0:
                raise ValueError(
                    f'{len(missing)} model genes are missing from the reference AnnData, '
                    f'e.g. {list(missing[:5])}'
                )
            _log('Reordering reference genes to match the scArches model...')
            adata_ref._inplace_subset_var(pd.Index(var_names))
        _log('Converting scArches scPoli checkpoint to scpolivi...')
        return SCPOLI.from_scarches(ref_dir, adata_ref)

    raise FileNotFoundError(
        f'No scPoli model found in {ref_dir} (expected model.pt or attr.pkl).'
    )


def _set_query_conditions(adata, vae, col_batch):
    """Copy the query batch column into the model's batch/condition column(s)."""
    if not col_batch:
        return
    if _is_scvi(vae):
        ref_batch_key = vae.adata_manager._registry['setup_args']['batch_key']
        if ref_batch_key is not None:
            adata.obs[ref_batch_key] = adata.obs[col_batch].copy()
    else:
        for cond_key in vae.condition_keys_:
            if cond_key != col_batch:
                adata.obs[cond_key] = adata.obs[col_batch].copy()


def _mask_cell_type_columns(adata, vae):
    """Set the scPoli model's cell-type columns of the query to the model's unknown label.

    Query cells are mapped as unlabeled, so any query annotation that happens to share a
    column name with the reference (e.g. ``class``) must not be used. Returns the original
    columns so that they can be restored with :func:`_restore_columns`.
    """
    if not _is_scpoli(vae) or not vae.cell_type_keys_:
        return {}
    unknown = vae.unknown_ct_names_[0] if vae.unknown_ct_names_ else np.nan
    saved = {k: (adata.obs[k].copy() if k in adata.obs else None) for k in vae.cell_type_keys_}
    for k in vae.cell_type_keys_:
        adata.obs[k] = pd.Series(unknown, index=adata.obs_names, dtype=object)
    return saved


def _restore_columns(adata, saved):
    for k, col in saved.items():
        if col is None:
            adata.obs.drop(columns=k, inplace=True, errors='ignore')
        else:
            adata.obs[k] = col


def _warn_shared_conditions(adata, vae):
    """Warn if query batch names coincide with reference ones (their embedding is reused)."""
    if not _is_scpoli(vae):
        return
    for cond_key, known in vae.conditions_.items():
        shared = sorted(set(adata.obs[cond_key].astype(str)) & set(map(str, known)))
        if shared:
            warnings.warn(
                f'Query values of "{cond_key}" also exist in the reference model ({shared[:5]}'
                f'{"..." if len(shared) > 5 else ""}); these query cells reuse the reference '
                'condition embedding instead of getting a new one. Rename the query batches '
                'if they are unrelated to the reference samples.',
                UserWarning,
                stacklevel=2,
            )
            _log(f'WARNING: query batches shared with the reference: {shared}')


def _prepare_query_anndata(adata, vae):
    """Pad missing genes with zeros and reorder them to the reference model (in place).

    Uses scvi-tools' ``prepare_query_anndata`` for all supported models (scpolivi's SCPOLI
    inherits it), which also warns when less than 80% of the reference genes are present.
    """
    _model_class(vae).prepare_query_anndata(adata, vae)


def _ensure_csr(adata, vae, name, verbose=True):
    """Convert the expression matrix the model reads (``.X`` or its layer) to CSR in place.

    Minibatches are rows (cells); slicing rows of a CSC matrix is ~30x slower than of a CSR
    one and then dominates training and inference time (the GPU mostly waits for data).
    scvi-tools only warns about this, and map2ref silences warnings.
    """
    layer = _model_layer(vae)
    mat = adata.X if layer is None else adata.layers[layer]
    if not scipy.sparse.issparse(mat) or mat.format == 'csr':
        return
    if verbose:
        _log(f'Converting the {name} expression matrix ({"X" if layer is None else layer}) '
             f'from {mat.format.upper()} to CSR for fast minibatch loading...')
    mat = csr_matrix(mat)
    if layer is None:
        adata.X = mat
    else:
        adata.layers[layer] = mat


def train_scarches(adata,
                   ref_adata,
                   vae,
                   col_batch = None,
                   batch_size = 1024,
                   epochs = None,
                   keep_original_adata = False,
                   skip_scale_check = False,
                   verbose = True):
    """Map query cells onto a reference model with scArches-style architecture surgery.

    ``epochs=None`` uses the model's default: 200 for SCVI/SCANVI and 100 for scPoli (the
    scPoli default, as in scArches).
    """
    if verbose:
        _log('Preparing data and model...')

    _check_supported(vae)
    model = _model_class(vae)

    if keep_original_adata:
        ref_idx = adata.var_names.isin(ref_adata.var_names)
        adata = adata[:, ref_idx]
        adata.varm = dict()

    # Determine if reference is log-normalized
    used_layer = _model_layer(vae)
    r_x_log = is_lognorm(ref_adata.X)
    if used_layer is None:
        expected_log = r_x_log
    else:
        expected_log = is_lognorm(ref_adata.layers[used_layer])
    if verbose:
        _log('Data expected to be in log: ' + str(expected_log))
    # make X a copy of the expected layer in ref
    if used_layer is not None and r_x_log != expected_log:
        ref_adata.layers['_raw_X'] = ref_adata.X.copy()
        ref_adata.X = ref_adata.layers[used_layer].copy()

    # make sure the query data share the same scale as ref
    if not skip_scale_check:
        if used_layer and used_layer in adata.layers.keys():
            q_layer_log = is_lognorm(adata.layers[used_layer])
            if q_layer_log != expected_log:
                raise RuntimeError('The query data and reference data do not have the same expression scale (raw counts vs. log-normalized)')
            adata.layers['_raw_X'] = adata.X.copy()
            adata.X = adata.layers[used_layer].copy()
        else:
            q_x_log = is_lognorm(adata.X)
            if q_x_log != expected_log:
                raise RuntimeError('The query data and reference data do not have the same expression scale (raw counts vs. log-normalized)')
            if used_layer:
                adata.layers[used_layer] = adata.X.copy()
    else:
        if verbose:
            _log('Skipping expression scale sanity check as requested')
        if used_layer and used_layer in adata.layers.keys():
            adata.layers['_raw_X'] = adata.X.copy()
            adata.X = adata.layers[used_layer].copy()
        elif used_layer:
            adata.layers[used_layer] = adata.X.copy()

    _set_query_conditions(adata, vae, col_batch)
    _prepare_query_anndata(adata, vae)
    _ensure_csr(adata, vae, 'query', verbose=verbose)

    if _is_scpoli(vae):
        _warn_shared_conditions(adata, vae)
        # labeled_indices=[]: query labels (if any) are not used; all query cells are unlabeled
        saved_ct = _mask_cell_type_columns(adata, vae)
        try:
            vae_q = model.load_query_data(adata, vae, labeled_indices=[])
            n_epochs = _SCPOLI_DEFAULT_EPOCHS if epochs is None else epochs
            if verbose:
                _log(f'Fitting model ({n_epochs} epochs)...')
            vae_q.train(n_epochs=n_epochs, batch_size=batch_size)
        finally:
            _restore_columns(adata, saved_ct)
    else:
        vae_q = model.load_query_data(adata, vae)
        n_epochs = _SCVI_DEFAULT_EPOCHS if epochs is None else epochs
        if verbose:
            _log(f'Fitting model ({n_epochs} epochs)...')
        vae_q.train(
            batch_size=batch_size,
            max_epochs=n_epochs,
            plan_kwargs=dict(weight_decay=0.0),
        )

    if verbose:
        _log('scArches model training is done.')

    return vae_q


_LATENT_BATCH_SIZE = 4096


def _latent(vae, adata):
    # Inference is per cell (eval mode), so a large minibatch only speeds things up;
    # scvi-tools' default of 128 is very slow for references with millions of cells.
    old_bs = scvi.settings.batch_size
    scvi.settings.batch_size = _LATENT_BATCH_SIZE
    try:
        if _is_scpoli(vae):
            return vae.get_latent(adata, mean=True)
        return vae.get_latent_representation(adata, batch_size=_LATENT_BATCH_SIZE)
    finally:
        scvi.settings.batch_size = old_bs


def get_latent_space(adata,
                     vae,
                     ref_adata = None,
                     col_batch = None,
                     ref_annot_unknown = True
                    ):
    """Latent representation of the query (and reference) cells with the query model.

    Query and reference are embedded separately (the encoders work per cell), which avoids
    concatenating the - often very large - reference with the query. ``.obs`` of both objects
    is restored afterwards.
    """
    _check_supported(vae)
    obs_q = adata.obs.copy()
    if ref_adata is not None:
        obs_ref = ref_adata.obs.copy()

    try:
        _set_query_conditions(adata, vae, col_batch)
        _mask_cell_type_columns(adata, vae)  # undone by restoring obs below
        if isinstance(vae, scvi.model.SCANVI):
            ref_annot_key = vae.adata_manager._registry["setup_args"]["labels_key"]
            label_unknown = vae.adata_manager._registry["setup_args"]["unlabeled_category"]
            adata.obs[ref_annot_key] = label_unknown
            if ref_adata is not None:
                if ref_annot_unknown:
                    ref_adata.obs[ref_annot_key] = label_unknown
                else:
                    ref_adata.obs[ref_annot_key] = ref_adata.obs[ref_annot_key].tolist()

        _ensure_csr(adata, vae, 'query')
        if ref_adata is not None:
            _ensure_csr(ref_adata, vae, 'reference')
        lat_rep_q = _latent(vae, adata)
        lat_rep_ref = _latent(vae, ref_adata) if ref_adata is not None else None
    finally:
        adata.obs = obs_q
        if ref_adata is not None:
            ref_adata.obs = obs_ref

    if ref_adata is None:
        return lat_rep_q
    return {'ref' : lat_rep_ref, 'query' : lat_rep_q}

def cmd_interface():
    parser = argparse.ArgumentParser(description="Run scARCHES training to map query data to the pre-trained autoencoder")
    
    parser.add_argument(
        "-q","--h5ad",
        type=str,
        dest="H5AD",
        required=True,
        help='H5AD file of the query data',
    )
    parser.add_argument(
        "-r","--ref_h5ad",
        type=str,
        dest="REF_H5AD",
        required=True,
        help='H5AD file of the reference data'
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        dest="model",
        required=True,
        help='Location of the pre-trained VAE model given the reference data'
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        dest="output",
        default="output",
        help='Path to the output folder (default: ./output)'
    )
    parser.add_argument(
        "--model_type",
        type=str,
        dest="model_type",
        default='scanvi',
        help='Type of the VAE model. Should be one of SCVI, SCANVI, and SCPOLI (default: SCANVI)'
    )
    parser.add_argument(
        "--query_batch_key",
        type=str,
        dest="query_batch_key",
        default='batch',
        help='Batch/sample variable of the query data (default: batch)'
    )
    parser.add_argument(
        "--epochs",
        type=int,
        dest="epochs",
        default=None,
        help='Epochs in scARCHES model training (default: 200 for SCVI/SCANVI, 100 for SCPOLI)'
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        dest="batch_size",
        default=1024,
        help='Size of mini-batch in scARCHES model training (default 1024)'
    )
    parser.add_argument(
        "--save_anndata",
        action="store_true",
        dest="save_anndata",
        help='Save annData data when saving the scARCHES model'
    )
    parser.add_argument(
        "--use_ref_annot",
        action="store_true",
        dest="use_ref_annot",
        help='When getting the latent representation of the reference, take into account the reference annotation labels instead of converting to unknown'
    )
    parser.add_argument(
        "--skip-scale-check",
        action="store_true",
        dest="skip_scale_check",
        help='Skip expression scale sanity check between query and reference (use if detection fails)'
    )
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = cmd_interface()
    print(args)
    
    if args.model_type.upper() == 'SCANVI':
        model = scvi.model.SCANVI
    elif args.model_type.upper() == 'SCVI':
        model = scvi.model.SCVI
    elif args.model_type.upper() == 'SCPOLI':
        model = None  # loaded with scpolivi below
    else:
        raise RuntimeError('The VAE model type is not yet supported. Should be one of SCANVI, SCVI and SCPOLI')
    
    _log('Reading data...')
    adata = sc.read(args.H5AD)
    ref_adata = sc.read(args.REF_H5AD)
    
    if model is None:
        vae = load_scpoli_reference(args.model, ref_adata)
    else:
        vae = model.load(args.model, ref_adata)
    vae_q = train_scarches(adata, ref_adata, vae = vae, col_batch = args.query_batch_key, batch_size = args.batch_size, epochs = args.epochs, keep_original_adata = False, skip_scale_check = args.skip_scale_check, verbose = True)
    
    _log('Saving model and params...')
    os.makedirs(args.output, exist_ok=True)
    vae_q.save(
        args.output,
        overwrite=True,
        save_anndata=args.save_anndata,
    )
    
    with open(os.path.join(args.output, "params.txt"), "w") as f:
        f.write(str(args))
    
    _log('Saving latent representations...')
    lat_reps = get_latent_space(adata, vae_q, ref_adata, col_batch = args.query_batch_key, ref_annot_unknown = not args.use_ref_annot)
    lat_rep_ref, lat_rep_q = (lat_reps['ref'], lat_reps['query'])
    np.save(os.path.join(args.output, "lat_rep_ref.npy"), lat_rep_ref)
    np.save(os.path.join(args.output, "lat_rep_query.npy"), lat_rep_q)
