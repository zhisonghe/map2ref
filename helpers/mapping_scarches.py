import warnings

warnings.filterwarnings("ignore")

import os
import sys
import argparse

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import scipy
from scipy.sparse import csr_matrix
import scvi

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
    
_MIN_VAR_NAME_RATIO = 0.8


def _prepare_query_anndata(adata, vae):
    """Prepare query AnnData in-place for scArches query integration.

    For SCANVI/SCVI: delegates to the model class's ``prepare_query_anndata``
    method, which pads missing genes with zeros, reorders features to match the
    reference, and calls ``setup_anndata`` so that ``load_query_data`` can be
    called immediately afterwards.

    For scPoli: mirrors the same gene-level preparation manually (the built-in
    ``_validate_var_names`` is only invoked when passing a *path* to
    ``load_query_data``, not a model object) and additionally fills every
    cell-type obs column with the appropriate unknown label so that query cells
    are treated as unlabeled.
    """
    if isinstance(vae, scvi.model._scanvi.SCANVI):
        scvi.model.SCANVI.prepare_query_anndata(adata, vae)
        return
    if isinstance(vae, scvi.model._scvi.SCVI):
        scvi.model.SCVI.prepare_query_anndata(adata, vae)
        return

    import scarches
    if not isinstance(vae, scarches.models.scpoli.scPoli):
        raise RuntimeError('This VAE model is not yet supported')

    # --- 1. Gene subsetting / padding / reordering ---
    var_names = pd.Index(vae.adata.var_names)

    intersection = adata.var_names.intersection(var_names)
    if len(intersection) == 0:
        raise ValueError(
            'No reference var names found in query data. '
            'Please check that the gene identifiers match the reference.'
        )
    ratio = len(intersection) / len(var_names)
    if ratio < _MIN_VAR_NAME_RATIO:
        warnings.warn(
            f'Query data contains less than {_MIN_VAR_NAME_RATIO:.0%} of '
            f'reference var names ({ratio:.1%}). '
            'This may result in poor performance.',
            UserWarning,
            stacklevel=2,
        )

    genes_to_add = var_names.difference(adata.var_names)
    if len(genes_to_add) > 0:
        padding_mtx = csr_matrix(np.zeros((adata.n_obs, len(genes_to_add))))
        adata_padding = anndata.AnnData(
            X=padding_mtx.copy(),
            layers={layer: padding_mtx.copy() for layer in adata.layers},
        )
        adata_padding.var_names = genes_to_add
        adata_padding.obs_names = adata.obs_names
        adata_out = anndata.concat(
            [adata, adata_padding],
            axis=1,
            join='outer',
            index_unique=None,
            merge='unique',
        )
    else:
        adata_out = adata

    # Subset/reorder to exactly the reference var order
    if not var_names.equals(adata_out.var_names):
        adata_out._inplace_subset_var(var_names)

    if adata_out is not adata:
        adata._init_as_actual(adata_out)

    # --- 2. Fill cell-type obs columns with unknown label ---
    if vae.cell_type_keys_ is not None:
        unknown_label = vae.unknown_ct_names_[0] if vae.unknown_ct_names_ else 'Unknown'
        for ct_key in vae.cell_type_keys_:
            adata.obs[ct_key] = unknown_label


def train_scarches(adata,
                   ref_adata,
                   vae,
                   col_batch = None,
                   batch_size = 1024,
                   epochs = 500,
                   keep_original_adata = False,
                   skip_scale_check = False,
                   verbose = True):    
    if verbose:
        print('[PROGRESS] Preparing data and model...')
    
    if keep_original_adata:
        ref_idx = adata.var_names.isin(ref_adata.var_names)
        adata = adata[:, ref_idx]
        adata.varm = dict()
    
    if isinstance(vae, scvi.model._scanvi.SCANVI):
        model = scvi.model.SCANVI
    elif isinstance(vae, scvi.model._scvi.SCVI):
        model = scvi.model.SCVI
    else:
        import scarches
        if isinstance(vae, scarches.models.scpoli.scPoli):
            model = scarches.models.scPoli
        else:
            raise RuntimeError('This VAE model is not yet supported')
    
    # Determine if reference is log-normalized
    r_x_log = is_lognorm(ref_adata.X)
    if (isinstance(vae, scvi.model._scanvi.SCANVI) or isinstance(vae, scvi.model._scvi.SCVI)):
        used_layer = vae.adata_manager._registry["setup_args"]["layer"]
        if used_layer is None:
            expected_log = r_x_log
        else:
            expected_log = is_lognorm(ref_adata.layers[used_layer])
        print('[PROGRESS] Data expected to be in log: ' + str(expected_log))
        # make X a copy of the expected layer in ref
        if r_x_log != expected_log:
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
                print('[PROGRESS] Skipping expression scale sanity check as requested')
            if used_layer and used_layer in adata.layers.keys():
                adata.layers['_raw_X'] = adata.X.copy()
                adata.X = adata.layers[used_layer].copy()
            elif used_layer:
                adata.layers[used_layer] = adata.X.copy()
    else:
        print('[PROGRESS] Data expected to be in log: ' + str(r_x_log))
        if not skip_scale_check:
            q_x_log = is_lognorm(adata.X)
            if q_x_log != r_x_log:
                raise RuntimeError('The query data and reference data do not have the same expression scale (raw counts vs. log-normalized)')
        else:
            if verbose:
                print('[PROGRESS] Skipping expression scale sanity check as requested')
    
    if col_batch:
        if isinstance(vae, scvi.model._scanvi.SCANVI) or isinstance(vae, scvi.model._scvi.SCVI):
            ref_batch_key = vae.adata_manager._registry["setup_args"]["batch_key"]
            adata.obs[ref_batch_key] = adata.obs[col_batch].copy()
        else:
            import scarches
            if isinstance(vae, scarches.models.scpoli.scPoli):
                # scPoli: condition_keys_ is a list of condition obs columns used during training
                for cond_key in vae.condition_keys_:
                    if cond_key != col_batch:
                        adata.obs[cond_key] = adata.obs[col_batch].copy()
    
    _prepare_query_anndata(adata, vae)
    vae_q = model.load_query_data(adata, vae)
    
    if verbose:
        print('[PROGRESS] Fitting model...')
    
    vae_q.train(
        batch_size=batch_size,
        max_epochs=epochs,
        plan_kwargs=dict(weight_decay=0.0),
    )
    
    if verbose:
        print('[PROGRESS] The scARCHES model training is done.')
    
    return vae_q

def get_latent_space(adata,
                     vae,
                     ref_adata = None,
                     col_batch = None,
                     ref_annot_unknown = True
                    ):
    obs_q = adata.obs.copy()
    if ref_adata:
        obs_ref = ref_adata.obs.copy()
    
    if col_batch:
        if isinstance(vae, scvi.model._scanvi.SCANVI) or isinstance(vae, scvi.model._scvi.SCVI):
            ref_batch_key = vae.adata_manager._registry["setup_args"]["batch_key"]
            adata.obs[ref_batch_key] = adata.obs[col_batch].copy()
        else:
            import scarches
            if isinstance(vae, scarches.models.scpoli.scPoli):
                # scPoli: condition_keys_ is a list of condition obs columns used during training
                for cond_key in vae.condition_keys_:
                    if cond_key != col_batch:
                        adata.obs[cond_key] = adata.obs[col_batch].copy()
    if isinstance(vae, scvi.model._scanvi.SCANVI):
        ref_annot_key = vae.adata_manager._registry["setup_args"]["labels_key"]
        label_unknown = vae.adata_manager._registry["setup_args"]["unlabeled_category"]
        adata.obs[ref_annot_key] = label_unknown
        if ref_adata:
            if ref_annot_unknown:
                ref_adata.obs[ref_annot_key] = label_unknown
            else:
                ref_adata.obs[ref_annot_key] = ref_adata.obs[ref_annot_key].tolist()
    
    if ref_adata is None:
        adata_full = adata
    else:
        adata.obs['ref'] = 'query'
        ref_adata.obs['ref'] = 'ref'
        adata_full = adata.concatenate(ref_adata)
        adata_full.obs['batch'] = adata.obs["batch"].tolist() + ref_adata.obs["batch"].tolist()
    
    lat_rep = vae.get_latent_representation(adata_full)
    
    adata.obs = obs_q
    if ref_adata:
        ref_adata.obs = obs_ref
    
    if ref_adata is None:
        return lat_rep
    else:
        lat_rep_ref = lat_rep[adata_full.obs['ref'] == 'ref',:]
        lat_rep_q = lat_rep[adata_full.obs['ref'] == 'query',:]
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
        default=500,
        help='Epochs in scARCHES model training (default: 500)'
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
        import scarches
        model = scarches.models.scPoli
    else:
        raise RuntimeError('The VAE model type is not yet supported. Should be one of SCANVI, SCVI and SCPOLI')
    
    print("[PROGRESS] Reading data...")
    adata = sc.read(args.H5AD)
    ref_adata = sc.read(args.REF_H5AD)
    
    vae = model.load(args.model, ref_adata)
    vae_q = train_scarches(adata, ref_adata, vae = vae, col_batch = args.query_batch_key, batch_size = args.batch_size, epochs = args.epochs, keep_original_adata = False, skip_scale_check = args.skip_scale_check, verbose = True)
    
    print("[PROGRESS] Saving model and params...")
    os.makedirs(args.output, exist_ok=True)
    vae_q.save(
        args.output,
        overwrite=True,
        save_anndata=args.save_anndata,
    )
    
    with open(os.path.join(args.output, "params.txt"), "w") as f:
        f.write(str(args))
    
    print("[PROGRESS] Saving latent representations...")
    lat_reps = get_latent_space(adata, vae_q, ref_adata, col_batch = args.query_batch_key, ref_annot_unknown = not args.use_ref_annot)
    lat_rep_ref, lat_rep_q = (lat_reps['ref'], lat_reps['query'])
    np.save(os.path.join(args.output, "lat_rep_ref.npy"), lat_rep_ref)
    np.save(os.path.join(args.output, "lat_rep_query.npy"), lat_rep_q)
