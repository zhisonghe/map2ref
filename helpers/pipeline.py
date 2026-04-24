import os

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

from helpers.mapping_scarches import train_scarches, get_latent_space
from helpers.wknn import estimate_presence_score, transfer_labels
from helpers.report import generate_mapping_report
from helpers.log import get_logger

_log = get_logger('pipeline')


def run_mapping(args, load_vae, label_config, ref_annot_labs, post_label_hook=None):
    """Execute the full reference-mapping pipeline.

    Parameters
    ----------
    args:
        Parsed argparse namespace (from cmd_interface).
    load_vae : callable
        A function ``(adata_ref) -> vae`` that loads the reference model.
        Receives the already-loaded reference AnnData so the file is only
        read once.
    label_config : list of dict
        Each entry describes one standard label-transfer step::

            {
                'key':      str,   # obs column in adata_ref to transfer
                'obs_col':  str,   # destination obs column in adata_query
                'tsv':      str,   # filename for the saved TSV
                'report_key': str, # key used in the HTML report
            }

    ref_annot_labs : list of str
        Reference obs columns to plot on the reference UMAP panel.
    post_label_hook : callable or None
        Optional function called after the standard label-transfer loop.
        Signature::

            hook(adata_ref, adata_query, wknn, output_dir) -> dict

        The returned dict maps report_key -> DataFrame (with at least
        'best_label' and 'best_score' columns) and is merged into
        df_labels before the report is generated.  The hook is also
        responsible for writing any extra TSVs and populating obs columns
        on adata_query.
    """
    verbose = not args.quiet

    if args.query is None:
        raise ValueError('--query/-q is required unless --report-only is used')

    # ------------------------------------------------------------------ paths
    file_ref_adata      = os.path.join(args.ref, 'ref.h5ad')
    file_lat_ref        = os.path.join(args.output, 'lat_rep_ref.npy')
    file_lat_query      = os.path.join(args.output, 'lat_rep_query.npy')
    file_wknn           = os.path.join(args.output, 'wknn.npz')
    file_presence_score = os.path.join(args.output, 'presence_scores.tsv')
    file_q_adata2save   = os.path.join(args.output, 'query_processed.h5ad')
    loc_scarches_model  = os.path.join(args.output, 'model')
    loc_report          = os.path.join(args.output, 'report')

    # ------------------------------------------------------------ load data
    if verbose:
        _log('Loading data...')
    adata_ref = sc.read(file_ref_adata)
    adata_query = sc.read(args.query)
    if args.save_full_query:
        adata_query_full = adata_query.copy()

    # --------------------------------------------------------- model training
    if verbose:
        _log('scArches model training...')
    vae = load_vae(adata_ref)

    # ---------------------- optional query layer remapping
    if args.query_layer is not None:
        # detect target slot via duck-typing: SCANVI/SCVI expose adata_manager with setup_args
        try:
            used_layer = vae.adata_manager._registry["setup_args"]["layer"]
        except AttributeError:
            used_layer = None  # scPoli: always uses .X
        if used_layer is None:
            # model expects .X; only copy if the user pointed at a different layer
            if args.query_layer != 'X':
                if verbose:
                    _log(f'Copying query layer "{args.query_layer}" to .X')
                adata_query.X = adata_query.layers[args.query_layer].copy()
        else:
            # model expects a named layer; only copy if the user pointed at a different source
            if args.query_layer != used_layer:
                src = adata_query.X if args.query_layer == 'X' else adata_query.layers[args.query_layer]
                if verbose:
                    _log(f'Copying query layer "{args.query_layer}" to layer "{used_layer}"')
                adata_query.layers[used_layer] = src.copy()

    vae_q = train_scarches(
        adata_query, adata_ref,
        vae=vae,
        col_batch=args.query_batch_key,
        batch_size=args.batch_size,
        epochs=args.epochs,
        keep_original_adata=False,
        skip_scale_check=args.skip_scale_check,
        verbose=verbose,
    )

    if verbose:
        _log('Saving the model...')
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(loc_scarches_model, exist_ok=True)
    vae_q.save(loc_scarches_model, overwrite=True)

    # ------------------------------------------------- latent representation
    if verbose:
        _log('Generating latent representations...')
    lat_reps = get_latent_space(adata_query, vae_q, adata_ref, col_batch=args.query_batch_key)
    lat_rep_ref, lat_rep_q = lat_reps['ref'], lat_reps['query']

    if verbose:
        _log('Saving latent representations...')
    np.save(file_lat_ref, lat_rep_ref)
    np.save(file_lat_query, lat_rep_q)
    adata_ref.obsm['X_scarches'] = lat_rep_ref
    adata_query.obsm['X_scarches'] = lat_rep_q

    # ---------------------------------------------------- presence scores
    if verbose:
        _log('Calculating presence scores per query data set...')
    presence = estimate_presence_score(
        adata_ref, adata_query,
        use_rep_ref_wknn='X_scarches',
        use_rep_query_wknn='X_scarches',
        k_wknn=args.k_wknn,
        k_ref_trans_prop=args.k_ref,
        split_by=args.query_batch_key,
        do_random_walk=not args.no_smooth_presence,
    )
    max_presence, group_presence = presence['max'], presence['per_group']

    if verbose:
        _log('Saving wknn and presence scores...')
    wknn = presence['wknn']
    sparse.save_npz(file_wknn, wknn)
    df_presence = pd.concat(
        [max_presence, group_presence], axis=1
    ).set_axis(['max'] + list(group_presence.columns), axis=1)
    df_presence.to_csv(file_presence_score, sep='\t')

    # ---------------------------------------------------- label transfer
    df_labels = None
    if not args.no_lab_transfer:
        keys_str = ', '.join(cfg['key'] for cfg in label_config)
        _log(f'Transferring labels of {keys_str}...')

        df_labels = {}
        for cfg in label_config:
            df = transfer_labels(adata_ref, adata_query, wknn, label_key=cfg['key'])
            df_labels[cfg['report_key']] = df
            adata_query.obs[cfg['obs_col']] = df['best_label'].copy()

        _log('Saving transferred labels...')
        for cfg in label_config:
            df_labels[cfg['report_key']].to_csv(
                os.path.join(args.output, cfg['tsv']), sep='\t'
            )

        if post_label_hook is not None:
            extra = post_label_hook(adata_ref, adata_query, wknn, args.output)
            df_labels.update(extra)

    # ------------------------------------------------ UMAP for visualization
    if verbose:
        _log('Preparing embeddings for visualization...')
    if 'X_umap' not in adata_ref.obsm.keys():
        sc.pp.neighbors(adata_ref, use_rep='X_scarches')
        sc.tl.umap(adata_ref)
    if args.force_new_umap or (args.vis_rep_query not in adata_query.obsm.keys()):
        sc.pp.neighbors(adata_query, use_rep='X_scarches')
        sc.tl.umap(adata_query)
        args.vis_rep_query = 'X_umap'

    # --------------------------------------------------- save query h5ad
    if verbose:
        _log('Saving query data...')
    adata_query.write_h5ad(os.path.join(args.output, 'query.h5ad'))

    if args.save_full_query:
        if verbose:
            _log('Saving full query anndata (all genes + results)...')
        adata_query_full.obs = adata_query.obs.copy()
        adata_query_full.obsm['X_scarches'] = adata_query.obsm['X_scarches']
        adata_query_full.obsm[args.vis_rep_query] = adata_query.obsm[args.vis_rep_query]
        adata_query_full.write_h5ad(file_q_adata2save)

    # -------------------------------------------------------- HTML report
    if verbose:
        _log('Generating HTML report...')
    generate_mapping_report(
        adata_ref,
        adata_query,
        df_presence,
        df_labels=df_labels,
        ref_annot_labs=ref_annot_labs,
        vis_rep_query=args.vis_rep_query,
        output=loc_report,
        report_type=args.report_type,
        verbose=verbose,
    )

    if verbose:
        _log('Done.')


def run_report_only(args, label_config, ref_annot_labs, extra_label_config=None):
    """Re-generate the HTML report from existing output files.

    Loads ref.h5ad from the reference model directory, query.h5ad and
    presence_scores.tsv from the output folder, and reconstructs df_labels
    from the label-transfer TSVs listed in label_config / extra_label_config.

    Parameters
    ----------
    args:
        Parsed argparse namespace.
    label_config : list of dict
        Same format as used in run_mapping (keys: 'tsv', 'report_key',
        optionally 'obs_col' used as fallback when the TSV lacks a
        'best_label' column).
    ref_annot_labs : list of str
        Reference obs columns to plot on the reference UMAP panel.
    extra_label_config : list of dict or None
        Additional TSVs produced by post_label_hook (e.g. Region_hier).
        Same dict format as label_config.
    """
    verbose = not args.quiet

    file_ref_adata      = os.path.join(args.ref, 'ref.h5ad')
    file_query_adata    = os.path.join(args.output, 'query.h5ad')
    file_presence_score = os.path.join(args.output, 'presence_scores.tsv')
    loc_report          = os.path.join(args.output, 'report')

    if verbose:
        _log('Loading existing results for report...')
    adata_ref   = sc.read(file_ref_adata)
    adata_query = sc.read(file_query_adata)

    # presence_scores.tsv has a 'max' column plus one column per query group
    presence = pd.read_csv(file_presence_score, sep='\t', index_col=0)

    df_labels = None
    if not args.no_lab_transfer:
        df_labels = {}
        all_label_cfg = list(label_config) + (list(extra_label_config) if extra_label_config else [])
        for cfg in all_label_cfg:
            tsv_path = os.path.join(args.output, cfg['tsv'])
            if os.path.exists(tsv_path):
                df = pd.read_csv(tsv_path, sep='\t', index_col=0)
                if 'best_label' not in df.columns:
                    # TSV was saved from a Series or lacks score columns;
                    # reconstruct a minimal DataFrame from the obs column.
                    obs_col = cfg.get('obs_col')
                    if obs_col and obs_col in adata_query.obs.columns:
                        df = pd.DataFrame(
                            {'best_label': adata_query.obs[obs_col], 'best_score': 1.0},
                            index=adata_query.obs_names,
                        )
                    else:
                        df = pd.DataFrame(
                            {'best_label': df.iloc[:, 0], 'best_score': 1.0},
                            index=df.index,
                        )
                df_labels[cfg['report_key']] = df
        if not df_labels:
            df_labels = None

    if verbose:
        _log('Generating HTML report...')
    generate_mapping_report(
        adata_ref,
        adata_query,
        presence,
        df_labels=df_labels,
        ref_annot_labs=ref_annot_labs,
        vis_rep_query=args.vis_rep_query,
        output=loc_report,
        report_type=args.report_type,
        verbose=verbose,
    )

    if verbose:
        _log('Done.')
