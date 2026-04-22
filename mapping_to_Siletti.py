import warnings
warnings.filterwarnings("ignore")

import os
import argparse

_DEFAULT_REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'model_Siletti')


def cmd_interface():
    parser = argparse.ArgumentParser(description="Do mapping of the provided data to the adult human brain scRNA-seq atlas (Siletti et al. 2023)")
    parser.add_argument(
        '-r', '--ref',
        type=str,
        dest='ref',
        default=_DEFAULT_REF,
        help='Location of the reference model directory (default: models/model_Siletti relative to this script)',
    )
    parser.add_argument(
        "-q","--query",
        type=str,
        dest="query",
        required=True,
        help='H5AD file of the query data',
    )
    parser.add_argument(
        '-o','--output',
        type=str,
        dest='output',
        default='output',
        help='Path to the output folder (default: ./output)'
    )
    parser.add_argument(
        '--save-full-query',
        action='store_true',
        dest='save_full_query',
        help='Save the full query anndata with the mapping results'
    )
    parser.add_argument(
        '-b','--query-batch-key',
        type=str,
        dest='query_batch_key',
        default='batch',
        help='Metadata key of batches in the query data (default: batch)',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        dest='batch_size',
        default=1024,
        help='Size of mini-batch in scPoli model training (default: 1024)'
    )
    parser.add_argument(
        "--epochs",
        type=int,
        dest='epochs',
        default=200,
        help="Epochs in scPoli model training (default: 200)"
    )
    parser.add_argument(
        "-k", "--k-wknn",
        type=int,
        dest="k_wknn",
        default=100,
        help="Number of NNs per query cell in reference (default: 100)"
    )
    parser.add_argument(
        "-n", "--k-ref",
        type=int,
        dest="k_ref",
        default=100,
        help="Number of NNs per reference cell in reference (default: 100)"
    )
    parser.add_argument(
        "--no-smooth-presence",
        action="store_true",
        dest="no_smooth_presence",
        help="Skip the random-walk-based smoothening of presence scores"
    )
    parser.add_argument(
        "--no-label-transfer",
        action='store_true',
        dest='no_lab_transfer',
        help='Skip label transfer for ROI groups, cell type and supercluster'
    )
    parser.add_argument(
        '--vis-rep-query',
        type=str,
        dest='vis_rep_query',
        default='X_umap',
        help='Embedding in the query data for visualization (default: X_umap)'
    )
    parser.add_argument(
        '--force-new-umap',
        action='store_true',
        dest='force_new_umap',
        help='Always make a new UMAP of the query data for visualization using the projected latent representation'
    )
    parser.add_argument(
        '--report-type',
        type=str,
        dest='report_type',
        default='basic',
        choices=['basic', 'fancy'],
        help='HTML report style to generate (default: basic)'
    )
    parser.add_argument(
        '--skip-scale-check',
        action='store_true',
        dest='skip_scale_check',
        help='Skip expression scale sanity check between query and reference (use if detection fails)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        dest='quiet',
        help='Quiet run without verbose message'
    )

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = cmd_interface()

    import scanpy as sc
    import numpy as np
    import pandas as pd
    import scarches
    import matplotlib.pyplot as plt
    from scipy import sparse
    from helpers.mapping_scarches import train_scarches, get_latent_space
    from helpers.wknn import estimate_presence_score, transfer_labels
    from helpers.report import generate_mapping_report

    verbose = not args.quiet
    file_ref_adata = os.path.join(args.ref, 'ref.h5ad')
    file_lat_ref = os.path.join(args.output, "lat_rep_ref.npy")
    file_lat_query = os.path.join(args.output, "lat_rep_query.npy")
    file_wknn = os.path.join(args.output, 'wknn.npz')
    file_presence_score = os.path.join(args.output, 'presence_scores.tsv')
    file_q_adata2save = os.path.join(args.output, 'query_processed.h5ad')
    loc_scarches_model = os.path.join(args.output, 'model')
    loc_report = os.path.join(args.output, 'report')

    if verbose:
        print('[PROGRESS] Loading data...')
    adata_ref = sc.read(file_ref_adata)
    adata_query = sc.read(args.query)
    if args.save_full_query:
        adata_query_full = adata_query.copy()

    if verbose:
        print('[PROGRESS] scPoli model training...')
    vae = scarches.models.scPoli.load(args.ref, adata_ref)
    vae_q = train_scarches(adata_query, adata_ref, vae=vae, col_batch=args.query_batch_key, batch_size=args.batch_size, epochs=args.epochs, keep_original_adata=False, skip_scale_check=args.skip_scale_check, verbose=not args.quiet)

    if verbose:
        print('[PROGRESS] Saving the model...')
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(loc_scarches_model, exist_ok=True)
    vae_q.save(loc_scarches_model, overwrite=True)

    if verbose:
        print('[PROGRESS] Generating latent representations...')
    lat_reps = get_latent_space(adata_query, vae_q, adata_ref, col_batch=args.query_batch_key)
    lat_rep_ref, lat_rep_q = (lat_reps['ref'], lat_reps['query'])

    if verbose:
        print('[PROGRESS] Saving latent representations...')
    np.save(file_lat_ref, lat_rep_ref)
    np.save(file_lat_query, lat_rep_q)
    adata_ref.obsm['X_scarches'] = lat_rep_ref
    adata_query.obsm['X_scarches'] = lat_rep_q

    if verbose:
        print('[PROGRESS] Calculating presence scores per query data set...')
    presence = estimate_presence_score(adata_ref, adata_query, use_rep_ref_wknn='X_scarches', use_rep_query_wknn='X_scarches', k_wknn=args.k_wknn, k_ref_trans_prop=args.k_ref, split_by=args.query_batch_key, do_random_walk=not args.no_smooth_presence)
    max_presence, group_presence = (presence['max'], presence['per_group'])

    if verbose:
        print('[PROGRESS] Saving wknn and presence scores...')
    wknn = presence['wknn']
    sparse.save_npz(file_wknn, wknn)
    df_presence = pd.concat([max_presence, group_presence], axis=1).set_axis(['max']+list(group_presence.columns), axis=1)
    df_presence.to_csv(file_presence_score, sep='\t')

    if not args.no_lab_transfer:
        print('[PROGRESS] Transferring labels of ROIGroup, ROIGroupFine, cell_type and supercluster_term...')
        df_labs_roi = transfer_labels(adata_ref, adata_query, wknn, label_key='ROIGroup')
        df_labs_roi_fine = transfer_labels(adata_ref, adata_query, wknn, label_key='ROIGroupFine')
        df_labs_cell_type = transfer_labels(adata_ref, adata_query, wknn, label_key='cell_type')
        df_labs_supercluster = transfer_labels(adata_ref, adata_query, wknn, label_key='supercluster_term')
        print('[PROGRESS] Saving transferred labels...')
        df_labs_roi.to_csv(os.path.join(args.output, 'label_transfer_ROIGroup.tsv'), sep='\t')
        df_labs_roi_fine.to_csv(os.path.join(args.output, 'label_transfer_ROIGroupFine.tsv'), sep='\t')
        df_labs_cell_type.to_csv(os.path.join(args.output, 'label_transfer_cell_type.tsv'), sep='\t')
        df_labs_supercluster.to_csv(os.path.join(args.output, 'label_transfer_supercluster_term.tsv'), sep='\t')

        adata_query.obs['pred_Siletti_ROIGroup'] = df_labs_roi['best_label'].copy()
        adata_query.obs['pred_Siletti_ROIGroupFine'] = df_labs_roi_fine['best_label'].copy()
        adata_query.obs['pred_Siletti_cell_type'] = df_labs_cell_type['best_label'].copy()
        adata_query.obs['pred_Siletti_supercluster_term'] = df_labs_supercluster['best_label'].copy()

    if verbose:
        print('[PROGRESS] Preparing embeddings for visualization...')
    if 'X_umap' not in adata_ref.obsm.keys():
        sc.pp.neighbors(adata_ref, use_rep='X_scarches')
        sc.tl.umap(adata_ref)
    if args.force_new_umap or (args.vis_rep_query is None) or (args.vis_rep_query not in adata_query.obsm.keys()):
        sc.pp.neighbors(adata_query, use_rep='X_scarches')
        sc.tl.umap(adata_query)
        args.vis_rep_query = 'X_umap'

    if verbose:
        print('[PROGRESS] Saving query data...')
    adata_query.write_h5ad(os.path.join(args.output, 'query.h5ad'))

    if args.save_full_query:
        if verbose:
            print('[PROGRESS] Saving full query anndata (all genes + results)...')
        adata_query_full.obsm['X_scarches'] = adata_query.obsm['X_scarches']
        adata_query_full.obsm['X_umap'] = adata_query.obsm['X_umap']
        if not args.no_lab_transfer:
            adata_query_full.obs['pred_Siletti_ROIGroup'] = adata_query.obs['pred_Siletti_ROIGroup'].copy()
            adata_query_full.obs['pred_Siletti_ROIGroupFine'] = adata_query.obs['pred_Siletti_ROIGroupFine'].copy()
            adata_query_full.obs['pred_Siletti_cell_type'] = adata_query.obs['pred_Siletti_cell_type'].copy()
            adata_query_full.obs['pred_Siletti_supercluster_term'] = adata_query.obs['pred_Siletti_supercluster_term'].copy()
        adata_query_full.write_h5ad(file_q_adata2save)

    if verbose:
        print('[PROGRESS] Generating HTML report...')
    generate_mapping_report(
        adata_ref,
        adata_query,
        df_presence,
        df_labels={'ROIGroup': df_labs_roi, 'ROIGroupFine': df_labs_roi_fine, 'cell_type': df_labs_cell_type, 'supercluster_term': df_labs_supercluster} if not args.no_lab_transfer else None,
        ref_annot_labs=['ROIGroup', 'ROIGroupFine', 'cell_type', 'supercluster_term'],
        vis_rep_query=args.vis_rep_query,
        output=loc_report,
        report_type=args.report_type,
        verbose=verbose
    )

    if verbose:
        print('[DONE]')
