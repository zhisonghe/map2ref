import warnings
warnings.filterwarnings("ignore")

import scanpy as sc
import numpy as np
import pandas as pd
import scvi
import os
import argparse

from scipy import sparse
from mapping_scarches import train_scarches, get_latent_space
from wknn import estimate_presence_score, transfer_labels
from report import report

from tqdm import tqdm

def hierarchical_region_lab_transfer(adata_ref, adata_query, wknn):
    region_incl = {'Forebrain' : ['Forebrain','Telencephalon','Cortex','Subcortex','Striatum','Hippocampus','Diencephalon','Hypothalamus','Thalamus'],
                   'Telencephalon': ['Telencephalon','Cortex','Subcortex','Striatum'],
                   'Cortex': ['Cortex'],
                   'Subcortex': ['Subcortex','Striatum'],
                   'Diencephalon': ['Diencephalon','Thalamus','Hypothalamus'],
                   'Hypothalamus': ['Hypothalamus'],
                   'Thalamus': ['Thalamus'],
                   'Midbrain': ['Midbrain','Midbrain dorsal','Midbrain ventral'],
                   'Midbrain dorsal': ['Midbrain dorsal'],
                   'Midbrain ventral': ['Midbrain ventral'],
                   'Hindbrain': ['Hindbrain','Cerebellum','Pons','Medulla'],
                   'Cerebellum': ['Cerebellum'],
                   'Pons': ['Pons'],
                   'Medulla': ['Medulla']}
    region_hier = {'Brain' : ['Forebrain','Midbrain','Hindbrain'],
                   'Forebrain' : ['Telencephalon','Diencephalon'],
                   'Telencephalon' : ['Cortex','Subcortex'],
                   'Diencephalon' : ['Hypothalamus','Thalamus'],
                   'Midbrain' : ['Midbrain dorsal','Midbrain ventral'],
                   'Hindbrain' : ['Cerebellum','Pons','Medulla']}
    
    def find_commonnestn_recursive(x, hier, start_at, thres_over = 0.5, nmax=2):
        idx_max = x[hier[start_at]].idxmax()
        if idx_max in hier.keys():
            return find_commonnestn_recursive(x, hier, idx_max, thres_over = thres_over, nmax=nmax)
        largestn = x[hier[start_at]].nlargest(nmax)
        if nmax == 1:
            return largestn.index[0]
        else:
            if largestn.max() - largestn.min() > largestn.min() * thres_over:
                largestn = largestn[largestn > largestn.min() + largestn.min() * thres_over]
            return list(largestn.index)
    
    df_labs_region = transfer_labels(adata_ref, adata_query, wknn, label_key='Subregion')
    freq_hier_regions = pd.concat([ pd.DataFrame(df_labs_region.loc[:,region_incl[x]].sum(1), columns=[x]) for x in region_incl.keys()], axis=1)
    tqdm.pandas()
    comm_region = freq_hier_regions.progress_apply(lambda x: find_commonnestn_recursive(x, region_hier, 'Brain', nmax=1), axis=1)
    
    return comm_region

def cmd_interface():
    parser = argparse.ArgumentParser(description="Do mapping of the provided data to the developing human brain scRNA-seq atlas (Braun et al. 2023)")
    parser.add_argument(
        '-r', '--ref',
        type=str,
        dest='ref',
        default='/links/groups/treutlein/USERS/zhisong_he/Work/public_datasets/Linnarsson_fetal_human_brain_atlas/preprint_cellranger/mapping_as_reference/model_Braun',
        help='Location of the reference model directory (default: /links/groups/treutlein/USERS/zhisong_he/Work/public_datasets/Linnarsson_fetal_human_brain_atlas/preprint_cellranger/mapping_as_reference/model_Braun)',
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
        '--save-vae-query',
        action='store_true',
        dest='save_vae_query',
        help='Save the processed query anndata for scArches training'
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
        help='Size of mini-batch in scARCHES model training (default: 1024)'
    )
    parser.add_argument(
        "--epochs",
        type=int,
        dest='epochs',
        default=200,
        help="Epochs in scARCHES model training (default: 200)"
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
        help='Skip label transfer for cell class and subregions'
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
        '--quiet',
        action='store_true',
        dest='quiet',
        help='Quiet run without verbose message'
    )
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = cmd_interface()
    
    verbose = not args.quiet
    file_ref_adata = os.path.join(args.ref,'ref.h5ad')
    file_lat_ref = os.path.join(args.output, "lat_rep_ref.npy")
    file_lat_query = os.path.join(args.output, "lat_rep_query.npy")
    file_wknn = os.path.join(args.output, 'wknn.npz')
    file_presence_score = os.path.join(args.output, 'presence_scores.tsv')
    file_q_adata2save = os.path.join(args.output, 'query_processed.h5ad')
    loc_scarches_model = os.path.join(args.output,'model')
    loc_report = os.path.join(args.output, 'report')
    
    if verbose:
        print('[PROGRESS] Loading data...')
    adata_ref = sc.read(file_ref_adata)
    adata_query = sc.read(args.query)
    if args.save_full_query:
        adata_query_full = adata_query
    
    if verbose:
        print('[PROGRESS] scArches model training...')
    vae = scvi.model.SCANVI.load(args.ref, adata_ref)
    vae_q = train_scarches(adata_query, adata_ref, vae = vae, col_batch = args.query_batch_key, batch_size = args.batch_size, epochs = args.epochs, keep_original_adata = False, verbose = not args.quiet)
    
    if verbose:
        print('[PROGRESS] Saving the model...')
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(loc_scarches_model, exist_ok=True)
    vae_q.save(
        loc_scarches_model,
        overwrite=True,
        save_anndata=args.save_vae_query,
    )
    
    if verbose:
        print('[PROGRESS] Generating latent representations...')
    lat_reps = get_latent_space(adata_query, vae_q, adata_ref, col_batch = args.query_batch_key)
    lat_rep_ref, lat_rep_q = (lat_reps['ref'], lat_reps['query'])
    
    if verbose:
        print('[PROGRESS] Saving latent representations...')
    np.save(file_lat_ref, lat_rep_ref)
    np.save(file_lat_query, lat_rep_q)
    adata_ref.obsm['X_scarches'] = lat_rep_ref
    adata_query.obsm['X_scarches'] = lat_rep_q
    if args.save_full_query:
        adata_query_full.obsm['X_scarches'] = lat_rep_q
    
    if verbose:
        print('[PROGRESS] Calculating presence scores per query data set...')
    presence = estimate_presence_score(adata_ref, adata_query, use_rep_ref_wknn = 'X_scarches', use_rep_query_wknn = 'X_scarches', k_wknn = args.k_wknn, k_ref_trans_prop = args.k_ref, split_by = args.query_batch_key, do_random_walk = not args.no_smooth_presence)
    max_presence, group_presence = (presence['max'], presence['per_group'])
    
    if verbose:
        print('[PROGRESS] Saving wknn and presence scores...')
    wknn = presence['wknn']
    sparse.save_npz(file_wknn, wknn)
    df_presence = pd.concat([max_presence, group_presence], axis=1).set_axis(['max']+list(group_presence.columns), axis=1)
    df_presence.to_csv(file_presence_score, sep='\t')
    
    if not args.no_lab_transfer:
        print('[PROGRESS] Transferring labels of CellClass, Subregion and Neuron_NTT...')
        df_labs_class = transfer_labels(adata_ref, adata_query, wknn, label_key='CellClass')
        df_labs_region = transfer_labels(adata_ref, adata_query, wknn, label_key='Subregion')
        df_labs_ntt = transfer_labels(adata_ref, adata_query, wknn, label_key='Neuron_NTT')
        vec_lab_region = hierarchical_region_lab_transfer(adata_ref, adata_query, wknn)
        print('[PROGRESS] Saving transferred labels...')
        df_labs_class.to_csv(os.path.join(args.output, 'label_transfer_class.tsv'), sep='\t')
        df_labs_region.to_csv(os.path.join(args.output, 'label_transfer_region.tsv'), sep='\t')
        df_labs_ntt.to_csv(os.path.join(args.output, 'label_transfer_NTT.tsv'), sep='\t')
        vec_lab_region.to_csv(os.path.join(args.output, 'label_transfer_region_hier.tsv'), sep='\t')
        
        if args.save_full_query:
            adata_query_full.obs['CellClass'] = df_labs_class['best_label'].copy()
            adata_query_full.obs['Subregion'] = df_labs_region['best_label'].copy()
            adata_query_full.obs['Subregion_hier'] = vec_lab_region.copy()
            adata_query_full.obs['Neuron_NTT'] = df_labs_ntt['best_label'].copy()
    
    if verbose:
        print('[PROGRESS] Generating HTML report...')
    report(adata_ref,
           adata_query,
           df_presence,
           df_labels = {'Class':df_labs_class, 'Region':df_labs_region, 'Region_hier': pd.DataFrame({'best_label': vec_lab_region, 'best_score' : np.repeat(1,len(vec_lab_region))}, index=vec_lab_region.index), 'NTT':df_labs_ntt} if not args.no_lab_transfer else None,
           ref_annot_labs = ['CellClass','Subregion','Neuron_NTT'],
           vis_rep_query = args.vis_rep_query,
           lat_rep_query = 'X_scarches',
           output = loc_report,
           verbose=verbose)
    
    if args.save_full_query:
        if verbose:
            print('[PROGRESS] Saving query anndata...')
        adata_query_full.write_h5ad(file_q_adata2save)
    
    if verbose:
        print('[DONE]')
