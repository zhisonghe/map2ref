import warnings
warnings.filterwarnings("ignore")

import scanpy as sc
import numpy as np
import pandas as pd
import scvi
import os
import argparse

import matplotlib.pyplot as plt
from scipy import sparse
from helpers.mapping_scarches import train_scarches, get_latent_space
from helpers.wknn import estimate_presence_score, transfer_labels
from helpers.report import _fig_to_base64, _write_basic_html, _write_fancy_html

from tqdm import tqdm

_DEFAULT_REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'model_Braun')


def generate_braun_report(adata_ref,
           adata_query,
           presence,
           df_labels = None,
           ref_annot_labs = [],
           vis_rep_ref = 'X_umap',
           vis_rep_query = 'X_umap',
           output = 'report',
           report_type = 'basic',
           verbose = True
          ):
    obs_ref = adata_ref.obs.copy()
    obs_query = adata_query.obs.copy()
    os.makedirs(output, exist_ok=True)
    
    if verbose:
        print('[PROGRESS] Making figures...')
    
    ref_info2plot = np.intersect1d(np.array(ref_annot_labs), adata_ref.obs.columns).tolist() + ['max_presence']
    adata_ref.obs['max_presence'] = presence['max'][adata_ref.obs_names]
    fig, axs = plt.subplots(1, len(ref_info2plot), figsize=(5*len(ref_info2plot),4))
    axs = np.atleast_1d(axs)
    for i in range(len(ref_info2plot)-1):
        sc.pl.embedding(adata_ref, ax=axs[i], basis=vis_rep_ref, color = ref_info2plot[i], show=False, frameon=False, add_outline=False)
        axs[i].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=5)
    sc.pl.embedding(adata_ref, ax=axs[len(ref_info2plot)-1], basis=vis_rep_ref, color = 'max_presence', color_map='RdBu', title='Max presence score', frameon=False, size=0.2, sort_order=False, show=False)
    fig.patch.set_facecolor('#fffaf1')
    plt.tight_layout()
    ref_image_b64 = _fig_to_base64(fig)
    if report_type == 'basic':
        fig.savefig(os.path.join(output, 'ref.png'))
    plt.close(fig)
    
    if df_labels is not None and len(df_labels)>0:
        if (isinstance(df_labels, list) or isinstance(df_labels, dict)) and len(df_labels)>1:
            fig, axs = plt.subplots(len(df_labels), 2, figsize=(9,4*len(df_labels)))
            for i in range(len(df_labels)):
                df = df_labels[i] if isinstance(df_labels, list) else df_labels[list(df_labels.keys())[i]]
                suf = str(i) if isinstance(df_labels, list) else list(df_labels.keys())[i]
                adata_query.obs['best_score'] = df['best_score'][adata_query.obs_names]
                adata_query.obs['best_label'] = df['best_label'][adata_query.obs_names]
                sc.pl.embedding(adata_query, ax=axs[i,0], basis=vis_rep_query, color = 'best_score', color_map='YlGnBu', title='Best score ('+suf+')', show=False, frameon=False, add_outline=False)
                sc.pl.embedding(adata_query, ax=axs[i,1], basis=vis_rep_query, color = 'best_label', title='Best label ('+suf+')', show=False, frameon=False, add_outline=False)
                axs[i,1].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=5)
            fig.patch.set_facecolor('#fffaf1')
            plt.tight_layout()
        else:
            if isinstance(df_labels, list):
                df_labels = df_labels[0]
            adata_query.obs['best_score'] = df_labels['best_score'][adata_query.obs_names]
            adata_query.obs['best_label'] = df_labels['best_label'][adata_query.obs_names]
            fig, axs = plt.subplots(1, 2, figsize=(9,4))
            sc.pl.embedding(adata_query, ax=axs[0], basis=vis_rep_query, color = 'best_score', color_map='YlGnBu', title='Best score', show=False, frameon=False, add_outline=False)
            sc.pl.embedding(adata_query, ax=axs[1], basis=vis_rep_query, color = 'best_label', title='Best label', show=False, frameon=False, add_outline=False)
            axs[1].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=5)
            fig.patch.set_facecolor('#fffaf1')
            plt.tight_layout()
    else:
        fig, axs = plt.subplots(1, 1, figsize=(4,2))
        axs.axis([0, 10, 0, 10])
        axs.text(5,5,'No label transfer', verticalalignment='center', horizontalalignment='center')
        axs.set_axis_off()
        fig.patch.set_facecolor('#fffaf1')

    query_image_b64 = _fig_to_base64(fig)
    if report_type == 'basic':
        fig.savefig(os.path.join(output, 'query.png'))
    plt.close(fig)
        
    
    if verbose:
        print('[PROGRESS] Generating HTML...')
    
    title_text = 'Reference mapping report'
    text = 'This is the brief report of the reference mapping results'
    ref_text = 'Reference plots (annotation and presence scores)'
    query_text = 'Query plots (label transfer)'

    if report_type == 'basic':
        _write_basic_html(output, 'Report', title_text, text, ref_text, query_text)
    elif report_type == 'fancy':
        _write_fancy_html(output,
                          title_text,
                          text,
                          ref_text,
                          query_text,
                          ref_image_b64,
                          query_image_b64,
                          presence,
                          df_labels,
                          ref_annot_labs,
                          vis_rep_ref,
                          vis_rep_query)
    else:
        raise ValueError("report_type must be 'basic' or 'fancy'")

    adata_ref.obs = obs_ref
    adata_query.obs = obs_query


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
        default=_DEFAULT_REF,
        help='Location of the reference model directory (default: models/model_Braun relative to this script)',
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
        adata_query_full = adata_query.copy()
    
    if verbose:
        print('[PROGRESS] scArches model training...')
    vae = scvi.model.SCANVI.load(args.ref, adata_ref)
    vae_q = train_scarches(adata_query, adata_ref, vae = vae, col_batch = args.query_batch_key, batch_size = args.batch_size, epochs = args.epochs, keep_original_adata = False, skip_scale_check = args.skip_scale_check, verbose = not args.quiet)
    
    if verbose:
        print('[PROGRESS] Saving the model...')
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(loc_scarches_model, exist_ok=True)
    vae_q.save(
        loc_scarches_model,
        overwrite=True,
        save_anndata=False,
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
        
        adata_query.obs['pred_Braun_CellClass'] = df_labs_class['best_label'].copy()
        adata_query.obs['pred_Braun_Subregion'] = df_labs_region['best_label'].copy()
        adata_query.obs['pred_Braun_Subregion_hier'] = vec_lab_region.copy()
        adata_query.obs['pred_Braun_Neuron_NTT'] = df_labs_ntt['best_label'].copy()
    
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
            adata_query_full.obs['pred_Braun_CellClass'] = adata_query.obs['pred_Braun_CellClass'].copy()
            adata_query_full.obs['pred_Braun_Subregion'] = adata_query.obs['pred_Braun_Subregion'].copy()
            adata_query_full.obs['pred_Braun_Subregion_hier'] = adata_query.obs['pred_Braun_Subregion_hier'].copy()
            adata_query_full.obs['pred_Braun_Neuron_NTT'] = adata_query.obs['pred_Braun_Neuron_NTT'].copy()
        adata_query_full.write_h5ad(file_q_adata2save)
    
    if verbose:
        print('[PROGRESS] Generating HTML report...')
    generate_braun_report(
        adata_ref,
        adata_query,
        df_presence,
        df_labels = {'Class':df_labs_class, 'Region':df_labs_region, 'Region_hier': pd.DataFrame({'best_label': vec_lab_region, 'best_score' : np.repeat(1,len(vec_lab_region))}, index=vec_lab_region.index), 'NTT':df_labs_ntt} if not args.no_lab_transfer else None,
        ref_annot_labs = ['CellClass','Subregion','Neuron_NTT'],
        vis_rep_query = args.vis_rep_query,
        output = loc_report,
        report_type = args.report_type,
        verbose=verbose
        )
    
    if verbose:
        print('[DONE]')
