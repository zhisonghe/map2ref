import warnings
warnings.filterwarnings("ignore")

import os
from helpers.cli import build_arg_parser

_DEFAULT_REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'model_Braun')


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
    parser = build_arg_parser(
        description='Do mapping of the provided data to the developing human brain scRNA-seq atlas (Braun et al. 2023)',
        default_ref=_DEFAULT_REF,
        no_lab_transfer_help='Skip label transfer for cell class and subregions',
    )
    return parser.parse_args()

if __name__ == '__main__':
    args = cmd_interface()

    _LABEL_CONFIG = [
        {'key': 'CellClass',  'obs_col': 'pred_Braun_CellClass',  'tsv': 'label_transfer_class.tsv',  'report_key': 'Class'},
        {'key': 'Subregion',  'obs_col': 'pred_Braun_Subregion',  'tsv': 'label_transfer_region.tsv', 'report_key': 'Region'},
        {'key': 'Neuron_NTT', 'obs_col': 'pred_Braun_Neuron_NTT', 'tsv': 'label_transfer_NTT.tsv',    'report_key': 'NTT'},
    ]
    _REF_ANNOT_LABS = ['CellClass', 'Subregion', 'Neuron_NTT']

    if args.report_only:
        from helpers.pipeline import run_report_only
        run_report_only(
            args,
            label_config=_LABEL_CONFIG,
            ref_annot_labs=_REF_ANNOT_LABS,
            extra_label_config=[
                {'tsv': 'label_transfer_region_hier.tsv', 'obs_col': 'pred_Braun_Subregion_hier', 'report_key': 'Region_hier'},
            ],
        )
    else:
        import numpy as np
        import pandas as pd
        import scvi
        from helpers.wknn import transfer_labels
        from helpers.pipeline import run_mapping
        from tqdm import tqdm

        def _braun_post_hook(adata_ref, adata_query, wknn, output_dir):
            vec = hierarchical_region_lab_transfer(adata_ref, adata_query, wknn)
            vec.to_csv(os.path.join(output_dir, 'label_transfer_region_hier.tsv'), sep='\t')
            adata_query.obs['pred_Braun_Subregion_hier'] = vec.copy()
            return {'Region_hier': pd.DataFrame(
                {'best_label': vec, 'best_score': np.repeat(1, len(vec))},
                index=vec.index,
            )}

        import scanpy as sc

        run_mapping(
            args,
            load_vae=lambda adata_ref: scvi.model.SCANVI.load(args.ref, adata_ref),
            label_config=_LABEL_CONFIG,
            ref_annot_labs=_REF_ANNOT_LABS,
            post_label_hook=_braun_post_hook,
        )

