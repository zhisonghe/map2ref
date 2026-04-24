import warnings
warnings.filterwarnings("ignore")

import os
from helpers.cli import build_arg_parser

_DEFAULT_REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'model_Siletti')


def cmd_interface():
    parser = build_arg_parser(
        description='Do mapping of the provided data to the adult human brain scRNA-seq atlas (Siletti et al. 2023)',
        default_ref=_DEFAULT_REF,
        no_lab_transfer_help='Skip label transfer for ROI groups, cell type and supercluster',
    )
    return parser.parse_args()

if __name__ == '__main__':
    args = cmd_interface()

    _LABEL_CONFIG = [
        {'key': 'ROIGroup',          'obs_col': 'pred_Siletti_ROIGroup',          'tsv': 'label_transfer_ROIGroup.tsv',          'report_key': 'ROIGroup'},
        {'key': 'ROIGroupFine',      'obs_col': 'pred_Siletti_ROIGroupFine',      'tsv': 'label_transfer_ROIGroupFine.tsv',      'report_key': 'ROIGroupFine'},
        {'key': 'cell_type',         'obs_col': 'pred_Siletti_cell_type',         'tsv': 'label_transfer_cell_type.tsv',         'report_key': 'cell_type'},
        {'key': 'supercluster_term', 'obs_col': 'pred_Siletti_supercluster_term', 'tsv': 'label_transfer_supercluster_term.tsv', 'report_key': 'supercluster_term'},
    ]
    _REF_ANNOT_LABS = ['ROIGroup', 'ROIGroupFine', 'cell_type', 'supercluster_term']

    if args.report_only:
        from helpers.pipeline import run_report_only
        run_report_only(
            args,
            label_config=_LABEL_CONFIG,
            ref_annot_labs=_REF_ANNOT_LABS,
        )
    else:
        import scarches
        import scanpy as sc
        from helpers.pipeline import run_mapping

        run_mapping(
            args,
            load_vae=lambda adata_ref: scarches.models.scPoli.load(args.ref, adata_ref),
            label_config=_LABEL_CONFIG,
            ref_annot_labs=_REF_ANNOT_LABS,
        )

