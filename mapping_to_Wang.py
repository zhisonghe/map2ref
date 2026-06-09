import warnings
warnings.filterwarnings("ignore")

import os
from helpers.cli import build_arg_parser

_DEFAULT_REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'model_Wang')


def cmd_interface():
    parser = build_arg_parser(
        description='Do mapping of the provided data to the developing human cerebral cortex scMultiome atlas (RNA portion) (Wang et al. 2025)',
        default_ref=_DEFAULT_REF,
        no_lab_transfer_help='Skip label transfer for Group, class, subclass, type',
    )
    return parser.parse_args()

if __name__ == '__main__':
    args = cmd_interface()

    _LABEL_CONFIG = [
        {'key': 'Group',          'obs_col': 'pred_Wang_Group',          'tsv': 'label_transfer_Group.tsv',          'report_key': 'Group'},
        {'key': 'class',          'obs_col': 'pred_Wang_class',          'tsv': 'label_transfer_class.tsv',          'report_key': 'class'},
        {'key': 'subclass',       'obs_col': 'pred_Wang_subclass',       'tsv': 'label_transfer_subclass.tsv',       'report_key': 'subclass'},
        {'key': 'type',           'obs_col': 'pred_Wang_type',           'tsv': 'label_transfer_type.tsv',           'report_key': 'type'},
    ]
    _REF_ANNOT_LABS = ['Group', 'class', 'subclass', 'type']

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

