import os
import argparse


def build_arg_parser(description, default_ref, no_lab_transfer_help):
    """Build the shared argument parser for reference-mapping CLIs.

    Parameters
    ----------
    description : str
        Description shown in ``--help``.
    default_ref : str
        Absolute path to the default reference model directory.
    no_lab_transfer_help : str
        Help text for the ``--no-label-transfer`` flag describing which
        atlas-specific labels are skipped.

    Returns
    -------
    argparse.ArgumentParser
        Parser pre-populated with all shared arguments.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        '-r', '--ref',
        type=str,
        dest='ref',
        default=default_ref,
        help=f'Location of the reference model directory (default: {os.path.basename(default_ref)} relative to this script)',
    )
    parser.add_argument(
        '-q', '--query',
        type=str,
        dest='query',
        default=None,
        help='H5AD file of the query data (required unless --report-only is used)',
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        dest='output',
        default='output',
        help='Path to the output folder (default: ./output)',
    )
    parser.add_argument(
        '--save-full-query',
        action='store_true',
        dest='save_full_query',
        help='Save the full query anndata with the mapping results',
    )
    parser.add_argument(
        '-b', '--query-batch-key',
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
        help='Size of mini-batch in model training (default: 1024)',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        dest='epochs',
        default=200,
        help='Epochs in model training (default: 200)',
    )
    parser.add_argument(
        '-k', '--k-wknn',
        type=int,
        dest='k_wknn',
        default=100,
        help='Number of NNs per query cell in reference (default: 100)',
    )
    parser.add_argument(
        '-n', '--k-ref',
        type=int,
        dest='k_ref',
        default=100,
        help='Number of NNs per reference cell in reference (default: 100)',
    )
    parser.add_argument(
        '--no-smooth-presence',
        action='store_true',
        dest='no_smooth_presence',
        help='Skip the random-walk-based smoothening of presence scores',
    )
    parser.add_argument(
        '--no-label-transfer',
        action='store_true',
        dest='no_lab_transfer',
        help=no_lab_transfer_help,
    )
    parser.add_argument(
        '--vis-rep-query',
        type=str,
        dest='vis_rep_query',
        default='X_umap',
        help='Embedding in the query data for visualization (default: X_umap)',
    )
    parser.add_argument(
        '--force-new-umap',
        action='store_true',
        dest='force_new_umap',
        help='Always make a new UMAP of the query data for visualization using the projected latent representation',
    )
    parser.add_argument(
        '--report-type',
        type=str,
        dest='report_type',
        default='basic',
        choices=['basic', 'fancy'],
        help='HTML report style to generate (default: basic)',
    )
    parser.add_argument(
        '--query-layer',
        type=str,
        dest='query_layer',
        default=None,
        help='Layer in the query AnnData to use as expression input. Use "X" to refer to the .X slot. '
             'By default the layer expected by the pretrained model is used (for SCANVI-based mapping) '
             'or .X (for scPoli-based mapping). When specified, the given layer is copied to the '
             'expected slot before model training.',
    )
    parser.add_argument(
        '--skip-scale-check',
        action='store_true',
        dest='skip_scale_check',
        help='Skip expression scale sanity check between query and reference (use if detection fails)',
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        dest='quiet',
        help='Quiet run without verbose message',
    )
    parser.add_argument(
        '--report-only',
        action='store_true',
        dest='report_only',
        help='Re-generate the HTML report from existing output files without re-running the mapping pipeline. '
             'Loads query.h5ad and presence_scores.tsv from the output folder and ref.h5ad from the model directory.',
    )
    return parser
