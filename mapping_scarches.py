import warnings

warnings.filterwarnings("ignore")

import os
import sys
import argparse

import numpy as np
import pandas as pd
import scanpy as sc
import scipy
import scvi

def is_lognorm(mat,
               max_row_num = 10000,
               max_col_num = None):
    max_vals = mat.max(0)
    if scipy.sparse.issparse(mat): max_vals = np.array(max_vals.todense()).flatten()
    diff_from_int = max_vals - np.round(max_vals)
    return np.max(np.abs(diff_from_int)) > 1e-5
    
def train_scarches(adata,
                   ref_adata,
                   vae,
                   col_batch = None,
                   batch_size = 1024,
                   epochs = 500,
                   keep_original_adata = False,
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
        print('[PROGRESS] Data expected to be in log: ' + str(r_x_log))
        q_x_log = is_lognorm(adata.X)
        if q_x_log != r_x_log:
            raise RuntimeError('The query data and reference data do not have the same expression scale (raw counts vs. log-normalized)')
    
    if col_batch:
        if isinstance(vae, scvi.model._scanvi.SCANVI) or isinstance(vae, scvi.model._scvi.SCVI):
            ref_batch_key = vae.adata_manager._registry["setup_args"]["batch_key"]
            adata.obs[ref_batch_key] = adata.obs[col_batch].copy()
    
    if isinstance(vae, scvi.model._scanvi.SCANVI) or isinstance(vae, scvi.model._scvi.SCVI):
        model.prepare_query_anndata(adata, vae)
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
    vae_q = train_scarches(adata, ref_adata, vae = vae, col_batch = args.query_batch_key, batch_size = args.batch_size, epochs = args.epochs, keep_original_adata = False, verbose = True)
    
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
