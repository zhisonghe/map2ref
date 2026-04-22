import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import os

def report(adata_ref,
           adata_query,
           presence,
           df_labels = None,
           ref_annot_labs = [],
           vis_rep_ref = 'X_umap',
           lat_rep_ref = 'X_scarches',
           vis_rep_query = 'X_umap',
           lat_rep_query = 'X_scarches',
           output = 'report',
           verbose = True
          ):
    if (vis_rep_ref is None) or (vis_rep_ref not in adata_ref.obsm.keys()):
        if verbose:
            print('[PROGRESS] Generating UMAP for the reference...')
        sc.pp.neighbors(adata_ref, use_rep = lat_rep_ref)
        sc.tl.umap(adata_ref)
        vis_rep_ref = 'X_umap'
    if (vis_rep_query is None) or (vis_rep_query not in adata_query.obsm.keys()):
        if verbose:
            print('[PROGRESS] Generating UMAP for the query...')
        sc.pp.neighbors(adata_query, use_rep = lat_rep_query)
        sc.tl.umap(adata_query)
    
    obs_ref = adata_ref.obs.copy()
    obs_query = adata_query.obs.copy()
    os.makedirs(output, exist_ok=True)
    
    if verbose:
        print('[PROGRESS] Making figures...')
    
    ref_info2plot = np.intersect1d(np.array(ref_annot_labs), adata_ref.obs.columns).tolist() + ['max_presence']
    adata_ref.obs['max_presence'] = presence['max'][adata_ref.obs_names]
    fig, axs = plt.subplots(1, len(ref_info2plot), figsize=(5*len(ref_info2plot),4))
    if len(ref_info2plot) > 1:
        for i in range(len(ref_info2plot)-1):
            sc.pl.embedding(adata_ref, ax=axs[i], basis=vis_rep_ref, color = ref_info2plot[i], show=False, frameon=False, add_outline=False)
            axs[i].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=5)
        sc.pl.embedding(adata_ref, ax=axs[len(ref_info2plot)-1], basis=vis_rep_ref, color = 'max_presence', color_map='RdBu', title='Max presence score', frameon=False, size=0.2, sort_order=False, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output,'ref.png'))
    
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
            plt.tight_layout()
            plt.savefig(os.path.join(output,'query.png'))
        else:
            if isinstance(df_labels, list):
                df_labels = df_labels[0]
            adata_query.obs['best_score'] = df_labels['best_score'][adata_query.obs_names]
            adata_query.obs['best_label'] = df_labels['best_label'][adata_query.obs_names]
            fig, axs = plt.subplots(1, 2, figsize=(9,4))
            sc.pl.embedding(adata_query, ax=axs[0], basis=vis_rep_query, color = 'best_score', color_map='YlGnBu', title='Best score', show=False, frameon=False, add_outline=False)
            sc.pl.embedding(adata_query, ax=axs[1], basis=vis_rep_query, color = 'best_label', title='Best label', show=False, frameon=False, add_outline=False)
            axs[1].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=5)
            plt.tight_layout()
            plt.savefig(os.path.join(output,'query.png'))
    else:
        fig, axs = plt.subplots(1, 1, figsize=(4,2))
        axs.axis([0, 10, 0, 10])
        axs.text(5,5,'No label transfer', verticalalignment='center', horizontalalignment='center')
        axs.set_axis_off()
        plt.savefig(os.path.join(output,'query.png'))
        
    
    if verbose:
        print('[PROGRESS] Generating HTML...')
    
    page_title_text='Report'
    title_text = 'Reference mapping report'
    text = 'This is the brief report of the reference mapping results'
    ref_text = 'Reference plots (annotation and presence scores)'
    query_text = 'Query plots (label transfer)'
    
    html = f'''
        <html>
            <head>
                <title>{page_title_text}</title>
            </head>
            <body>
                <h1>{title_text}</h1>
                <p>{text}</p>
                <h2>{ref_text}</h2>
                <img src='ref.png'>
                <h2>{query_text}</h2>
                <img src='query.png'>
            </body>
        </html>
        '''
    # 3. Write the html string as an HTML file
    with open(os.path.join(output,'report.html'), 'w') as f:
        f.write(html)

