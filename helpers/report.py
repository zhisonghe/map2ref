import base64
from io import BytesIO
import os

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from helpers.log import get_logger

_log = get_logger('report')


def generate_mapping_report(adata_ref,
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
        _log('Making figures...')

    ref_info2plot = np.intersect1d(np.array(ref_annot_labs), adata_ref.obs.columns).tolist() + ['max_presence']
    adata_ref.obs['max_presence'] = presence['max'][adata_ref.obs_names]
    n_ref = len(ref_info2plot)
    if report_type == 'fancy':
        # one column so it stacks to match the height of the multi-row query panel
        fig, axs = plt.subplots(n_ref, 1, figsize=(5, 4 * n_ref))
    else:
        # one row for the basic side-by-side HTML layout
        fig, axs = plt.subplots(1, n_ref, figsize=(5 * n_ref, 4))
    axs = np.atleast_1d(axs)
    for i in range(n_ref - 1):
        sc.pl.embedding(adata_ref, ax=axs[i], basis=vis_rep_ref, color=ref_info2plot[i], show=False, frameon=False, add_outline=False)
        axs[i].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=5)
    sc.pl.embedding(adata_ref, ax=axs[n_ref - 1], basis=vis_rep_ref, color='max_presence', color_map='RdBu', title='Max presence score', frameon=False, size=0.2, sort_order=False, show=False)
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
        _log('Generating HTML...')

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

def _fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=180, bbox_inches='tight', facecolor=fig.get_facecolor())
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('ascii')


def _summarize_presence(presence):
    max_presence = presence['max']
    return {
        'n_ref_cells': len(max_presence),
        'mean': float(max_presence.mean()),
        'median': float(max_presence.median()),
        'max': float(max_presence.max()),
    }


def _presence_table_html(presence, top_n=12):
    top_presence = presence['max'].sort_values(ascending=False).head(top_n)
    rows = []
    for cell_id, value in top_presence.items():
        rows.append(
            f"<tr><td>{cell_id}</td><td>{value:.3f}</td></tr>"
        )
    return ''.join(rows)


def _label_summary_html(df_labels, top_n=8):
    if not df_labels:
        return "<div class='empty-state'>Label transfer was skipped for this run.</div>"

    blocks = []
    items = df_labels.items() if isinstance(df_labels, dict) else enumerate(df_labels)
    for key, df in items:
        counts = df['best_label'].value_counts().head(top_n)
        rows = ''.join(
            f"<tr><td>{label}</td><td>{count}</td></tr>" for label, count in counts.items()
        )
        blocks.append(
            f"""
            <section class='table-card'>
                <h3>{key}</h3>
                <table>
                    <thead>
                        <tr><th>Top label</th><th>Cells</th></tr>
                    </thead>
                    <tbody>{rows}</tbody>
                </table>
            </section>
            """
        )
    return ''.join(blocks)


def _write_basic_html(output, page_title_text, title_text, text, ref_text, query_text):
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
    with open(os.path.join(output, 'report.html'), 'w') as f:
        f.write(html)


def _write_fancy_html(output,
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
                      vis_rep_query):
    presence_stats = _summarize_presence(presence)
    ref_annotations = ', '.join(ref_annot_labs) if ref_annot_labs else 'None'
    html = f'''
    <!DOCTYPE html>
    <html lang="en">
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <title>{title_text}</title>
            <style>
                :root {{
                    --bg: #f3efe5;
                    --surface: rgba(255, 252, 247, 0.84);
                    --surface-strong: #fffaf1;
                    --ink: #1f1b16;
                    --muted: #665c51;
                    --line: rgba(53, 39, 24, 0.14);
                    --accent: #0c7c59;
                    --accent-2: #c84c09;
                    --accent-3: #1f5aa6;
                    --shadow: 0 18px 50px rgba(56, 42, 24, 0.12);
                    --radius: 22px;
                }}
                * {{ box-sizing: border-box; }}
                body {{
                    margin: 0;
                    font-family: Georgia, "Iowan Old Style", "Palatino Linotype", serif;
                    color: var(--ink);
                    background:
                        radial-gradient(circle at top left, rgba(200, 76, 9, 0.10), transparent 28%),
                        radial-gradient(circle at top right, rgba(12, 124, 89, 0.12), transparent 24%),
                        linear-gradient(180deg, #f7f2e8 0%, var(--bg) 100%);
                }}
                .page {{
                    width: min(1280px, calc(100% - 40px));
                    margin: 28px auto 48px;
                }}
                .hero {{
                    background: linear-gradient(135deg, rgba(255,255,255,0.88), rgba(255,247,234,0.92));
                    border: 1px solid var(--line);
                    border-radius: 28px;
                    padding: 28px;
                    box-shadow: var(--shadow);
                    position: relative;
                    overflow: hidden;
                }}
                .hero::after {{
                    content: "";
                    position: absolute;
                    inset: auto -80px -80px auto;
                    width: 240px;
                    height: 240px;
                    border-radius: 50%;
                    background: radial-gradient(circle, rgba(31, 90, 166, 0.18), transparent 68%);
                }}
                h1, h2, h3 {{ margin: 0; font-weight: 600; }}
                h1 {{ font-size: clamp(2rem, 4vw, 3.4rem); letter-spacing: -0.04em; }}
                h2 {{ font-size: 1.4rem; margin-bottom: 14px; }}
                h3 {{ font-size: 1rem; margin-bottom: 10px; }}
                p {{ margin: 0; line-height: 1.6; color: var(--muted); }}
                .hero-copy {{ max-width: 760px; display: grid; gap: 16px; }}
                .meta {{
                    display: flex;
                    flex-wrap: wrap;
                    gap: 10px;
                    margin-top: 18px;
                }}
                .pill {{
                    background: rgba(12, 124, 89, 0.08);
                    color: var(--accent);
                    border: 1px solid rgba(12, 124, 89, 0.12);
                    padding: 8px 12px;
                    border-radius: 999px;
                    font-size: 0.92rem;
                }}
                .section {{ margin-top: 26px; }}
                .stats {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                    gap: 14px;
                }}
                .stat-card, .panel, .table-card {{
                    background: var(--surface);
                    border: 1px solid var(--line);
                    border-radius: var(--radius);
                    box-shadow: var(--shadow);
                    backdrop-filter: blur(10px);
                }}
                .stat-card {{ padding: 18px 20px; }}
                .stat-label {{ font-size: 0.86rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); }}
                .stat-value {{ margin-top: 8px; font-size: 2rem; line-height: 1; }}
                .gallery {{
                    display: grid;
                    grid-template-columns: 1fr 2fr;
                    gap: 18px;
                }}
                .panel {{ padding: 18px; }}
                .panel img {{
                    width: 100%;
                    display: block;
                    border-radius: 16px;
                    border: 1px solid rgba(53, 39, 24, 0.08);
                    background: var(--surface-strong);
                }}
                .panel-note {{ margin-top: 12px; font-size: 0.95rem; }}
                .tables {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
                    gap: 18px;
                }}
                .table-card {{ padding: 18px; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ padding: 10px 0; text-align: left; border-bottom: 1px solid var(--line); font-size: 0.95rem; }}
                th {{ color: var(--muted); font-weight: 600; }}
                tbody tr:last-child td {{ border-bottom: 0; }}
                .empty-state {{
                    padding: 24px;
                    border-radius: var(--radius);
                    background: rgba(31, 90, 166, 0.06);
                    border: 1px dashed rgba(31, 90, 166, 0.2);
                    color: var(--muted);
                }}
                @media (max-width: 720px) {{
                    .page {{ width: min(100% - 24px, 1280px); }}
                    .hero {{ padding: 22px; border-radius: 22px; }}
                    .gallery {{ grid-template-columns: 1fr; }}
                }}
            </style>
        </head>
        <body>
            <main class="page">
                <section class="hero">
                    <div class="hero-copy">
                        <h1>{title_text}</h1>
                        <p>{text}</p>
                        <div class="meta">
                            <span class="pill">Reference embedding: {vis_rep_ref}</span>
                            <span class="pill">Query embedding: {vis_rep_query}</span>
                            <span class="pill">Reference annotations: {ref_annotations}</span>
                            <span class="pill">Single-file HTML report</span>
                        </div>
                    </div>
                </section>

                <section class="section stats">
                    <article class="stat-card">
                        <div class="stat-label">Reference cells scored</div>
                        <div class="stat-value">{presence_stats['n_ref_cells']}</div>
                    </article>
                    <article class="stat-card">
                        <div class="stat-label">Mean max presence</div>
                        <div class="stat-value">{presence_stats['mean']:.3f}</div>
                    </article>
                    <article class="stat-card">
                        <div class="stat-label">Median max presence</div>
                        <div class="stat-value">{presence_stats['median']:.3f}</div>
                    </article>
                    <article class="stat-card">
                        <div class="stat-label">Peak max presence</div>
                        <div class="stat-value">{presence_stats['max']:.3f}</div>
                    </article>
                </section>

                <section class="section gallery">
                    <article class="panel">
                        <h2>{ref_text}</h2>
                        <img src="data:image/png;base64,{ref_image_b64}" alt="Reference plots">
                        <p class="panel-note">Reference annotation panels and normalized presence score overview.</p>
                    </article>
                    <article class="panel">
                        <h2>{query_text}</h2>
                        <img src="data:image/png;base64,{query_image_b64}" alt="Query plots">
                        <p class="panel-note">Transferred labels and confidence overlays for the mapped query cells.</p>
                    </article>
                </section>

                <section class="section tables">
                    <article class="table-card">
                        <h2>Top Presence Signals</h2>
                        <table>
                            <thead>
                                <tr><th>Reference cell</th><th>Max presence</th></tr>
                            </thead>
                            <tbody>{_presence_table_html(presence)}</tbody>
                        </table>
                    </article>
                    {_label_summary_html(df_labels)}
                </section>
            </main>
        </body>
    </html>
    '''
    with open(os.path.join(output, 'report.html'), 'w') as f:
        f.write(html)
