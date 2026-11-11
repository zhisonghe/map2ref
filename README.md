# map2ref: one-liner CLI for scArches and label transfer with human brain cell atlas references
This is a CLI pipeline to map query scRNA-seq data to a pretrained single-cell brain atlas reference. For each atlas, the pipeline:

1. **Projects the query** into the reference latent space via scArches-style architecture surgery on the pretrained VAE model (scANVI for Braun, scPoli for Wang and Siletti). scPoli models are run with [scpolivi](https://github.com/zhisonghe/scpoli_scvi-tools), a reimplementation of scPoli in scvi-tools; the original scArches package is no longer needed
2. **Calculates presence scores** using a weighted k-nearest-neighbour (WKNN) graph, with optional random-walk smoothing, to quantify how well each reference cell is represented in the query
3. **Transfers labels** from the reference to the query via the WKNN graph; for Braun mapping this includes cell class, subregion, neuron transmitter type (NTT), and a hierarchical region label that resolves each cell to the most specific unambiguous brain region in the developmental hierarchy; for Siletti mapping this includes ROI group, fine ROI group, cell type, and supercluster term
4. **Generates an HTML report** with UMAP visualisations of the mapping results

It includes CLI commands for mapping to the transcriptomic cell atlas of the developing ([Braun et al. 2023](https://www.science.org/doi/10.1126/science.adf1226)) and adult ([Siletti et al. 2023](https://www.science.org/doi/10.1126/science.add7046)) human brain. The pretrained models and the reference AnnData objects (.h5ad files) are available for download from Zenodo (https://doi.org/10.5281/zenodo.19698794). To use them, download the files and unpack their contained folders into the `models/` folder.

## Code layout

- `mapping_to_Braun.py`: CLI for mapping to the developing human brain atlas (Braun et al. 2023)
- `mapping_to_Siletti.py`: CLI for mapping to the adult human brain atlas (Siletti et al. 2023)
- `mapping_to_Wang.py`: CLI for mapping to the developing human cerebral cortex atlas (RNA portion of the scMultiome data; Wang et al. 2025)
- `helpers/`: helper modules (`mapping_scarches.py`, `wknn.py`, `report.py`, `pipeline.py`, `cli.py`)
- `models/`: directory for the pretrained reference models (populated after downloading from Zenodo)

## Dependency installation with uv

This project requires **Python 3.10–3.13** (Python 3.12 is recommended). Create a local virtual environment with uv before installing dependencies:

```bash
uv venv --python 3.12 .venv
```

> **Earlier versions.** map2ref up to [v0.1.0](https://github.com/zhisonghe/map2ref/releases/tag/v0.1.0) ran scPoli with the original scArches package (Python 3.9, `scvi-tools==1.1.2`, `scArches==0.6.1`). Check out that release if you need the old behaviour.

This repository includes split dependency files under `requirements/`:

- `requirements/base.txt`: shared runtime dependencies
- `requirements/macos.txt`: macOS dependencies (excludes RAPIDS `cuml`)
- `requirements/linux-cuda.txt`: Linux/CUDA dependencies (includes RAPIDS `cuml-cu12`)

Install into the uv-managed environment in this folder:

```bash
# macOS
uv pip install --python .venv/bin/python -r requirements/macos.txt

# Linux with CUDA
uv pip install --python .venv/bin/python -r requirements/linux-cuda.txt
```

Key version requirements:

- `scvi-tools>=1.5.2,<1.6`
- `scpolivi` (installed from https://github.com/zhisonghe/scpoli_scvi-tools)
- `igraph` and `leidenalg` (Leiden initialisation of unlabeled scPoli prototypes)
- `umap-learn<0.6` (umap-learn 0.6.0 breaks `scanpy.tl.umap` on larger data)
- `cuml-cu12` (Linux/CUDA dependency set only; optional, the pipeline falls back to CPU `pynndescent` without it)

## Alternative environment managers

conda/mamba and pixi can also be used to create the Python environment before running the `uv pip install` steps above:

```bash
# conda / mamba
conda create -n map2ref python=3.12
conda activate map2ref

# pixi
pixi init --channel conda-forge
pixi add python=3.12
```

## Usage

Activate the virtual environment first to ensure the correct Python is used:

```bash
source map2ref/.venv/bin/activate
```

### Mapping to the developing human brain atlas (Braun et al. 2023)

```bash
python map2ref/mapping_to_Braun.py \
    -q query.h5ad \
    -o output_map2Braun \
    --save-full-query \
    --epochs 100 \
    -b batch \
    --no-smooth-presence \
    --force-new-umap
```

### Mapping to the adult human brain atlas (Siletti et al. 2023)

```bash
python map2ref/mapping_to_Siletti.py \
    -q query.h5ad \
    -o output_map2Siletti \
    --save-full-query \
    --epochs 100 \
    -b batch \
    --no-smooth-presence \
    --force-new-umap
```

### Mapping to the developing human cerebral cortex atlas (Wang et al. 2025)

```bash
python map2ref/mapping_to_Wang.py \
    -q query.h5ad \
    -o output_map2Wang \
    --save-full-query \
    -b batch \
    --query-layer counts \
    --force-new-umap
```

The scPoli models (Wang, Siletti) are trained on raw counts. If `.X` of the query holds log-normalised data, point `--query-layer` to the layer with the raw counts (e.g. `counts`); the expression-scale check otherwise stops the run. Query batch names that also exist in the reference reuse that reference batch's condition embedding (a warning is printed), so give unrelated query samples distinct names.

The scPoli reference models on Zenodo are stored in the scArches format (`attr.pkl`, `model_params.pt`, `var_names.csv`). They are converted to scpolivi automatically when loaded; a model directory saved by scpolivi (`model.pt`) works as well. The query model in `<output>/model` is saved in the scvi-tools/scpolivi format.

### Arguments

| Argument | Short | Default | Description |
|---|---|---|---|
| `--query` | `-q` | — | Path to the query H5AD file (**required**) |
| `--ref` | `-r` | `models/model_Braun`, `models/model_Siletti` or `models/model_Wang` | Path to the reference model directory |
| `--output` | `-o` | `output` | Path to the output folder |
| `--save-full-query` | | off | Save a full-gene query AnnData (alongside the gene-subset `query.h5ad`) with all mapping results |
| `--query-batch-key` | `-b` | `batch` | Key in `adata.obs` identifying sample/batch for scArches surgery |
| `--query-layer` | | `None` | Layer in the query AnnData to use as expression input (use `X` to refer to `.X`); by default the layer expected by the pretrained model is used |
| `--batch-size` | | `1024` | Mini-batch size for model training |
| `--epochs` | | `200` (scANVI) / `100` (scPoli) | Number of training epochs |
| `--k-wknn` | `-k` | `100` | Number of nearest neighbours per query cell in the reference for WKNN construction |
| `--k-ref` | `-n` | `100` | Number of nearest neighbours per reference cell in the reference for label propagation |
| `--no-smooth-presence` | | off | Skip random-walk smoothing of presence scores |
| `--no-label-transfer` | | off | Skip all label transfer steps |
| `--vis-rep-query` | | `X_umap` | Key in `adata.obsm` to use as the query embedding for visualisation |
| `--force-new-umap` | | off | Always recompute UMAP of the query from the projected latent representation, ignoring any pre-existing embedding |
| `--report-type` | | `basic` | HTML report style: `basic` or `fancy` |
| `--skip-scale-check` | | off | Skip the expression-scale sanity check between query and reference |
| `--quiet` | | off | Suppress progress messages |
