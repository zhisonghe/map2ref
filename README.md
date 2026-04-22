# map2ref: one-liner CLI for scArches and label transfer
This is a CLI pipeline to map query scRNA-seq data to a pretrained single-cell brain atlas reference. For each atlas, the pipeline:

1. **Projects the query** into the reference latent space via scArches surgery on the pretrained VAE model (scANVI for Braun, scPoli for Siletti)
2. **Calculates presence scores** using a weighted k-nearest-neighbour (WKNN) graph, with optional random-walk smoothing, to quantify how well each reference cell is represented in the query
3. **Transfers labels** from the reference to the query via the WKNN graph; for Braun mapping this includes cell class, subregion, neuron transmitter type (NTT), and a hierarchical region label that resolves each cell to the most specific unambiguous brain region in the developmental hierarchy; for Siletti mapping this includes ROI group, fine ROI group, cell type, and supercluster term
4. **Generates an HTML report** with UMAP visualisations of the mapping results

It includes CLI commands for mapping to the transcriptomic cell atlas of the developing ([Braun et al. 2023](https://www.science.org/doi/10.1126/science.adf1226)) and adult ([Siletti et al. 2023](https://www.science.org/doi/10.1126/science.add7046)) human brain. The pretrained models and the reference AnnData objects (.h5ad files) are available for download from Zenodo (https://doi.org/10.5281/zenodo.19698794). To use them, download the files and unpack their contained folders into the `models/` folder.

## Code layout

- `mapping_to_Braun.py`: CLI for mapping to the developing human brain atlas (Braun et al. 2023)
- `mapping_to_Siletti.py`: CLI for mapping to the adult human brain atlas (Siletti et al. 2023)
- `helpers/`: helper modules (`mapping_scarches.py`, `wknn.py`, `report.py`, `pipeline.py`, `cli.py`)
- `models/`: directory for the pretrained reference models (populated after downloading from Zenodo)

## Dependency installation with uv

This project requires **Python 3.9**. A newer version is not used because Python 3.9 is the highest version compatible with the scArches package. Create a local virtual environment with uv before installing dependencies:

```bash
uv venv --python 3.9 .venv
```

This repository includes split dependency files under `requirements/`:

- `requirements/base.txt`: shared runtime dependencies
- `requirements/macos.txt`: macOS dependencies (excludes RAPIDS `cuml`)
- `requirements/linux-cuda.txt`: Linux/CUDA dependencies (includes `cuml==24.4`)

Install into the uv-managed environment in this folder:

```bash
# macOS
uv pip install --python .venv/bin/python -r requirements/macos.txt

# Linux with CUDA
uv pip install --python .venv/bin/python -r requirements/linux-cuda.txt
```

Pinned versions required by this project are included:

- `setuptools<=81.0.0`
- `scvi-tools==1.1.2`
- `scArches==0.6.1`
- `cuml==24.4` (Linux/CUDA dependency set)

## Alternative environment managers

conda/mamba and pixi can also be used to create the Python 3.9 environment before running the `uv pip install` steps above:

```bash
# conda / mamba
conda create -n map2ref python=3.9
conda activate map2ref

# pixi
pixi init --channel conda-forge
pixi add python=3.9
```

## Usage

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

### Arguments

| Argument | Short | Default | Description |
|---|---|---|---|
| `--query` | `-q` | — | Path to the query H5AD file (**required**) |
| `--ref` | `-r` | `models/model_Braun` or `models/model_Siletti` | Path to the reference model directory |
| `--output` | `-o` | `output` | Path to the output folder |
| `--save-full-query` | | off | Save a full-gene query AnnData (alongside the gene-subset `query.h5ad`) with all mapping results |
| `--query-batch-key` | `-b` | `batch` | Key in `adata.obs` identifying sample/batch for scArches surgery |
| `--batch-size` | | `1024` | Mini-batch size for model training |
| `--epochs` | | `200` | Number of training epochs |
| `--k-wknn` | `-k` | `100` | Number of nearest neighbours per query cell in the reference for WKNN construction |
| `--k-ref` | `-n` | `100` | Number of nearest neighbours per reference cell in the reference for label propagation |
| `--no-smooth-presence` | | off | Skip random-walk smoothing of presence scores |
| `--no-label-transfer` | | off | Skip all label transfer steps |
| `--vis-rep-query` | | `X_umap` | Key in `adata.obsm` to use as the query embedding for visualisation |
| `--force-new-umap` | | off | Always recompute UMAP of the query from the projected latent representation, ignoring any pre-existing embedding |
| `--report-type` | | `basic` | HTML report style: `basic` or `fancy` |
| `--skip-scale-check` | | off | Skip the expression-scale sanity check between query and reference |
| `--quiet` | | off | Suppress progress messages |
