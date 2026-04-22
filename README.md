# map2ref: one-liner CLI for scArches and label transfer
This is a simple CLI pipeline to run scArches transfer learning of query scRNA-seq data to the reference with the pretrained VAE model. It includes CLI commands to do mapping of the query scRNA-seq data to the reference transcriptomic cell atlas of developing ([Braun et al. 2023](https://www.science.org/doi/10.1126/science.adf1226)) and adult ([Siletti et al. 2023](https://www.science.org/doi/10.1126/science.add7046)) human brains. The pretrained models and the reference AnnData objects (.h5ad files) are available for download from Zenodo (https://doi.org/10.5281/zenodo.19698794). To use them, download the files and unpack their contained folders into the `models/` folder.

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
