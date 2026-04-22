# map2ref: one-liner CLI for scArches and label transfer
This is a simple CLI pipeline to run scArches transfer learning of query scRNA-seq data to the reference with the pretrained VAE model. It includes CLI commands to do mapping of the query scRNA-seq data to the reference transcriptomic cell atlas of developing ([Braun et al. 2023](https://www.science.org/doi/10.1126/science.adf1226)) and adult ([Siletti et al. 2023](https://www.science.org/doi/10.1126/science.add7046)) human brains. The pretrained models and the reference AnnData objects (.h5ad files) will be available at Zenodo.

## Dependency installation with uv

This repository includes split dependency files under `requirements/`:

- `requirements/base.txt`: shared runtime dependencies
- `requirements/macos.txt`: macOS dependencies (excludes RAPIDS `cuml`)
- `requirements/linux-cuda.txt`: Linux/CUDA dependencies (includes `cuml==24.4`)

Code layout:

- `mapping_to_Braun.py`: root CLI entrypoint
- `helpers/`: helper modules used by the CLI (`mapping_scarches.py`, `wknn.py`, `report.py`)

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
