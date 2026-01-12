# Copilot instructions for forecast-video-diffmodels

Purpose: help an AI coding agent become productive quickly in this repository. Keep answers concise and follow the repository conventions.

Top-level overview
- This project implements video diffusion models for tropical cyclone forecasting. Major directories:
  - `dataproc/` : data download, preprocessing and dataloader creation (scripts and notebooks: `*create-dataloaders.py`, `era5-data-download.py`, etc.).
  - `imagen/`  : model implementation, training, sampling and evaluation (variants in `64_128/`, `64_FC/`). Contains training scripts (`m01_*.py`), sampling/eval scripts (`t02-*.py`, `t07-*.py`) and PBS job scripts for cluster runs.
  - `dataviz/`  : notebooks and helper scripts for visualization.

Essential patterns and conventions
- Data sources and paths: production data and metadata in this codebase are expected at absolute paths used in code (examples in `dataproc/utils.py`: `BASE_DIR = "/vol/bitbucket/pn222/satellite/metadata"`). Many model checkpoints are loaded from absolute cluster paths (e.g. `/vol/bitbucket/pn222/models/<run_name>/models/...`).
- Region naming: see `dataproc/utils.py` `abbv_to_region` / `region_to_abbv` maps used everywhere; file naming uses `abbv_region_<cyclonename>.metadata`.
- Model run names / checkpoint naming: run names like `64_FC_rot904_sep_3e-4` are used to look up best epoch numbers in code (`best_epoch_dict` in multiple model wrapper classes). Avoid changing those strings unless you update every mapping.
- Data loaders: scripts that create dataloaders are `*create-dataloaders.py` under `dataproc/`. The `ModelDataLoader` and `CycloneDataLoader` types live in `dataproc/utils.py` and are lightweight in-memory loaders.

Developer workflows (how to run things)
- Environment: use the repository README instructions. Recommended: conda env `research_env` with Python 3.10 and install `imagen/requirements.txt`.
- Data download: run the notebooks or python scripts in `dataproc/` (e.g. `era5-data-download.py`, `am1-goes-eamerica-cyclones-data-download.py`). They create metadata files and lists referenced by dataloaders.
- Create dataloaders: run `fc1-create-dataloaders.py`, `pr1-create-dataloaders.py`, etc. These produce serialized dataloaders/metadata used for training.
- Training & cluster jobs: training is usually run via PBS scripts in `imagen/64_128/` (e.g. `gpu_cluster.sh`, `cx2_*.pbs`) or `m01_*.py` for single-job debugging. Check scripts for specific hyperparameters.
- Sampling/evaluation: sampling and evaluation scripts include `t02-sampling-and-evaluation.py`, `t07-test-set-evaluation.py`. FVD metrics require placing `common_metrics_on_video_quality` into `imagen/64_FC` (see README).

Integration & external deps
- External repos referenced: `imagen-pytorch` (used via `sys.path.append` in `dataproc/utils.py` or via `imagen/` local code), and `common_metrics_on_video_quality` for FVD.
- Cluster-specific assumptions: hard-coded paths and PBS scripts expect a cluster filesystem and GPUs. When running locally, set path and checkpoint variables to your local copies.

What to look for when editing or adding features
- Be explicit when changing strings that correspond to run names / checkpoint lookups (`best_epoch_dict`) or file naming conventions — these are used across scripts.
- Prefer adding config variables rather than editing absolute paths inline. Introduce a small `config.py` or `.env` and load it in scripts that currently use hard-coded `/vol/bitbucket/...` paths.
- Keep notebook analyses (in `dataproc/` and `dataviz/`) as reproductions of pipeline steps; convert stable analysis into scripts when you need automation.

Key files to read first
- `README.md` (project overview and environment setup)
- `dataproc/utils.py` (data handling, model wrappers, data loader types)
- `imagen/64_128/` and `imagen/64_FC/` (training, sampling and PBS job scripts)
- `dataproc/*create-dataloaders.py` (how training data is assembled)

If something is unclear
- Ask for the target environment (local vs cluster) and whether you have access to model checkpoints or large datasets. If running local experiments, I will add configurable path overrides and reproduce a small-data example.

— End of file —
