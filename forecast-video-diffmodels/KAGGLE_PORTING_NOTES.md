# Kaggle Porting Notes (Logic-Preserving)

## Scope
This project was originally organized for an HPC environment (`/rds/...`, PBS scripts, and fixed absolute paths).
The updates below only change path resolution and environment configuration. Model architecture, training loops, sampling, metrics, and data split logic remain unchanged.

## Directory Overview
- `forecast-video-diffmodels/dataproc/`
  - Data download, metadata creation, dataloader creation, split helpers.
  - Core shared module: `utils.py`.
- `forecast-video-diffmodels/imagen/`
  - Diffusion training/evaluation code.
  - Stage-1 entrypoint: `64_FC/train64.py`.
  - Stage-1 evaluation: `64_FC/v_t02-sampling-and-evaluation.py`, `64_FC/test64.py`.
- `forecast-video-diffmodels/Kaggle_Stage1_Replication.ipynb`
  - Kaggle notebook scaffold to run Stage-1 end-to-end.

## Files Changed
1. `forecast-video-diffmodels/imagen/helpers.py`
- Replaced hardcoded `BASE_HOME` and `BASE_DATA` defaults with environment-driven defaults.
- Added `FDM_DATAPROC_DIR` support so `utils.py` import path is portable.
- Replaced hardcoded test split path with `FDM_TEST_SET_PATH` (defaulting to local `dataproc/test_set.pkl`).

2. `forecast-video-diffmodels/dataproc/utils.py`
- Replaced hardcoded `BASE_HOME` with env-driven default.
- Replaced fixed `imagen` path with candidate lookup via `FDM_IMAGEN_DIR` and local project path.
- Updated `Cyclone` metadata default path to use `METADATA_DIR`/`BASE_DATA` fallback instead of `/vol/...`.

3. `forecast-video-diffmodels/dataproc/fc2-pytorch-dataloader.py`
- Added env support:
  - `DATALOADER_DIR` for input `.dat` directory.
  - `TEST_SET_PATH` for split mapping file.

4. `forecast-video-diffmodels/dataproc/pr1-create-dataloaders.py`
- Added `DATALOADER_DIR` override for output directory.

5. `forecast-video-diffmodels/dataproc/sr1-create-dataloaders.py`
- Added `DATALOADER_DIR` override for output directory.

## What Is Different vs Original Author Setup
- Infrastructure only:
  - HPC/PBS job launching replaced by notebook cells.
  - Absolute filesystem paths replaced by environment variables.
  - Data/checkpoint directories point to Kaggle writable paths.
- Not changed:
  - Training objectives.
  - Two-stage switching behavior.
  - Model definitions and hyperparameters in the training scripts.
  - Metric computations and sampling logic.

## Kaggle Required Environment Variables
Set these in notebook before running Stage-1:

```bash
export FDM_PROJECT_ROOT=/kaggle/working/forecast-video-diffmodels
export FDM_BASE_HOME=/kaggle/working
export FDM_BASE_DATA=/kaggle/working
export FDM_DATAPROC_DIR=/kaggle/working/forecast-video-diffmodels/dataproc
export FDM_IMAGEN_DIR=/kaggle/working/forecast-video-diffmodels/imagen
export FDM_TEST_SET_PATH=/kaggle/working/forecast-video-diffmodels/dataproc/test_set.pkl
```

## Files Not Needed for Kaggle Stage-1 Run
You can ignore these for Stage-1 replication in notebook:
- `*.pbs`
- `gpu_cluster*.sh`
- download notebooks/scripts in `dataproc/` if data is already prepared

