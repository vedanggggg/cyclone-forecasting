# Cyclone Forecasting Preprocessing and Replication Audit (Kaggle)

This document explains the preprocessing pipeline end-to-end, maps each step to repository code, and audits what matches the authors versus what was changed for Kaggle execution.

It is intentionally explicit and operational so a new contributor can reproduce without guessing.

## 1) Scope and Inputs

Current Kaggle inputs used in your run:

- `/kaggle/input/final-dataset-with-splits/dataloaders_fixed/*.dat` (15 cyclone files)
- `/kaggle/input/final-dataset-with-splits/split_dataloaders/train_loader.pkl`
- `/kaggle/input/final-dataset-with-splits/split_dataloaders/test_loader.pkl`

Important: these are already processed artifacts (not raw satellite downloads).  
So your Kaggle notebook is reproducing from processed `.dat` artifacts, not regenerating everything from raw files.

## 2) Canonical Author Pipeline (from repo code)

### 2.1 Raw data + metadata alignment

Code reference: `dataproc/md1.py`

- Reads cyclone list from `dataproc/list_of_cyclones.xlsx`.
- For each cyclone:
  - Picks a fixed map center and half-side length by basin/cyclone special cases.
  - Loads ERA5 files and computes map crop bounds.
  - Enumerates IR source files per basin:
    - `nio` -> INSAT h5
    - `aus/wpo` -> Himawari bz2 dirs
    - `wio` -> MSG `.nat`
    - `use/usw` -> GOES nc
  - Parses timestamps per source format.
  - Matches IR time to ERA5 `time` index (`era5_idx`).
  - Stores metadata dict to `<region>_<name>.metadata`.

Relevant lines:

- `dataproc/md1.py:19` to `dataproc/md1.py:134` (core metadata assembly)
- `dataproc/md1.py:117` to `dataproc/md1.py:126` (ERA5 timestamp matching)
- `dataproc/md1.py:149` (writes metadata file)

### 2.2 FC dataloader construction (`.dat`)

Code reference: `dataproc/fc1-create-dataloaders.py`

Per cyclone, for each timestamp `t` (starting from index 1):

- Loads current IR image at `t`.
- Replaces NaNs with zeros.
- Makes square crop.
- Resizes IR to `o_size` (64 by default) as `img_o`.
- Uses placeholder `img_n` as zeros `(1, 128, 128)` for FC mode.
- Loads ERA5 at matched `era5_idx`, resizes to `(3, 64, 64)`.
- Loads previous IR at `t-1`, preprocesses similarly.
- Concatenates previous IR channel + ERA5 channels => `era5` with 4 channels.
- Appends sample into `CycloneDataLoader(mode="fc")`.
- Pickles to `<region>_<name>.dat`.

Relevant lines:

- `dataproc/fc1-create-dataloaders.py:67` and `dataproc/fc1-create-dataloaders.py:85` (NaN -> 0)
- `dataproc/fc1-create-dataloaders.py:93` (4-channel conditioning construction)
- `dataproc/fc1-create-dataloaders.py:106` (writes `.dat`)

### 2.3 In-memory dataloader structures

Code reference: `dataproc/utils.py`

- `CycloneDataLoader(mode="fc")` stores:
  - `img_64`: `(N, 64, 64)`
  - `img_128`: `(N, 128, 128)` (placeholder zeros in FC script)
  - `era5`: `(N, 4, 64, 64)`
- `v_ModelDataLoader` converts per-sample arrays into:
  - image mode tensors (`img`, `img_cond`, `era5_img`)
  - video mode tensors (`vid`, `vid_cond`, `era5_vid`)
- Video sequence length hard-coded as `t = 10`.

Relevant lines:

- `dataproc/utils.py:405` to `dataproc/utils.py:435` (`CycloneDataLoader`)
- `dataproc/utils.py:544` (`t = 10`)
- `dataproc/utils.py:620` to `dataproc/utils.py:646` (builds 10-frame windows + image tensors)

### 2.4 Train/test split in author notebook

Code reference: `dataproc/MD3 - Train Test Bifurcation.ipynb`

- Uses random sampling with seed 42.
- Samples `round(0.2 * cyclones_in_region)` per region into test set.
- Writes `test_set.pkl` mapping `{region_abbv: [cyclone names]}`.

This split is cyclone-level, not frame-level.

## 3) Kaggle Replication Pipeline You Ran

Because you started from prepared `.dat` + split pickles, the Kaggle path is:

1. Validate dataset paths exist.
2. Clone repo + dependencies.
3. Recreate author expected filesystem layout (`/rds/...`) for path compatibility.
4. Copy `.dat` files into run location.
5. Build `test_set.pkl` by matching counts from provided `train_loader.pkl` and `test_loader.pkl`.
6. Build additional validation split from train set:
   - `val_set.pkl`
   - `fit_holdout_set.pkl` (val + test) so training excludes both.
7. Train with `FDM_TEST_SET_PATH=fit_holdout_set.pkl`.
8. Validate every 4 epochs with `FDM_TEST_SET_PATH=val_set.pkl`.
9. Final test evaluation with `FDM_TEST_SET_PATH=test_set.pkl`.

## 4) What The Counts Mean

You have 15 cyclone files, but split counts are sample counts:

- `img` counts are frame-level training samples after dataloader aggregation.
- `vid` counts are 10-frame windows (`floor(sample_count / 10)` per cyclone stream).

Example from your run:

- Train image samples: `2229`
- Test image samples: `816`
- Train video windows: `216`
- Test video windows: `81`

So 15 cyclones is consistent with thousands of frame samples.

## 5) Metrics Pipeline

Metrics implementation is in `imagen/helpers.py`:

- `rmse`, `mae`, `psnr`, `ssim`, `fid`, `fvd`, `lpips` are computed in `calculate_metrics`.
- Validation/test scripts write `metrics_test.pkl`.

Relevant lines:

- `imagen/helpers.py:249` to `imagen/helpers.py:265` (`calculate_metrics`)

Your current validation loop logs these every 4 epochs to W&B:

- `val/mae`
- `val/psnr`
- `val/ssim`
- `val/fid`
- `val/fvd`
- `val/rmse`
- `val/lpips`

## 6) Match vs Difference Audit

### 6.1 Matches authors/repo intent

- FC preprocessing uses previous IR + ERA5 conditioning channels (`4` channels total).
- NaN replacement with zeros in dataloader creation logic.
- 10-frame video sequence handling (`t=10`).
- Batch size set to 1 in training/eval scripts.
- Learning rate `3e-4`.

### 6.2 Differences introduced in your Kaggle replication

These are real differences from an untouched original environment:

1. **Dataset scope**
   - You are using 15 cyclones, not the full paper global dataset.
2. **Split reconstruction method**
   - `test_set.pkl` reconstructed from counts in provided split pickles.
   - Multiple valid cyclone subsets existed; deterministic tie-break used.
3. **Validation split addition**
   - Added `val_set.pkl` and `fit_holdout_set.pkl` for periodic validation.
   - This is not the exact original MD3 train/test-only flow.
4. **Kaggle path adaptation**
   - `/rds/...` symlink emulation and env var path routing.
5. **Imagen API compatibility layer**
   - Added runtime compatibility shim because public `imagen-pytorch` API differs from author-used API.
6. **Observability changes**
   - W&B integration + periodic monitoring/logging cells were added.

### 6.3 Unknown / cannot be asserted from current artifacts

- Exact original cyclone identity for test split in paper runs (if not explicitly listed).
- Whether your current `.dat` files were generated with identical software versions to author environment.
- Exact source cadence used in your `.dat` generation for every cyclone (hourly vs otherwise) without inspecting raw-time metadata per file.

## 7) Hard Verification Checklist (No Guesswork)

Run these checks in notebook:

1. Confirm file naming and basin mapping:
   - `<cyclone>_<region>.dat` in dataset input.
   - canonical copy renamed to `<region>_<cyclone>.dat` for repo loaders.
2. Confirm split files:
   - `test_set.pkl` exists.
   - `val_set.pkl` exists.
   - `fit_holdout_set.pkl` exists.
3. Confirm counts:
   - `train + val + test == total samples`.
4. Confirm train phase uses holdout split:
   - env var `FDM_TEST_SET_PATH=fit_holdout_set.pkl`.
5. Confirm val phase uses validation split:
   - env var `FDM_TEST_SET_PATH=val_set.pkl`.
6. Confirm final test uses test split only:
   - env var `FDM_TEST_SET_PATH=test_set.pkl`.
7. Confirm checkpoint selection is numeric (not lexical sort).

## 8) Recommended Documentation Artifacts to Keep with Results

For reproducibility, store with each run:

- Run config cell (dataset path, epochs, val frequency, run name).
- Exact `test_set.pkl`, `val_set.pkl`, `fit_holdout_set.pkl`.
- W&B run URLs for:
  - train run
  - val run
  - final test run
- Commit hash of repo snapshot used for run.

## 9) Decisions Needed From You (Required for strict “exact replication” claim)

To declare strict equivalence confidently, these must be explicitly fixed by you:

1. Should test cyclones be the deterministic reconstructed subset currently used, or a specific named subset you choose?
2. Should validation split remain enabled for model selection, or should we keep pure original train/test-only behavior for strict baseline?
3. Do you want to freeze the current Kaggle notebook as canonical and tie it to a git commit hash now?

