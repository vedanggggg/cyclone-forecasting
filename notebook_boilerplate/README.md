# Kaggle Notebook Boilerplate for Tropical Cyclone Forecasting

This folder contains boilerplate code for replicating the "Improving Tropical Cyclone Forecasting With Video Diffusion Models" paper on Kaggle.

## Quick Start

### Cell 1: Setup
```python
# Copy entire contents of kaggle_setup.py here
exec(open('/kaggle/working/notebook_boilerplate/kaggle_setup.py').read())
```

### Cell 2: Download Cyclones
```python
from cyclone_downloader import download_cyclones
download_cyclones(["Ian", "Fiona"])
```

## Prerequisites

1. **GPU Runtime**: Settings → Accelerator → GPU P100
2. **Internet Access**: Settings → Internet → On
3. **Kaggle Secrets** (Add → Secrets):
   - `GITHUB_TOKEN` - GitHub Personal Access Token (for private repo)
   - `CDS_API_KEY` - Copernicus CDS API key (for ERA5 data)

## Available Cyclones

### North Atlantic Ocean (GOES-16 East)
- Ian, Fiona, Ida, Grace, Iota, Eta, Zeta, Delta, Sally, Laura, Bonnie

### North Pacific Ocean (GOES-17 West)
- Genevieve

### Other Regions (not yet automated in this boilerplate)
- West Pacific Ocean, Australia, North Indian Ocean, West Indian Ocean

## Files

| File | Purpose |
|------|---------|
| `kaggle_setup.py` | Environment setup, dependencies, repo cloning |
| `cyclone_downloader.py` | Download satellite & ERA5 data by cyclone name |

## Storage Estimates

| Cyclone | Duration | Satellite | ERA5 | Total |
|---------|----------|-----------|------|-------|
| Ian | 11 days | ~4 GB | ~1.1 GB | ~5 GB |
| Fiona | 10 days | ~3.5 GB | ~1 GB | ~4.5 GB |

**Kaggle Limit**: 20 GB - Download max 2-3 cyclones at once!

## Directory Structure After Setup

```
/kaggle/working/
├── data/
│   ├── goes_east/data/nc/{cyclone}/{date}/  ← GOES-16 files
│   ├── era5/data/nc/{cyclone}/              ← ERA5 files
│   ├── metadata/                            ← Generated metadata
│   └── dataloader/                          ← PyTorch dataloaders
├── outputs/
│   ├── checkpoints/
│   ├── predictions/
│   └── logs/
└── forecast-video-diffmodels/               ← Cloned repo
```

## Next Steps After Download

```python
# Generate metadata
!cd /kaggle/working/forecast-video-diffmodels/dataproc && python md1.py

# Create dataloaders
!cd /kaggle/working/forecast-video-diffmodels/dataproc && python fc1-create-dataloaders.py
```
