"""
Kaggle Notebook Boilerplate for Tropical Cyclone Forecasting Replication
=========================================================================

Copy and paste this entire file into the FIRST CELL of a new Kaggle notebook.
Run the cell to set up the environment.

Prerequisites:
1. GPU runtime enabled (Settings -> Accelerator -> GPU)
2. Internet enabled (Settings -> Internet -> On)
3. CDS API key added as Kaggle Secret named 'CDS_API_KEY' (for ERA5 data)
   - Get it from: https://cds.climate.copernicus.eu/user (after registration)

Usage:
    cell 1: Paste this file and run
    cell 2: from cyclone_downloader import download_cyclones
            download_cyclones(["Ian", "Fiona"])
"""

import os
import sys
import subprocess
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

REPO_URL = "https://github.com/Vedang-P/tc-weather-forecasting.git"
WORK_DIR = Path("/kaggle/working")
DATA_DIR = WORK_DIR / "data"
REPO_DIR = WORK_DIR / "tc-weather-forecasting"  # Cloned repo root
CODE_DIR = REPO_DIR / "forecast-video-diffmodels"  # Actual code folder
OUTPUT_DIR = WORK_DIR / "outputs"

# Create directory structure
DIRS_TO_CREATE = [
    DATA_DIR / "goes_east" / "data" / "nc",
    DATA_DIR / "goes_west" / "data" / "nc", 
    DATA_DIR / "era5" / "data" / "nc",
    DATA_DIR / "metadata",
    DATA_DIR / "dataloader",
    OUTPUT_DIR / "checkpoints",
    OUTPUT_DIR / "predictions",
    OUTPUT_DIR / "logs",
]

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def print_status(emoji, message):
    """Print a formatted status message."""
    print(f"\n{emoji} {message}")
    print("=" * 60)

def run_cmd(cmd, check=True, capture=False):
    """Run a shell command with optional output capture."""
    if capture:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout.strip()
    else:
        subprocess.run(cmd, shell=True, check=check)

def check_kaggle_environment():
    """Verify we're running on Kaggle."""
    if not os.path.exists("/kaggle"):
        print("⚠️  Warning: Not running on Kaggle. Some features may not work.")
        return False
    return True

# =============================================================================
# SETUP FUNCTIONS
# =============================================================================

def setup_cds_credentials():
    """Set up CDS API credentials from Kaggle Secrets."""
    print_status("🔑", "Setting up CDS API credentials")
    
    try:
        from kaggle_secrets import UserSecretsClient
        secrets = UserSecretsClient()
        cds_api_key = secrets.get_secret("CDS_API_KEY")
        
        # Write .cdsapirc file
        cdsapirc_path = Path.home() / ".cdsapirc"
        with open(cdsapirc_path, "w") as f:
            f.write(f"url: https://cds.climate.copernicus.eu/api\n")
            f.write(f"key: {cds_api_key}\n")
        
        print("✅ CDS API credentials configured successfully!")
        return True
        
    except Exception as e:
        print(f"⚠️  Could not set up CDS credentials: {e}")
        print("   ERA5 data download will not work.")
        print("   To fix: Add your CDS API key as a Kaggle Secret named 'CDS_API_KEY'")
        return False

def install_dependencies():
    """Install required Python packages."""
    print_status("📦", "Installing dependencies")
    
    packages = [
        "awscli",           # For GOES data from AWS
        "cdsapi",           # For ERA5 data
        "satpy",            # Satellite data processing
        "pyproj",           # Coordinate transformations
        "xarray",           # NetCDF handling
        "h5netcdf",         # HDF5/NetCDF support
        "scikit-image",     # Image processing
        "einops",           # Tensor operations
        "openpyxl",         # Excel file reading
        "fsspec",           # File system operations
        "tqdm",             # Progress bars
    ]
    
    for pkg in packages:
        print(f"  Installing {pkg}...")
        run_cmd(f"pip install -q {pkg}", check=False)
    
    print("✅ Dependencies installed!")

def clone_repository():
    """Clone the tc-weather-forecasting private repository."""
    print_status("📥", "Cloning repository")
    
    if REPO_DIR.exists():
        print(f"  Repository already exists at {REPO_DIR}")
    else:
        # Get GitHub token from Kaggle Secrets for private repo
        try:
            from kaggle_secrets import UserSecretsClient
            secrets = UserSecretsClient()
            github_token = secrets.get_secret("GITHUB_TOKEN")
            auth_url = REPO_URL.replace("https://", f"https://{github_token}@")
            run_cmd(f"git clone {auth_url} {REPO_DIR}")
            print(f"✅ Private repository cloned to {REPO_DIR}")
        except Exception as e:
            print(f"⚠️  Could not get GitHub token: {e}")
            print("   Add your GitHub Personal Access Token as Kaggle Secret 'GITHUB_TOKEN'")
            print("   Generate at: https://github.com/settings/tokens")
            return False
    
    # Add repo to Python path (actual code is in forecast-video-diffmodels subfolder)
    sys.path.insert(0, str(CODE_DIR / "imagen"))
    sys.path.insert(0, str(CODE_DIR / "dataproc"))

def create_directories():
    """Create the required directory structure."""
    print_status("📁", "Creating directory structure")
    
    for dir_path in DIRS_TO_CREATE:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"  Created: {dir_path}")
    
    print("✅ Directory structure ready!")

def patch_hardcoded_paths():
    """Create environment variables to override hardcoded paths."""
    print_status("🔧", "Patching hardcoded paths")
    
    # Set environment variables that scripts can use
    os.environ["BASE_HOME"] = str(CODE_DIR)
    os.environ["BASE_DATA"] = str(DATA_DIR)
    os.environ["GOES_EAST_DIR"] = str(DATA_DIR / "goes_east")
    os.environ["GOES_WEST_DIR"] = str(DATA_DIR / "goes_west")
    os.environ["ERA5_DIR"] = str(DATA_DIR / "era5")
    os.environ["METADATA_DIR"] = str(DATA_DIR / "metadata")
    os.environ["DATALOADER_DIR"] = str(DATA_DIR / "dataloader")
    os.environ["OUTPUT_DIR"] = str(OUTPUT_DIR)
    
    print("✅ Environment variables set!")
    print(f"   BASE_DATA = {DATA_DIR}")
    print(f"   OUTPUT_DIR = {OUTPUT_DIR}")

def print_summary():
    """Print a summary of the setup."""
    print_status("🎉", "Setup Complete!")
    
    print("""
Next Steps:
-----------
1. Download cyclone data (in a new cell):
   
   from cyclone_downloader import download_cyclones
   download_cyclones(["Ian", "Fiona"])

2. Or download manually:
   
   !python /kaggle/working/forecast-video-diffmodels/dataproc/am1-goes-eamerica-cyclones-data-download.py

Directory Structure:
--------------------
/kaggle/working/
├── data/
│   ├── goes_east/data/nc/    <- GOES-16 satellite data
│   ├── era5/data/nc/         <- ERA5 reanalysis data
│   ├── metadata/             <- Generated metadata files
│   └── dataloader/           <- PyTorch dataloaders
├── outputs/
│   ├── checkpoints/          <- Model checkpoints
│   ├── predictions/          <- Generated predictions
│   └── logs/                 <- Training logs
└── tc-weather-forecasting/       <- Cloned repository
    └── forecast-video-diffmodels/ <- Actual code
""")

# =============================================================================
# MAIN SETUP
# =============================================================================

def run_setup():
    """Run the complete setup process."""
    print("\n" + "=" * 60)
    print("🌀 TROPICAL CYCLONE FORECASTING - KAGGLE SETUP")
    print("=" * 60)
    
    check_kaggle_environment()
    install_dependencies()
    clone_repository()
    create_directories()
    setup_cds_credentials()
    patch_hardcoded_paths()
    print_summary()
    
    return {
        "repo_dir": REPO_DIR,
        "code_dir": CODE_DIR,
        "data_dir": DATA_DIR,
        "output_dir": OUTPUT_DIR,
    }

# Run setup when this file is executed
if __name__ == "__main__":
    config = run_setup()
else:
    # Also run when imported (pasted into notebook cell)
    config = run_setup()
