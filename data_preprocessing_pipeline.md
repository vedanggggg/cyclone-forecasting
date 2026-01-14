
 **Data Processing Pipeline**. They bridge the gap between the raw satellite/weather files you just downloaded and the AI model that needs to be trained.

1. **`FC1 - Create Dataloaders`**: This is the **Builder**.
* It reads the raw files (NetCDF) for each cyclone.
* It pairs the satellite images with the corresponding ERA5 weather data (wind, pressure, etc.).
* It resizes/crops them to  pixels (or ).
* **Result:** It saves processed `.dat` files (pickled datasets) that are fast to load.


2. **`FC2 - Pytorch Dataloader`**: This is the **Loader**.
* It reads the `.dat` files created by FC1.
* It prepares them for PyTorch (the AI framework) to use during training.


3. **`fc1.pbs`**: This is a **Cluster Job Script**. It is used to run these tasks on a university supercomputer. **You do not need this on Kaggle.**

---

### **How to run this on Kaggle**

The original code relies on `utils.py` and hardcoded paths like `/rds/general/...`. To make this work on Kaggle, we need to **set environment variables** that tell the scripts to look in your `/kaggle/working/` folder instead.

#### **Step 1: Setup Environment & Paths**

Run this cell first to tell Python where your downloaded data lives.

```python
import os
import sys

# 1. Add current directory to path so we can import utils
sys.path.append("/kaggle/working/")

# 2. OVERRIDE PATHS: Tell utils.py to look in Kaggle directories
# These match the folders we created in previous steps
os.environ["ERA5_DIR"] = "/kaggle/working/era5_downloads"
os.environ["GOES_EAST_DIR"] = "/kaggle/working/goes_east_downloads"
os.environ["GOES_WEST_DIR"] = "/kaggle/working/goes_west_downloads"
os.environ["MSG_DIR"] = "/kaggle/working/meteosat_cropped"
os.environ["HIMAWARI_DIR"] = "/kaggle/working/himawari_downloads" # If you have this
os.environ["INSAT_DIR"] = "/kaggle/working/insat_downloads"       # If you have this

# 3. OUTPUT DIRECTORY: Where the processed dataloaders will be saved
output_dir = "/kaggle/working/dataloaders/64_FC/"
os.makedirs(output_dir, exist_ok=True)
os.environ["DATALOADER_DIR"] = output_dir

# 4. Set Image Size (64x64 as per the filename)
os.environ["O_SIZE"] = "64"

print("✅ Environment paths configured for Kaggle.")

```

#### **Step 2: The "Fixed" FC1 Script (Create Dataloaders)**

I have modified the code to remove email sending, fix imports, and add a selection menu so you don't process every empty folder.

**Prerequisite:** Ensure `utils.py` is in your Kaggle output directory. If it's in a dataset, copy it: `!cp /kaggle/input/path/to/utils.py .`

```python
import pandas as pd
import torch
import skimage.transform
import pickle
import numpy as np
import glob
import warnings
from tqdm.notebook import tqdm
import os

# Import your utils (Make sure utils.py is in the working directory)
try:
    from utils import Cyclone, ModelDataLoader
except ImportError:
    print("❌ ERROR: utils.py not found. Please upload utils.py to /kaggle/working/")
    raise

warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
BASE_DIR = os.environ["DATALOADER_DIR"]
O_SIZE = int(os.environ["O_SIZE"])

# Define specific cyclones you have actually downloaded data for
# (Add the ones you downloaded in previous steps here)
TARGET_CYCLONES = ["Roslyn", "Orlene", "Ian", "Fiona", "Freddy"] 

def transform_make_sq_image(img):
    """Helper to square crop images from center"""
    h, w = img.shape
    min_dim = min(h, w)
    start_x = (w - min_dim) // 2
    start_y = (h - min_dim) // 2
    return img[start_y:start_y+min_dim, start_x:start_x+min_dim]

def process_cyclone(name, region):
    """
    Main logic to load raw data, pair it, and save the .dat file
    """
    filename = f"{region}_{name}.dat"
    save_path = os.path.join(BASE_DIR, filename)
    
    if os.path.exists(save_path):
        print(f"[{name}] Already processed. Skipping.")
        return

    print(f"[{name}] Loading Data...")
    
    try:
        # Initialize Cyclone helper from utils.py
        cyclone = Cyclone(region, name)
        
        # Load ERA5 Data (This might fail if you didn't download ERA5 for this storm)
        try:
            cyclone.load_era5()
        except Exception as e:
            print(f"[{name}] ⚠️ Failed to load ERA5: {e}. Skipping.")
            return

        data_loader = CycloneDataLoader(name, region) # Helper container
        
        # Get intersection of dates (Satellite + ERA5)
        # This relies on Cyclone class internals to find matching files
        common_dates = cyclone.get_common_dates() 
        
        if not common_dates:
            print(f"[{name}] No matching dates found between Sat and ERA5.")
            return

        print(f"[{name}] Processing {len(common_dates)} time steps...")

        for date in tqdm(common_dates, leave=False):
            # 1. Get ERA5 tensor for this timestamp
            era5 = cyclone.get_era5_tensor(date)
            
            # 2. Get Satellite Image
            # Note: utils.py likely expects specific channel names (IR108, etc.)
            sat_img_raw = cyclone.get_sat_image(date) 
            
            if sat_img_raw is None: continue

            # 3. Preprocessing (Resize/Crop)
            img = np.nan_to_num(sat_img_raw, nan=0.0)
            img = transform_make_sq_image(img)
            img = skimage.transform.resize(img, (O_SIZE, O_SIZE), anti_aliasing=True)
            
            # Convert to Tensor (1, 64, 64)
            img_tensor = torch.from_numpy(img).float().unsqueeze(0)
            
            # Combine
            # (Logic adapted from your snippet: pair current img with ERA5)
            # Depending on your model, you might stack them or keep separate
            
            # Simple check for NaNs
            if torch.isnan(img_tensor).any() or torch.isnan(era5).any():
                continue
                
            data_loader.add(img_tensor, era5)

        # Save result
        with open(save_path, 'wb') as f:
            pickle.dump(data_loader, f)
            
        print(f"[{name}] ✅ Saved to {save_path}")

    except Exception as e:
        print(f"[{name}] ❌ Error: {e}")

# --- HELPER CLASS (If not in utils) ---
class CycloneDataLoader:
    def __init__(self, name, region):
        self.name = name
        self.region = region
        self.data = []
    def add(self, img, era5):
        self.data.append((img, era5))

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Ensure you have your list_of_cyclones CSV loaded to map Name -> Region
    # Or just hardcode it for the ones you have:
    
    MY_DATA = [
        {"Name": "Roslyn", "Region": "North Pacific Ocean"},
        {"Name": "Orlene", "Region": "North Pacific Ocean"},
        {"Name": "Ian",    "Region": "North Atlantic Ocean"},
        {"Name": "Freddy", "Region": "West Indian Ocean"},
    ]
    
    print(f"Starting Processing for {len(MY_DATA)} cyclones...")
    
    for row in MY_DATA:
        if row["Name"] in TARGET_CYCLONES:
            process_cyclone(row["Name"], row["Region"])
            
    print("\nDONE. Check Output folder for .dat files.")

```

#### **Step 3: Verification (FC2)**

Once the `.dat` files are generated, run this to verify they are loadable.

```python
import glob
import pickle
import torch
from torch.utils.data import Dataset, DataLoader

class SimpleDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

print("Verifying generated dataloaders...")
dat_files = glob.glob(os.environ["DATALOADER_DIR"] + "/*.dat")

if not dat_files:
    print("No .dat files found. Did FC1 run successfully?")
else:
    for fn in dat_files:
        try:
            with open(fn, "rb") as f:
                loader_obj = pickle.load(f)
                
            # Create a PyTorch Loader
            ds = SimpleDataset(loader_obj.data)
            dl = DataLoader(ds, batch_size=4, shuffle=True)
            
            # Grab one batch
            imgs, era5 = next(iter(dl))
            print(f"✅ {loader_obj.name}: Batch Shape Img={imgs.shape}, ERA5={era5.shape}")
            
        except Exception as e:
            print(f"❌ Failed to load {fn}: {e}")

```