"""
Cyclone Data Downloader for Kaggle
==================================

This module provides functions to download satellite and ERA5 data
for specific cyclones. Use after running kaggle_setup.py.

Usage:
    from cyclone_downloader import download_cyclones
    download_cyclones(["Ian", "Fiona"])  # Download Ian and Fiona
"""

import os
import sys
import glob
import datetime
import subprocess
import pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm

# =============================================================================
# CONFIGURATION
# =============================================================================

# Kaggle paths
WORK_DIR = Path("/kaggle/working")
DATA_DIR = WORK_DIR / "data"
REPO_DIR = WORK_DIR / "forecast-video-diffmodels"

# Satellite data directories
GOES_EAST_DIR = DATA_DIR / "goes_east"
GOES_WEST_DIR = DATA_DIR / "goes_west"
ERA5_DIR = DATA_DIR / "era5"

# Cyclone list
CYCLONES_PATH = REPO_DIR / "dataproc" / "list_of_cyclones.xlsx"

# Region to satellite mapping
REGION_SATELLITE_MAP = {
    "North Atlantic Ocean": ("GOES-16 East", GOES_EAST_DIR, "am1"),
    "North Pacific Ocean": ("GOES-17 West", GOES_WEST_DIR, "am2"),
    "West Pacific Ocean": ("Himawari", DATA_DIR / "himawari", "jp1"),
    "Australia": ("Himawari", DATA_DIR / "himawari", "jp1"),
    "West Indian Ocean": ("MSG SEVIRI", DATA_DIR / "msg", "eu1"),
    "North Indian Ocean": ("INSAT-3D", DATA_DIR / "mosdac", "in1"),
}

# ERA5 bounding boxes by region
ERA5_BBOXES = {
    "North Indian Ocean": ([90, 0.78, -81.22, 163.22, 81.22], "NIO"),
    "Australia": ([90, 0, -90, 180, 90], "AUS"),
    "West Indian Ocean": ([90, -35.7, -81.26, 126.7, 81.26], "SWIO"),
    "North Atlantic Ocean": ([90, -156.2, -81.15, 6.2, 81.15], "USE"),
    "North Pacific Ocean": ([90, -180, -81.15, 180, 81.15], "USW"),
    "West Pacific Ocean": ([90, 0, -90, 180, 90], "PHI"),
}

ERA5_VARIABLES = [
    '100m_u_component_of_wind', '100m_v_component_of_wind', '10m_u_component_of_neutral_wind',
    '10m_u_component_of_wind', '10m_v_component_of_neutral_wind', '10m_v_component_of_wind',
    '10m_wind_gust_since_previous_post_processing', 'cloud_base_height', 'convective_precipitation',
    'convective_rain_rate', 'high_cloud_cover', 'instantaneous_10m_wind_gust',
    'instantaneous_large_scale_surface_precipitation_fraction', 'large_scale_precipitation', 
    'large_scale_precipitation_fraction', 'large_scale_rain_rate', 'low_cloud_cover', 
    'maximum_total_precipitation_rate_since_previous_post_processing', 'medium_cloud_cover', 
    'minimum_total_precipitation_rate_since_previous_post_processing', 'precipitation_type',
    'total_cloud_cover', 'total_column_cloud_ice_water', 'total_column_cloud_liquid_water',
    'total_column_rain_water', 'total_precipitation', 
    'vertical_integral_of_divergence_of_cloud_frozen_water_flux',
    'vertical_integral_of_divergence_of_cloud_liquid_water_flux', 
    'vertical_integral_of_eastward_cloud_frozen_water_flux', 
    'vertical_integral_of_eastward_cloud_liquid_water_flux',
    'vertical_integral_of_northward_cloud_frozen_water_flux', 
    'vertical_integral_of_northward_cloud_liquid_water_flux',
]

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_cyclone_info(cyclone_names):
    """Get cyclone information from the Excel file."""
    df = pd.read_excel(CYCLONES_PATH)
    df = df.drop('Unnamed: 8', axis=1, errors='ignore')
    df = df.dropna()
    
    # Filter for requested cyclones
    mask = df["Name"].str.lower().isin([n.lower() for n in cyclone_names])
    selected = df[mask]
    
    if len(selected) == 0:
        print(f"❌ No cyclones found matching: {cyclone_names}")
        print(f"   Available cyclones: {df['Name'].tolist()}")
        return None
    
    return selected

def get_dayno(dt):
    """Get day number of year."""
    return (dt - datetime.datetime(dt.year, 1, 1)).days + 1

def estimate_download_size(cyclone_row):
    """Estimate download size for a cyclone in GB."""
    start = datetime.datetime.strptime(cyclone_row["Form Date"], "%d-%m-%Y")
    end = datetime.datetime.strptime(cyclone_row["Dissipated Date"], "%d-%m-%Y")
    hours = (end - start).total_seconds() / 3600
    
    # Estimate: ~50MB per hour of satellite data + ~100MB per day ERA5
    satellite_gb = (hours * 50) / 1024
    era5_gb = ((end - start).days + 1) * 0.1
    
    return satellite_gb + era5_gb

# =============================================================================
# GOES-16 EAST DOWNLOADER (For Ian, Fiona, etc.)
# =============================================================================

def download_goes_east_hour(date, name, base_dir):
    """Download one hour of GOES-16 East data."""
    year = date.year
    month = date.month
    day = date.day
    hour = date.hour
    day_no = get_dayno(date)
    
    # AWS S3 path
    s3_path = f"s3://noaa-goes16/ABI-L1b-RadF/{year}/{day_no:03}/{hour:02}/"
    
    # List available files
    cmd = f"aws s3 ls --no-sign-request {s3_path}OR_ABI-L1b-RadF-M6C13_G16"
    result = subprocess.run(cmd.split(), capture_output=True, text=True)
    
    if result.returncode != 0 or not result.stdout.strip():
        return False
    
    # Get first file (Band 13 - IR)
    filenames = [x.split()[-1] for x in result.stdout.strip().split('\n') if x]
    if not filenames:
        return False
    
    stub = filenames[0]
    
    # Destination folder
    dest_folder = base_dir / "data" / "nc" / name.lower() / f"{year}-{month:02}-{day:02}"
    dest_folder.mkdir(parents=True, exist_ok=True)
    
    # Download
    cmd = f"aws s3 cp --no-sign-request {s3_path}{stub} {dest_folder}/"
    subprocess.run(cmd.split(), capture_output=True)
    
    return True

def download_goes_east_cyclone(name, start_date, end_date, progress_callback=None):
    """Download all GOES-16 East data for a cyclone."""
    print(f"\n🛰️  Downloading GOES-16 East data for {name}...")
    
    base_dir = GOES_EAST_DIR
    
    # Generate list of hours
    current = start_date
    dates = []
    while current <= end_date:
        dates.append(current)
        current += datetime.timedelta(hours=1)
    
    # Download sequentially with progress bar
    success_count = 0
    for date in tqdm(dates, desc=f"[{name}] Satellite", unit="hr"):
        if download_goes_east_hour(date, name, base_dir):
            success_count += 1
    
    print(f"   ✅ Downloaded {success_count}/{len(dates)} hours")
    return success_count

# =============================================================================
# ERA5 DOWNLOADER
# =============================================================================

def download_era5_day(date, name, nbox, abbv):
    """Download ERA5 data for one day."""
    try:
        import cdsapi
        c = cdsapi.Client()
        
        dest_folder = ERA5_DIR / "data" / "nc" / name.replace(' ', '').lower()
        dest_folder.mkdir(parents=True, exist_ok=True)
        
        stub = f'ERA5_{abbv.upper()}_{date.year}{date.month:02}{date.day:02}.nc'
        local_path = dest_folder / stub
        
        if local_path.exists():
            return True
        
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': ERA5_VARIABLES,
                'year': f'{date.year}',
                'month': f'{date.month:02}',
                'day': [f'{date.day:02}'],
                'time': [f'{h:02}:00' for h in range(24)],
                'area': nbox,
            },
            str(local_path)
        )
        return True
        
    except Exception as e:
        print(f"   ⚠️  ERA5 error for {date}: {e}")
        return False

def download_era5_cyclone(name, region, start_date, end_date):
    """Download ERA5 data for a cyclone."""
    print(f"\n🌍 Downloading ERA5 reanalysis data for {name}...")
    
    if region not in ERA5_BBOXES:
        print(f"   ⚠️  Unknown region: {region}")
        return 0
    
    nbox, abbv = ERA5_BBOXES[region]
    
    # Generate list of days
    current = start_date
    dates = []
    while current <= end_date:
        dates.append(current)
        current += datetime.timedelta(days=1)
    
    # Download sequentially with progress
    success_count = 0
    for date in tqdm(dates, desc=f"[{name}] ERA5", unit="day"):
        if download_era5_day(date, name, nbox, abbv):
            success_count += 1
    
    print(f"   ✅ Downloaded {success_count}/{len(dates)} days")
    return success_count

# =============================================================================
# MAIN DOWNLOAD FUNCTION
# =============================================================================

def download_cyclones(cyclone_names, skip_era5=False, skip_satellite=False):
    """
    Download data for specified cyclones.
    
    Args:
        cyclone_names: List of cyclone names, e.g., ["Ian", "Fiona"]
        skip_era5: Skip ERA5 download (if you only need satellite data)
        skip_satellite: Skip satellite download (if you only need ERA5)
    
    Returns:
        dict: Summary of downloaded data
    
    Example:
        download_cyclones(["Ian", "Fiona"])
    """
    print("\n" + "=" * 60)
    print("🌀 CYCLONE DATA DOWNLOADER")
    print("=" * 60)
    
    # Get cyclone info
    cyclones = get_cyclone_info(cyclone_names)
    if cyclones is None:
        return None
    
    print(f"\n📋 Found {len(cyclones)} cyclones:")
    total_size = 0
    for _, row in cyclones.iterrows():
        size = estimate_download_size(row)
        total_size += size
        satellite, _, _ = REGION_SATELLITE_MAP.get(row["Region"], ("Unknown", None, None))
        print(f"   • {row['Name']} ({row['Region']}) - ~{size:.1f}GB - {satellite}")
    
    print(f"\n   Total estimated size: ~{total_size:.1f}GB")
    
    if total_size > 18:
        print("   ⚠️  Warning: This may exceed Kaggle's 20GB limit!")
    
    # Download each cyclone
    results = {}
    for _, row in cyclones.iterrows():
        name = row["Name"]
        region = row["Region"]
        start_date = datetime.datetime.strptime(row["Form Date"], "%d-%m-%Y")
        end_date = datetime.datetime.strptime(row["Dissipated Date"], "%d-%m-%Y")
        
        print(f"\n{'='*60}")
        print(f"🌀 Processing: {name}")
        print(f"   Region: {region}")
        print(f"   Period: {start_date.date()} to {end_date.date()}")
        print("=" * 60)
        
        satellite_hours = 0
        era5_days = 0
        
        # Download satellite data
        if not skip_satellite:
            satellite_type, base_dir, script_id = REGION_SATELLITE_MAP.get(region, (None, None, None))
            
            if script_id == "am1":  # GOES-16 East
                satellite_hours = download_goes_east_cyclone(name, start_date, end_date)
            else:
                print(f"   ⚠️  Satellite downloader for {satellite_type} not yet implemented")
                print(f"      Run manually: python {REPO_DIR}/dataproc/{script_id}-*.py")
        
        # Download ERA5 data
        if not skip_era5:
            era5_days = download_era5_cyclone(name, region, start_date, end_date)
        
        results[name] = {
            "satellite_hours": satellite_hours,
            "era5_days": era5_days,
            "region": region,
        }
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 DOWNLOAD SUMMARY")
    print("=" * 60)
    
    for name, data in results.items():
        print(f"\n{name}:")
        print(f"   Satellite data: {data['satellite_hours']} hours")
        print(f"   ERA5 data: {data['era5_days']} days")
    
    print(f"\n📁 Data saved to: {DATA_DIR}")
    print("\nNext steps:")
    print("   1. Generate metadata: python generate_metadata.py")
    print("   2. Create dataloaders: python create_dataloaders.py")
    
    return results


def list_available_cyclones():
    """List all available cyclones from the Excel file."""
    df = pd.read_excel(CYCLONES_PATH)
    df = df.drop('Unnamed: 8', axis=1, errors='ignore')
    df = df.dropna()
    
    print("\n📋 Available Cyclones:")
    print("=" * 60)
    
    for region in df["Region"].unique():
        print(f"\n{region}:")
        region_df = df[df["Region"] == region]
        satellite, _, _ = REGION_SATELLITE_MAP.get(region, ("Unknown", None, None))
        print(f"   Satellite: {satellite}")
        for _, row in region_df.iterrows():
            print(f"   • {row['Name']} ({row['Form Date']} - {row['Dissipated Date']})")


# =============================================================================
# QUICK ACCESS FUNCTIONS
# =============================================================================

def download_ian_and_fiona():
    """Quick function to download Ian and Fiona (North Atlantic cyclones)."""
    return download_cyclones(["Ian", "Fiona"])


if __name__ == "__main__":
    # If run directly, list available cyclones
    list_available_cyclones()
