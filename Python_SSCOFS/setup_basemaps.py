#!/usr/bin/env python3
"""
setup_basemaps.py
-----------------

Download and setup basemap data for plot_currents_enhanced.py

This script will:
1. Check for required dependencies (contextily, geopandas)
2. Download Natural Earth coastline data
3. Clip to Puget Sound region for faster rendering
4. Verify everything is working

Usage:
    python setup_basemaps.py --all          # Install everything
    python setup_basemaps.py --contextily   # Just contextily
    python setup_basemaps.py --natural-earth # Just Natural Earth data
"""

import argparse
import sys
from pathlib import Path
import subprocess

def check_package(package_name, pip_name=None, conda_name=None):
    """Check if a package is installed."""
    if pip_name is None:
        pip_name = package_name
    if conda_name is None:
        conda_name = package_name
    
    try:
        __import__(package_name)
        print(f"✓ {package_name} is already installed")
        return True
    except ImportError:
        print(f"✗ {package_name} is not installed")
        return False

def install_contextily():
    """Install contextily via pip."""
    print("\n" + "="*60)
    print("INSTALLING CONTEXTILY")
    print("="*60)
    
    if check_package('contextily'):
        return True
    
    print("\nContextily provides web tile basemaps.")
    print("Installing with pip...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "contextily"])
        print("✓ Contextily installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install contextily: {e}")
        print("  Try manually: pip install contextily")
        return False

def install_geopandas():
    """Install geopandas (prefer conda)."""
    print("\n" + "="*60)
    print("INSTALLING GEOPANDAS")
    print("="*60)
    
    if check_package('geopandas'):
        return True
    
    print("\nGeopandas is needed for Natural Earth shapefiles.")
    print("Recommended: Install with conda (not pip)")
    print()
    print("To install geopandas, run this command in your terminal:")
    print("  conda install -c conda-forge geopandas")
    print()
    
    response = input("Would you like to try installing with conda now? (y/n): ").lower().strip()
    
    if response == 'y':
        try:
            subprocess.check_call(["conda", "install", "-y", "-c", "conda-forge", "geopandas"])
            print("✓ Geopandas installed successfully!")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"✗ Failed to install geopandas: {e}")
            print("  Please install manually: conda install -c conda-forge geopandas")
            return False
    else:
        print("Skipping geopandas installation.")
        print("You can install it later with: conda install -c conda-forge geopandas")
        return False

def download_natural_earth(data_dir, region='puget_sound'):
    """Download Natural Earth coastline data and optionally clip to region."""
    print("\n" + "="*60)
    print("DOWNLOADING NATURAL EARTH DATA")
    print("="*60)
    
    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True, parents=True)
    
    # Check if we have geopandas
    try:
        import geopandas as gpd
        import requests
        from zipfile import ZipFile
        from io import BytesIO
    except ImportError as e:
        print(f"✗ Missing required package: {e}")
        print("  Need: geopandas, requests")
        print("  Install with: conda install -c conda-forge geopandas")
        return False
    
    # Natural Earth 10m coastline URL
    url = "https://github.com/nvkelso/natural-earth-vector/raw/master/10m_physical/ne_10m_coastline.zip"
    
    coastline_shp = data_dir / "ne_10m_coastline.shp"
    
    if coastline_shp.exists():
        print(f"✓ Natural Earth coastline already exists: {coastline_shp}")
        response = input("Download again? (y/n): ").lower().strip()
        if response != 'y':
            print("Skipping download.")
            return True
    
    print(f"\nDownloading from: {url}")
    print("This may take 30-60 seconds (file is ~5 MB)...")
    
    try:
        # Download
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        
        print("✓ Download complete, extracting...")
        
        # Extract
        with ZipFile(BytesIO(response.content)) as zip_file:
            zip_file.extractall(data_dir)
        
        print(f"✓ Extracted to: {data_dir}")
        
        # List what was extracted
        extracted_files = list(data_dir.glob("ne_10m_coastline.*"))
        print(f"  Files: {', '.join(f.name for f in extracted_files)}")
        
    except Exception as e:
        print(f"✗ Failed to download/extract: {e}")
        print("\nYou can manually download from:")
        print("  https://www.naturalearthdata.com/downloads/10m-physical-vectors/10m-coastline/")
        print(f"  Extract to: {data_dir.absolute()}")
        return False
    
    # Optionally clip to Puget Sound region for faster rendering
    if region == 'puget_sound':
        print("\n" + "-"*60)
        print("CLIPPING TO PUGET SOUND REGION")
        print("-"*60)
        
        try:
            print("Loading full coastline...")
            coastline = gpd.read_file(coastline_shp)
            
            # Puget Sound bounding box (approximate, in lat/lon)
            # Covers from Seattle to Bellingham, including San Juan Islands
            minx, miny, maxx, maxy = -123.5, 47.0, -122.0, 49.0
            
            print(f"Clipping to region: {minx:.1f}W to {maxx:.1f}W, {miny:.1f}N to {maxy:.1f}N")
            
            # Filter to region
            coastline_clipped = coastline.cx[minx:maxx, miny:maxy]
            
            print(f"  Original features: {len(coastline)}")
            print(f"  Clipped features: {len(coastline_clipped)}")
            
            # Save clipped version
            output_file = data_dir / "shoreline_puget.geojson"
            coastline_clipped.to_file(output_file, driver='GeoJSON')
            
            print(f"✓ Saved Puget Sound coastline: {output_file}")
            print(f"  File size: {output_file.stat().st_size / 1024:.1f} KB")
            
            # Calculate size reduction
            original_size = sum(f.stat().st_size for f in data_dir.glob("ne_10m_coastline.*"))
            new_size = output_file.stat().st_size
            savings = (1 - new_size / original_size) * 100
            
            print(f"  Size reduction: {savings:.0f}% (faster loading!)")
            
        except Exception as e:
            print(f"✗ Failed to clip region: {e}")
            print("  Full coastline data is still available in ne_10m_coastline.shp")
    
    return True

def verify_setup(data_dir):
    """Verify that basemap setup is working."""
    print("\n" + "="*60)
    print("VERIFICATION")
    print("="*60)
    
    data_dir = Path(data_dir)
    
    # Check contextily
    print("\n1. Contextily (web tiles):")
    has_contextily = check_package('contextily')
    if has_contextily:
        print("   → Can use: --basemap contextily")
    else:
        print("   → Install with: pip install contextily")
    
    # Check geopandas
    print("\n2. Geopandas (shapefile support):")
    has_geopandas = check_package('geopandas')
    
    # Check for data files
    print("\n3. Coastline data files:")
    data_files = [
        data_dir / "shoreline_puget.geojson",
        data_dir / "ne_10m_coastline.shp",
        data_dir / "coastline.geojson"
    ]
    
    found_files = [f for f in data_files if f.exists()]
    
    if found_files:
        for f in found_files:
            size_kb = f.stat().st_size / 1024
            print(f"   ✓ {f.name} ({size_kb:.1f} KB)")
        if has_geopandas:
            print("   → Can use: --basemap natural_earth")
        else:
            print("   → Need geopandas to use these files")
    else:
        print(f"   ✗ No coastline files found in {data_dir}")
        print("   → Run: python setup_basemaps.py --natural-earth")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    ready_options = []
    if has_contextily:
        ready_options.append("contextily")
    if has_geopandas and found_files:
        ready_options.append("natural_earth")
    
    if ready_options:
        print(f"✓ Ready to use basemaps: {', '.join(ready_options)}")
        print("\nTest with:")
        for option in ready_options:
            print(f"  python test_basemaps.py --{option}")
    else:
        print("✗ No basemap options are ready yet")
        print("  Run: python setup_basemaps.py --all")
    
    print()

def main():
    parser = argparse.ArgumentParser(
        description="Setup basemap data for SSCOFS current visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python setup_basemaps.py --all              # Setup everything
  python setup_basemaps.py --contextily       # Just web tiles
  python setup_basemaps.py --natural-earth    # Just coastline data
  python setup_basemaps.py --verify           # Check what's installed
        """
    )
    
    parser.add_argument(
        '--all', action='store_true',
        help='Install/download everything'
    )
    parser.add_argument(
        '--contextily', action='store_true',
        help='Install contextily (web tile basemaps)'
    )
    parser.add_argument(
        '--natural-earth', action='store_true',
        help='Download Natural Earth coastline data'
    )
    parser.add_argument(
        '--data-dir', type=str, default='data',
        help='Directory for data files (default: data/)'
    )
    parser.add_argument(
        '--verify', action='store_true',
        help='Verify what basemaps are ready to use'
    )
    parser.add_argument(
        '--no-clip', action='store_true',
        help='Don\'t clip Natural Earth data to Puget Sound (keep full dataset)'
    )
    
    args = parser.parse_args()
    
    # If no flags, show help and verify
    if not any([args.all, args.contextily, args.natural_earth, args.verify]):
        parser.print_help()
        print()
        verify_setup(args.data_dir)
        return 0
    
    # Verify only
    if args.verify:
        verify_setup(args.data_dir)
        return 0
    
    print("="*60)
    print("SSCOFS BASEMAP SETUP")
    print("="*60)
    print(f"Data directory: {Path(args.data_dir).absolute()}")
    print()
    
    success_count = 0
    total_count = 0
    
    # Install contextily
    if args.all or args.contextily:
        total_count += 1
        if install_contextily():
            success_count += 1
    
    # Install geopandas (if doing Natural Earth)
    if args.all or args.natural_earth:
        total_count += 1
        has_geopandas = check_package('geopandas')
        if not has_geopandas:
            if install_geopandas():
                success_count += 1
                has_geopandas = True
        else:
            success_count += 1
            has_geopandas = True
    else:
        has_geopandas = check_package('geopandas')
    
    # Download Natural Earth data
    if (args.all or args.natural_earth) and has_geopandas:
        total_count += 1
        region = '' if args.no_clip else 'puget_sound'
        if download_natural_earth(args.data_dir, region=region):
            success_count += 1
    
    # Final verification
    verify_setup(args.data_dir)
    
    # Exit status
    if success_count == total_count:
        print("✓ Setup completed successfully!")
        return 0
    else:
        print(f"⚠ Completed with warnings ({success_count}/{total_count} successful)")
        return 1

if __name__ == '__main__':
    exit(main())

