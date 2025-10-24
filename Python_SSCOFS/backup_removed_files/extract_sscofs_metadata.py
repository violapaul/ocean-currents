"""
extract_sscofs_metadata.py
---------------------------

Extract and display comprehensive metadata from SSCOFS NetCDF files.
This script loads SSCOFS data and extracts detailed information about:
- Global attributes
- Variables (names, types, dimensions, shapes, attributes)
- Coordinates and grid structure
- Temporal information
- Spatial extent
- Data statistics

Usage:
    # Extract metadata from the latest available data
    python extract_sscofs_metadata.py
    
    # Extract metadata from a specific date/cycle/forecast
    python extract_sscofs_metadata.py --date 2025-10-17 --cycle 9 --forecast 12
    
    # Save metadata to a text file
    python extract_sscofs_metadata.py --output metadata.txt
    
    # Save metadata as JSON
    python extract_sscofs_metadata.py --json metadata.json
    
    # Show detailed variable statistics
    python extract_sscofs_metadata.py --detailed-stats
"""

import argparse
import datetime as dt
from datetime import timezone, timedelta
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import xarray as xr

# Import helper functions from existing modules
from latest_cycle import latest_cycle_and_url_for_local_hour
from fetch_sscofs import build_sscofs_url
from sscofs_cache import load_sscofs_data


def extract_global_attributes(ds: xr.Dataset) -> Dict[str, Any]:
    """
    Extract global attributes from the dataset.
    
    Parameters:
    -----------
    ds : xr.Dataset
        SSCOFS dataset
        
    Returns:
    --------
    dict : Dictionary of global attributes
    """
    return dict(ds.attrs)


def extract_dimensions_info(ds: xr.Dataset) -> Dict[str, int]:
    """
    Extract dimension names and sizes.
    
    Parameters:
    -----------
    ds : xr.Dataset
        SSCOFS dataset
        
    Returns:
    --------
    dict : Dictionary mapping dimension names to sizes
    """
    return dict(ds.dims)


def extract_coordinates_info(ds: xr.Dataset) -> Dict[str, Dict[str, Any]]:
    """
    Extract information about coordinate variables.
    
    Parameters:
    -----------
    ds : xr.Dataset
        SSCOFS dataset
        
    Returns:
    --------
    dict : Dictionary with coordinate information
    """
    coords_info = {}
    
    for coord_name in ds.coords:
        coord = ds.coords[coord_name]
        
        # Clean attributes - truncate arrays in attributes
        clean_attrs = {}
        for key, value in coord.attrs.items():
            if isinstance(value, (list, np.ndarray)):
                if len(value) > 10:
                    clean_attrs[key] = f"[array of {len(value)} values, showing first 5: {list(value[:5])}...]"
                else:
                    clean_attrs[key] = list(value) if isinstance(value, np.ndarray) else value
            elif isinstance(value, str) and len(value) > 200:
                clean_attrs[key] = value[:200] + "..."
            else:
                clean_attrs[key] = value
        
        info = {
            'dtype': str(coord.dtype),
            'shape': coord.shape,
            'dimensions': list(coord.dims),
            'attributes': clean_attrs,
        }
        
        # Add basic statistics and sample values for numeric coordinates
        if np.issubdtype(coord.dtype, np.number):
            coord_values = coord.values
            info['min'] = float(coord.min().values)
            info['max'] = float(coord.max().values)
            info['mean'] = float(coord.mean().values)
            
            # For large coordinate arrays, only store a sample
            if coord_values.size > 20:
                if len(coord_values.shape) == 1:
                    # 1D array
                    info['value_sample'] = coord_values[:5].tolist() + ['...'] + coord_values[-5:].tolist()
                elif len(coord_values.shape) == 2:
                    # 2D array - just show dimensions, not values
                    info['note'] = f"2D array ({coord_values.shape[0]} x {coord_values.shape[1]}), values not shown"
                else:
                    info['note'] = f"Multi-dimensional array, shape {coord_values.shape}, values not shown"
            else:
                # Small arrays - show all values
                info['values'] = coord_values.tolist()
            
        coords_info[coord_name] = info
    
    return coords_info


def extract_variables_info(ds: xr.Dataset, detailed_stats: bool = False) -> Dict[str, Dict[str, Any]]:
    """
    Extract information about data variables.
    
    Parameters:
    -----------
    ds : xr.Dataset
        SSCOFS dataset
    detailed_stats : bool
        If True, compute detailed statistics for each variable
        
    Returns:
    --------
    dict : Dictionary with variable information
    """
    vars_info = {}
    
    for var_name in ds.data_vars:
        var = ds[var_name]
        
        # Clean attributes - truncate arrays in attributes
        clean_attrs = {}
        for key, value in var.attrs.items():
            if isinstance(value, (list, np.ndarray)):
                if len(value) > 10:
                    clean_attrs[key] = f"[array of {len(value)} values, showing first 5: {list(value[:5])}...]"
                else:
                    clean_attrs[key] = list(value) if isinstance(value, np.ndarray) else value
            elif isinstance(value, str) and len(value) > 200:
                clean_attrs[key] = value[:200] + "..."
            else:
                clean_attrs[key] = value
        
        info = {
            'dtype': str(var.dtype),
            'shape': var.shape,
            'dimensions': list(var.dims),
            'attributes': clean_attrs,
            'size_mb': var.nbytes / (1024 * 1024),
        }
        
        # Add basic statistics for numeric variables
        if np.issubdtype(var.dtype, np.number):
            try:
                # Use a sample for large arrays to avoid memory issues
                if var.size > 10_000_000:
                    # Sample 1% of the data
                    sample = var.values.flatten()[::100]
                    info['stats_note'] = 'Statistics computed from 1% sample due to large size'
                else:
                    sample = var.values.flatten()
                    info['stats_note'] = 'Statistics computed from full dataset'
                
                # Remove NaN values for statistics
                sample_valid = sample[~np.isnan(sample)]
                
                if len(sample_valid) > 0:
                    info['min'] = float(np.min(sample_valid))
                    info['max'] = float(np.max(sample_valid))
                    info['mean'] = float(np.mean(sample_valid))
                    info['std'] = float(np.std(sample_valid))
                    
                    if detailed_stats:
                        info['median'] = float(np.median(sample_valid))
                        info['percentile_25'] = float(np.percentile(sample_valid, 25))
                        info['percentile_75'] = float(np.percentile(sample_valid, 75))
                        info['percentile_95'] = float(np.percentile(sample_valid, 95))
                        info['percentile_99'] = float(np.percentile(sample_valid, 99))
                    
                    # Count NaN values
                    total_size = sample.size
                    nan_count = np.sum(np.isnan(sample))
                    info['nan_count'] = int(nan_count)
                    info['nan_percentage'] = float(nan_count / total_size * 100)
                else:
                    info['note'] = 'All values are NaN'
                    
            except Exception as e:
                info['stats_error'] = str(e)
        
        vars_info[var_name] = info
    
    return vars_info


def extract_temporal_info(ds: xr.Dataset) -> Dict[str, Any]:
    """
    Extract temporal information from the dataset.
    
    Parameters:
    -----------
    ds : xr.Dataset
        SSCOFS dataset
        
    Returns:
    --------
    dict : Dictionary with temporal information
    """
    temporal_info = {}
    
    if 'time' in ds.coords:
        time_coord = ds['time']
        temporal_info['num_timesteps'] = len(time_coord)
        
        # Convert to datetime strings
        times = [str(t.values) for t in time_coord]
        temporal_info['first_time'] = times[0]
        temporal_info['last_time'] = times[-1]
        
        # Calculate time step if more than one time
        if len(time_coord) > 1:
            dt_seconds = (time_coord[1] - time_coord[0]).values / np.timedelta64(1, 's')
            temporal_info['time_step_seconds'] = float(dt_seconds)
            temporal_info['time_step_hours'] = float(dt_seconds / 3600)
        
        temporal_info['all_times'] = times
    
    return temporal_info


def extract_spatial_info(ds: xr.Dataset) -> Dict[str, Any]:
    """
    Extract spatial extent and grid information.
    
    Parameters:
    -----------
    ds : xr.Dataset
        SSCOFS dataset
        
    Returns:
    --------
    dict : Dictionary with spatial information
    """
    spatial_info = {}
    
    # Get node coordinates if available
    if 'lon' in ds.coords and 'lat' in ds.coords:
        lons = ds['lon'].values
        lats = ds['lat'].values
        
        # Handle longitude convention (0-360 vs -180 to 180)
        if lons.max() > 180:
            lons_converted = np.where(lons > 180, lons - 360, lons)
        else:
            lons_converted = lons
        
        spatial_info['nodes'] = {
            'count': len(lons),
            'lon_range': [float(np.min(lons_converted)), float(np.max(lons_converted))],
            'lat_range': [float(np.min(lats)), float(np.max(lats))],
            'lon_mean': float(np.mean(lons_converted)),
            'lat_mean': float(np.mean(lats)),
        }
    
    # Get element coordinates if available
    if 'lonc' in ds.coords and 'latc' in ds.coords:
        lonsc = ds['lonc'].values
        latsc = ds['latc'].values
        
        # Handle longitude convention
        if lonsc.max() > 180:
            lonsc_converted = np.where(lonsc > 180, lonsc - 360, lonsc)
        else:
            lonsc_converted = lonsc
        
        spatial_info['elements'] = {
            'count': len(lonsc),
            'lon_range': [float(np.min(lonsc_converted)), float(np.max(lonsc_converted))],
            'lat_range': [float(np.min(latsc)), float(np.max(latsc))],
            'lon_mean': float(np.mean(lonsc_converted)),
            'lat_mean': float(np.mean(latsc)),
        }
    
    # Get vertical layer information
    if 'siglay' in ds.coords:
        sigma_values = ds['siglay'].values
        # Only include sigma values if there aren't too many
        if len(sigma_values) <= 20:
            spatial_info['vertical_layers'] = {
                'count': len(sigma_values),
                'sigma_values': sigma_values.tolist(),
            }
        else:
            spatial_info['vertical_layers'] = {
                'count': len(sigma_values),
                'sigma_range': [float(sigma_values.min()), float(sigma_values.max())],
                'sigma_sample': sigma_values[:5].tolist() + ['...'] + sigma_values[-5:].tolist(),
            }
    elif 'sigma_layer' in ds.dims:
        spatial_info['vertical_layers'] = {
            'count': ds.dims['sigma_layer'],
        }
    
    return spatial_info


def extract_grid_connectivity(ds: xr.Dataset) -> Dict[str, Any]:
    """
    Extract information about the grid connectivity (if available).
    
    Parameters:
    -----------
    ds : xr.Dataset
        SSCOFS dataset
        
    Returns:
    --------
    dict : Dictionary with grid connectivity information
    """
    connectivity_info = {}
    
    # Check for triangular mesh connectivity
    if 'nv' in ds.variables or 'nv' in ds.coords:
        nv = ds['nv'] if 'nv' in ds.variables else ds.coords['nv']
        connectivity_info['mesh_type'] = 'unstructured_triangular'
        connectivity_info['vertices_per_element'] = nv.shape[0] if len(nv.shape) > 1 else 3
        connectivity_info['num_elements'] = nv.shape[1] if len(nv.shape) > 1 else nv.shape[0]
    
    # Check for neighbor information
    if 'nbe' in ds.variables or 'nbe' in ds.coords:
        connectivity_info['has_neighbor_info'] = True
    
    return connectivity_info


def format_metadata_text(metadata: Dict[str, Any], detailed: bool = False) -> str:
    """
    Format metadata as readable text.
    
    Parameters:
    -----------
    metadata : dict
        Metadata dictionary
    detailed : bool
        If True, include more detailed information
        
    Returns:
    --------
    str : Formatted text
    """
    lines = []
    lines.append("=" * 80)
    lines.append("SSCOFS NETCDF METADATA REPORT")
    lines.append("=" * 80)
    lines.append("")
    
    # File information
    if 'file_info' in metadata:
        lines.append("FILE INFORMATION")
        lines.append("-" * 80)
        for key, value in metadata['file_info'].items():
            lines.append(f"  {key}: {value}")
        lines.append("")
    
    # Global attributes
    if 'global_attributes' in metadata:
        lines.append("GLOBAL ATTRIBUTES")
        lines.append("-" * 80)
        for key, value in metadata['global_attributes'].items():
            # Truncate very long values and arrays
            if isinstance(value, (list, np.ndarray)):
                if len(value) > 10:
                    value_str = f"[array of {len(value)} values]"
                    if detailed:
                        value_str += f" First 5: {list(value[:5])}..."
                else:
                    value_str = str(value)
            else:
                value_str = str(value)
                max_len = 500 if detailed else 200
                if len(value_str) > max_len:
                    value_str = value_str[:max_len] + "..."
            lines.append(f"  {key}: {value_str}")
        lines.append("")
    
    # Dimensions
    if 'dimensions' in metadata:
        lines.append("DIMENSIONS")
        lines.append("-" * 80)
        for dim_name, dim_size in metadata['dimensions'].items():
            lines.append(f"  {dim_name}: {dim_size:,}")
        lines.append("")
    
    # Spatial information
    if 'spatial_info' in metadata:
        lines.append("SPATIAL INFORMATION")
        lines.append("-" * 80)
        spatial = metadata['spatial_info']
        
        if 'nodes' in spatial:
            lines.append(f"  Nodes: {spatial['nodes']['count']:,}")
            lines.append(f"    Longitude range: {spatial['nodes']['lon_range'][0]:.4f} to {spatial['nodes']['lon_range'][1]:.4f}")
            lines.append(f"    Latitude range: {spatial['nodes']['lat_range'][0]:.4f} to {spatial['nodes']['lat_range'][1]:.4f}")
            lines.append(f"    Center: ({spatial['nodes']['lat_mean']:.4f}, {spatial['nodes']['lon_mean']:.4f})")
        
        if 'elements' in spatial:
            lines.append(f"  Elements: {spatial['elements']['count']:,}")
            lines.append(f"    Longitude range: {spatial['elements']['lon_range'][0]:.4f} to {spatial['elements']['lon_range'][1]:.4f}")
            lines.append(f"    Latitude range: {spatial['elements']['lat_range'][0]:.4f} to {spatial['elements']['lat_range'][1]:.4f}")
        
        if 'vertical_layers' in spatial:
            lines.append(f"  Vertical layers: {spatial['vertical_layers']['count']}")
            if 'sigma_range' in spatial['vertical_layers']:
                lines.append(f"    Sigma range: {spatial['vertical_layers']['sigma_range'][0]:.6f} to {spatial['vertical_layers']['sigma_range'][1]:.6f}")
                if detailed and 'sigma_sample' in spatial['vertical_layers']:
                    sigma_sample = spatial['vertical_layers']['sigma_sample']
                    lines.append(f"    Sigma sample (first 5 and last 5): {sigma_sample}")
            elif detailed and 'sigma_values' in spatial['vertical_layers']:
                lines.append(f"    Sigma values: {spatial['vertical_layers']['sigma_values']}")
        
        lines.append("")
    
    # Temporal information
    if 'temporal_info' in metadata:
        lines.append("TEMPORAL INFORMATION")
        lines.append("-" * 80)
        temporal = metadata['temporal_info']
        lines.append(f"  Number of time steps: {temporal.get('num_timesteps', 'N/A')}")
        lines.append(f"  First time: {temporal.get('first_time', 'N/A')}")
        lines.append(f"  Last time: {temporal.get('last_time', 'N/A')}")
        if 'time_step_hours' in temporal:
            lines.append(f"  Time step: {temporal['time_step_hours']:.2f} hours")
        
        if detailed and 'all_times' in temporal:
            all_times = temporal['all_times']
            if len(all_times) <= 10:
                lines.append("  All times:")
                for i, t in enumerate(all_times):
                    lines.append(f"    [{i}] {t}")
            else:
                lines.append(f"  All times (showing first 5 and last 5 of {len(all_times)}):")
                for i in range(5):
                    lines.append(f"    [{i}] {all_times[i]}")
                lines.append(f"    ...")
                for i in range(len(all_times) - 5, len(all_times)):
                    lines.append(f"    [{i}] {all_times[i]}")
        
        lines.append("")
    
    # Grid connectivity
    if 'grid_connectivity' in metadata and metadata['grid_connectivity']:
        lines.append("GRID CONNECTIVITY")
        lines.append("-" * 80)
        for key, value in metadata['grid_connectivity'].items():
            lines.append(f"  {key}: {value}")
        lines.append("")
    
    # Coordinates
    if 'coordinates' in metadata:
        lines.append("COORDINATES")
        lines.append("-" * 80)
        for coord_name, coord_info in metadata['coordinates'].items():
            lines.append(f"  {coord_name}:")
            lines.append(f"    Type: {coord_info['dtype']}")
            lines.append(f"    Shape: {coord_info['shape']}")
            lines.append(f"    Dimensions: {coord_info['dimensions']}")
            
            if 'min' in coord_info:
                lines.append(f"    Range: {coord_info['min']:.6g} to {coord_info['max']:.6g}")
                lines.append(f"    Mean: {coord_info['mean']:.6g}")
                
                # Show values or value sample
                if 'values' in coord_info:
                    lines.append(f"    Values: {coord_info['values']}")
                elif 'value_sample' in coord_info:
                    lines.append(f"    Value sample (first 5 and last 5): {coord_info['value_sample']}")
                    
                if 'note' in coord_info:
                    lines.append(f"    Note: {coord_info['note']}")
            
            if detailed and coord_info['attributes']:
                lines.append(f"    Attributes:")
                for attr_key, attr_val in coord_info['attributes'].items():
                    lines.append(f"      {attr_key}: {attr_val}")
        lines.append("")
    
    # Variables
    if 'variables' in metadata:
        lines.append("DATA VARIABLES")
        lines.append("-" * 80)
        
        # Calculate total size
        total_size_mb = sum(v['size_mb'] for v in metadata['variables'].values())
        lines.append(f"Total data size: {total_size_mb:.2f} MB")
        lines.append("")
        
        for var_name, var_info in metadata['variables'].items():
            lines.append(f"  {var_name}:")
            lines.append(f"    Type: {var_info['dtype']}")
            lines.append(f"    Shape: {var_info['shape']}")
            lines.append(f"    Dimensions: {var_info['dimensions']}")
            lines.append(f"    Size: {var_info['size_mb']:.2f} MB")
            
            if 'min' in var_info:
                lines.append(f"    Range: {var_info['min']:.6g} to {var_info['max']:.6g}")
                lines.append(f"    Mean: {var_info['mean']:.6g}")
                lines.append(f"    Std Dev: {var_info['std']:.6g}")
                
                if var_info.get('nan_count', 0) > 0:
                    lines.append(f"    NaN values: {var_info['nan_count']:,} ({var_info['nan_percentage']:.2f}%)")
                
                if detailed and 'median' in var_info:
                    lines.append(f"    Median: {var_info['median']:.6g}")
                    lines.append(f"    25th percentile: {var_info['percentile_25']:.6g}")
                    lines.append(f"    75th percentile: {var_info['percentile_75']:.6g}")
                    lines.append(f"    95th percentile: {var_info['percentile_95']:.6g}")
                    lines.append(f"    99th percentile: {var_info['percentile_99']:.6g}")
                
                if 'stats_note' in var_info:
                    lines.append(f"    Note: {var_info['stats_note']}")
            
            if 'note' in var_info:
                lines.append(f"    Note: {var_info['note']}")
            
            if detailed and var_info['attributes']:
                lines.append(f"    Attributes:")
                for attr_key, attr_val in var_info['attributes'].items():
                    attr_str = str(attr_val)
                    if len(attr_str) > 100:
                        attr_str = attr_str[:100] + "..."
                    lines.append(f"      {attr_key}: {attr_str}")
            
            lines.append("")
    
    lines.append("=" * 80)
    return "\n".join(lines)


def extract_all_metadata(ds: xr.Dataset, run_info: Optional[Dict] = None, 
                        detailed_stats: bool = False) -> Dict[str, Any]:
    """
    Extract all metadata from a dataset.
    
    Parameters:
    -----------
    ds : xr.Dataset
        SSCOFS dataset
    run_info : dict, optional
        Information about the model run
    detailed_stats : bool
        If True, compute detailed statistics
        
    Returns:
    --------
    dict : Dictionary with all metadata
    """
    metadata = {}
    
    # File information
    if run_info:
        metadata['file_info'] = {
            'run_date': run_info.get('run_date_utc', 'N/A'),
            'cycle': run_info.get('cycle_utc', 'N/A'),
            'forecast_hour': run_info.get('forecast_hour_index', 'N/A'),
            'url': run_info.get('url', 'N/A'),
        }
    
    # Extract all metadata components
    metadata['global_attributes'] = extract_global_attributes(ds)
    metadata['dimensions'] = extract_dimensions_info(ds)
    metadata['coordinates'] = extract_coordinates_info(ds)
    metadata['variables'] = extract_variables_info(ds, detailed_stats=detailed_stats)
    metadata['temporal_info'] = extract_temporal_info(ds)
    metadata['spatial_info'] = extract_spatial_info(ds)
    metadata['grid_connectivity'] = extract_grid_connectivity(ds)
    
    # Add extraction timestamp
    metadata['extraction_time'] = dt.datetime.now(timezone.utc).isoformat()
    
    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Extract metadata from SSCOFS NetCDF files"
    )
    
    # Data source options
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--latest",
        action="store_true",
        default=True,
        help="Use latest available data (default)"
    )
    source_group.add_argument(
        "--date",
        type=str,
        help="Date of model run in YYYY-MM-DD format"
    )
    
    parser.add_argument(
        "--cycle",
        type=int,
        choices=[0, 3, 9, 15, 21],
        help="Model cycle in UTC (00, 03, 09, 15, or 21)"
    )
    parser.add_argument(
        "--forecast",
        type=int,
        help="Forecast hour index (0-72)"
    )
    parser.add_argument(
        "--hour-of-day",
        type=int,
        help="Local hour of day (0-23) for automatic selection"
    )
    
    # Output options
    parser.add_argument(
        "--output",
        type=str,
        help="Save text output to file"
    )
    parser.add_argument(
        "--json",
        type=str,
        help="Save metadata as JSON to file"
    )
    parser.add_argument(
        "--detailed-stats",
        action="store_true",
        help="Compute detailed statistics (percentiles) for variables"
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Include detailed information in text output"
    )
    
    # Cache options
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Don't use cached data, always download fresh"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("SSCOFS METADATA EXTRACTOR")
    print("=" * 80)
    print()
    
    try:
        # Determine which data to load
        if args.date and args.cycle is not None and args.forecast is not None:
            # Use specified run
            url = build_sscofs_url(args.date, args.cycle, args.forecast)
            run_info = {
                'run_date_utc': args.date,
                'cycle_utc': f'{args.cycle:02d}z',
                'forecast_hour_index': args.forecast,
                'url': url,
            }
            print(f"Loading data for specified run:")
            print(f"  Date: {args.date}")
            print(f"  Cycle: {args.cycle:02d}z")
            print(f"  Forecast hour: {args.forecast:03d}")
        else:
            # Use latest available
            current_time_utc = dt.datetime.now(timezone.utc)
            try:
                from zoneinfo import ZoneInfo
                current_time_local = current_time_utc.astimezone(ZoneInfo("America/Los_Angeles"))
            except ImportError:
                current_time_local = current_time_utc - timedelta(hours=8)
            
            local_hour = args.hour_of_day if args.hour_of_day is not None else (current_time_local.hour * 100 + current_time_local.minute)
            
            print(f"Loading latest available data...")
            print(f"  Current time (UTC): {current_time_utc:%Y-%m-%d %H:%M:%S}")
            print(f"  Current time (local): {current_time_local:%Y-%m-%d %H:%M:%S}")
            
            run_info = latest_cycle_and_url_for_local_hour(local_hour, "America/Los_Angeles")
            print(f"  Using run: {run_info['run_date_utc']} {run_info['cycle_utc']} f{run_info['forecast_hour_index']:03d}")
        
        print()
        
        # Load the dataset
        ds = load_sscofs_data(run_info, use_cache=not args.no_cache, verbose=True)
        print()
        
        # Extract metadata
        print("Extracting metadata...")
        metadata = extract_all_metadata(ds, run_info, detailed_stats=args.detailed_stats)
        print("Done!")
        print()
        
        # Format and display
        text_output = format_metadata_text(metadata, detailed=args.detailed)
        print(text_output)
        
        # Save text output if requested
        if args.output:
            output_path = Path(args.output)
            output_path.write_text(text_output)
            print(f"\nMetadata saved to: {output_path}")
        
        # Save JSON output if requested
        if args.json:
            json_path = Path(args.json)
            with open(json_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            print(f"Metadata JSON saved to: {json_path}")
        
        return 0
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

