# Archived Python Scripts

These scripts have been moved from the main Python_SSCOFS directory to streamline the codebase.
They are preserved here for reference but are not part of the active codebase.

## Archived Files

### plot_currents_simple.py
- **Purpose**: Simplified standalone plotting without dependencies on other modules
- **Status**: Functionality duplicated in `plot_local_currents.py`
- **Reason for removal**: Not truly standalone (uses same dependencies) and duplicates main functionality

### diagnose_currents.py  
- **Purpose**: Examine current data statistics and identify features
- **Status**: Useful for debugging but not core functionality
- **Reason for removal**: Specialized debugging tool not needed for normal usage

### extract_sscofs_metadata.py
- **Purpose**: Extract and analyze SSCOFS NetCDF metadata
- **Status**: Already generated `metadata.txt` file in parent directory
- **Reason for removal**: One-time tool that has served its purpose

### plot_wet_nodes.py
- **Purpose**: Visualize FVCOM grid wet nodes
- **Status**: Very specialized use case
- **Reason for removal**: Too specialized for general use

### test_bulk_download.py
- **Purpose**: Test bulk data downloads from NOAA S3
- **Status**: Testing script
- **Reason for removal**: Development/testing script not needed for production

### test_water_mask.py
- **Purpose**: Test land/water masking functionality
- **Status**: Experimental testing script
- **Reason for removal**: Not used by any other module

## Notes

- These files were archived on 2025-10-23 to reduce codebase complexity
- The core functionality remains available through the main scripts:
  - `plot_local_currents.py` - Basic visualization
  - `plot_currents_enhanced.py` - Advanced tactical visualization
  - `sscofs_cache.py` - Data management
  - `latest_cycle.py` - Cycle determination
  - `fetch_sscofs.py` - S3 access
  - `basemap_utils.py` - Mapping utilities

If you need to restore any of these files, simply move them back to the parent directory.
