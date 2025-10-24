"""
fetch_sscofs.py
-----------------

This module provides a helper function to construct a valid SSCOFS
NetCDF file URL for a given date, model run cycle and forecast hour,
as well as an example routine that uses ``xarray`` to download and plot
surface current speed data.  The SSCOFS (Salish Sea & Columbia River
Operational Forecast System) model is published as part of NOAA's
Operational Forecast System archive on the ``noaa-nos-ofs-pds`` open
data S3 bucket.  File names follow the pattern described in NOAA's
documentation:

    sscofs.tCCz.YYYYMMDD.fields.[n|f]HHH.nc

where:

* ``CC`` is the two‑digit UTC cycle (00, 03, 09, 15 or 21);
* ``YYYYMMDD`` is the date of the run;
* ``n`` denotes a nowcast hour and ``f`` denotes a forecast hour;
* ``HHH`` is the three‑digit hour index.

Example usage:

>>> from fetch_sscofs import build_sscofs_url
>>> url = build_sscofs_url('2025-07-31', cycle=3, forecast_hour=3)
>>> print(url)
https://noaa-nos-ofs-pds.s3.amazonaws.com/sscofs/netcdf/2025/07/31/sscofs.t03z.20250731.fields.f003.nc

The script can also be executed as a module to download and plot
current speed using xarray (if installed):

    python fetch_sscofs.py --date 2025-07-31 --cycle 3 --forecast 3

This example will download the forecast file for 03 UTC on 31 July 2025
and produce a simple map of surface current speed.

Note
====
The Python environment in which this file is created may not include
scientific libraries such as ``xarray`` or ``cartopy``.  The example
code catches ``ImportError`` exceptions and will simply print the
constructed URL if the necessary libraries are unavailable.  To run
the plotting example on your own machine, ensure that you have
``xarray``, ``matplotlib``, ``s3fs`` and optionally ``cartopy``
installed.
"""

import argparse
from datetime import datetime, timedelta
from typing import Literal


def build_sscofs_url(
    date_str: str,
    cycle: int,
    forecast_hour: int,
    product: Literal["fields", "stations", "regulargrid"] = "fields",
    nowcast: bool = False,
) -> str:
    """Construct the SSCOFS NetCDF file URL.

    Parameters
    ----------
    date_str : str
        Date of the model run in ``YYYY-MM-DD`` format (UTC).
    cycle : int
        Model cycle in hours (UTC).  Valid values are 0, 3, 9, 15 or 21.
    forecast_hour : int
        Forecast or nowcast hour index.  This should be a non‑negative
        integer.  The value will be zero‑padded to three digits.
    product : {"fields", "stations", "regulargrid"}, optional
        Product type, default is "fields".  "stations" produces a
        time‑series file, while "regulargrid" refers to a uniformly
        interpolated grid.  Note that regular grid files are much
        larger than the unstructured fields.
    nowcast : bool, optional
        If ``True``, construct a nowcast ("nHHH") file; otherwise
        construct a forecast ("fHHH") file.  Default is ``False``.

    Returns
    -------
    str
        Fully qualified HTTPS URL to the NetCDF file in the
        ``noaa-nos-ofs-pds`` S3 bucket.

    Examples
    --------
    >>> build_sscofs_url('2025-07-31', cycle=3, forecast_hour=3)
    'https://noaa-nos-ofs-pds.s3.amazonaws.com/sscofs/netcdf/2025/07/31/sscofs.t03z.20250731.fields.f003.nc'
    >>> build_sscofs_url('2024-11-19', cycle=21, forecast_hour=0, nowcast=True)
    'https://noaa-nos-ofs-pds.s3.amazonaws.com/sscofs/netcdf/2024/11/19/sscofs.t21z.20241119.fields.n000.nc'
    """
    # Parse and validate the date string
    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError as e:
        raise ValueError(f"Invalid date format: {date_str!r}. Expected YYYY-MM-DD.") from e

    # Ensure cycle is one of the valid runs
    valid_cycles = {0, 3, 9, 15, 21}
    if cycle not in valid_cycles:
        raise ValueError(f"Invalid cycle {cycle}. Valid cycles are {sorted(valid_cycles)}.")

    # Compose path components
    date_path = date_obj.strftime("%Y/%m/%d")
    yyyymmdd = date_obj.strftime("%Y%m%d")
    cycle_str = f"{cycle:02d}"
    hour_str = f"{forecast_hour:03d}"
    suffix = "n" if nowcast else "f"
    file_name = f"sscofs.t{cycle_str}z.{yyyymmdd}.{product}.{suffix}{hour_str}.nc"

    # Construct the full URL
    return (
        "https://noaa-nos-ofs-pds.s3.amazonaws.com/sscofs/netcdf/"
        f"{date_path}/{file_name}"
    )


def example_plot(
    date_str: str, cycle: int, forecast_hour: int, *, nowcast: bool = False, use_cache: bool = True
) -> None:
    """Example function to download an SSCOFS file and plot surface current speed.

    This function attempts to import ``xarray`` and ``matplotlib``.  If
    either library is missing, a message will be printed and the
    constructed URL will be displayed instead.  When the necessary
    libraries are available, the function downloads the specified
    NetCDF file using the URL returned by :func:`build_sscofs_url`,
    extracts surface velocity components (``u`` and ``v``) at the
    first sigma layer, computes current speed, and displays a simple
    map.

    Parameters
    ----------
    date_str, cycle, forecast_hour : see :func:`build_sscofs_url`
    nowcast : bool, optional
        Whether to fetch a nowcast file instead of a forecast file.
    use_cache : bool, optional
        If True, use cached files when available. If False, always download fresh.
    """
    url = build_sscofs_url(date_str, cycle, forecast_hour, nowcast=nowcast)
    try:
        import xarray as xr
        import numpy as np
        import matplotlib.pyplot as plt
        from sscofs_cache import load_sscofs_data

        # Build run_info dict for cache
        run_info = {
            'run_date_utc': date_str,
            'cycle_utc': f'{cycle:02d}z',
            'forecast_hour_index': forecast_hour,
            'url': url
        }
        
        # Load data using shared cache
        ds = load_sscofs_data(run_info, use_cache=use_cache, verbose=True)

        # Extract surface currents (first sigma layer)
        u = ds["u"].isel(siglay=0)
        v = ds["v"].isel(siglay=0)
        speed_data = np.sqrt(u**2 + v**2)

        # Plot the first time step
        plt.figure(figsize=(8, 5))
        # Create a simple scatter plot since this is unstructured data
        plt.scatter(ds["lonc"], ds["latc"], c=speed_data.isel(time=0), 
                   cmap="viridis", s=1, alpha=0.7)
        plt.colorbar(label="Current speed (m/s)")
        plt.title(
            f"SSCOFS surface current speed – {date_str} cycle {cycle:02d} forecast hour {forecast_hour}"
        )
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.show()
    except ImportError as exc:
        print(
            "Required libraries for data download or plotting are missing: "
            f"{exc.name}.\n"
            f"Constructed URL: {url}"
        )


# -----------------------------------------------------------------------------
# Helper functions to determine the latest available cycle and forecast index
# -----------------------------------------------------------------------------

from datetime import time, timezone
try:
    # ``zoneinfo`` is available in Python 3.9+.  It may not exist in earlier
    # versions.  If unavailable, the functions below will assume UTC for
    # conversions.
    from zoneinfo import ZoneInfo  # type: ignore
except ImportError:  # pragma: no cover - zoneinfo may not be present
    ZoneInfo = None  # type: ignore


def _latest_cycle_for_time(dt_utc: datetime) -> tuple[datetime.date, int]:
    """Determine the latest SSCOFS run cycle available relative to a UTC time.

    SSCOFS cycles start at 00, 03, 09, 15 and 21 UTC.  Given a UTC
    ``datetime``, this function returns the date and cycle hour for the
    most recent cycle that is not in the future relative to ``dt_utc``.

    Parameters
    ----------
    dt_utc : datetime
        Current time in UTC.

    Returns
    -------
    tuple
        A tuple ``(date, cycle_hour)`` where ``date`` is the run date (as
        ``datetime.date``) and ``cycle_hour`` is the integer cycle hour.
    """
    cycles = [0, 3, 9, 15, 21]
    # Find the latest cycle <= current hour
    for hour in reversed(cycles):
        if dt_utc.hour >= hour:
            return dt_utc.date(), hour
    # If no cycle in the current day is <= current hour, use the last cycle of
    # the previous day (21z)
    prev_date = dt_utc.date() - timedelta(days=1)
    return prev_date, cycles[-1]


def compute_latest_file_for_local_hour(
    now: datetime,
    local_hour: int,
    tz_str: str = "America/Los_Angeles",
    max_forecast_hours: int = 72,
) -> tuple[str, datetime.date, int, int]:
    """Compute the URL, run date, cycle and forecast index for a local hour.

    Given the current UTC time and a desired local hour of the day (e.g., 14
    for 2 p.m. local), this function determines the most recent SSCOFS run
    cycle and the forecast hour index that covers the desired time.  It
    returns the constructed NetCDF URL along with the run date, cycle and
    forecast index.  If the computed forecast hour exceeds ``max_forecast_hours``
    (default 72), it is capped at that value.

    Parameters
    ----------
    now : datetime
        Current time in UTC (must be timezone‑aware).
    local_hour : int
        Hour of the day in the user's local timezone (0–23).
    tz_str : str, optional
        IANA timezone name for the user's location.  Default is
        ``'America/Los_Angeles'``.
    max_forecast_hours : int, optional
        Maximum forecast horizon to use when computing the index.  Values
        beyond this will be capped to prevent indexing beyond the model
        range.  Default is 72 hours.

    Returns
    -------
    tuple
        A tuple ``(url, run_date, cycle_hour, forecast_hour_index)`` where
        ``url`` is the constructed NetCDF file URL and the other values
        specify the run cycle and hour used.
    """
    if now.tzinfo is None or now.tzinfo.utcoffset(now) is None:
        raise ValueError("'now' must be a timezone‑aware UTC datetime")
    if not (0 <= local_hour <= 23):
        raise ValueError("local_hour must be in the range 0–23")

    # Determine latest cycle date and hour
    run_date, cycle_hour = _latest_cycle_for_time(now)
    cycle_start_utc = datetime.combine(run_date, time(hour=cycle_hour), tzinfo=timezone.utc)

    # Convert cycle start to local time zone for comparison
    if ZoneInfo is not None:
        tz = ZoneInfo(tz_str)
        cycle_start_local = cycle_start_utc.astimezone(tz)
        # Target time on the same local date as cycle start
        target_local = datetime.combine(
            cycle_start_local.date(), time(hour=local_hour), tzinfo=tz
        )
        # Compute difference in hours; if negative, wrap forward by 24 hours
        diff_hours = (target_local - cycle_start_local).total_seconds() / 3600.0
        if diff_hours < 0:
            diff_hours += 24
    else:
        # Fallback: assume local=UTC
        cycle_start_local = cycle_start_utc
        diff_hours = (local_hour - cycle_start_utc.hour) % 24

    # Round down to nearest integer forecast hour
    forecast_hour_index = int(diff_hours)
    if forecast_hour_index > max_forecast_hours:
        forecast_hour_index = max_forecast_hours

    url = build_sscofs_url(
        run_date.isoformat(), cycle=cycle_hour, forecast_hour=forecast_hour_index, nowcast=False
    )
    return url, run_date, cycle_hour, forecast_hour_index


def compute_file_for_datetime(
    target_datetime: datetime,
    tz_str: str = "America/Los_Angeles",
    max_forecast_hours: int = 72,
    forecast_hour_override: int = None,
) -> dict:
    """Find the closest SSCOFS model file for a specific datetime in local timezone.

    Given a target datetime in a local timezone, this function determines the best
    SSCOFS model run cycle and forecast hour that most closely matches the target
    time. It searches for the most recent cycle that has a forecast covering the
    target time.

    Parameters
    ----------
    target_datetime : datetime
        Target date and time. Can be timezone-aware or naive. If naive, it will
        be assumed to be in the timezone specified by tz_str.
    tz_str : str, optional
        IANA timezone name for the target datetime if it's naive, or for
        interpreting the target. Default is ``'America/Los_Angeles'``.
    max_forecast_hours : int, optional
        Maximum forecast horizon to consider. Default is 72 hours.
    forecast_hour_override : int, optional
        If specified, use this forecast hour instead of calculating from target_datetime.
        This allows you to specify which model run to use (via target_datetime) and
        then select a different forecast hour from that run.

    Returns
    -------
    dict
        Dictionary containing:
        - 'url': str - The constructed NetCDF file URL
        - 'run_date_utc': str - Run date in UTC (format: "YYYY-MM-DD")
        - 'cycle_utc': str - Cycle hour in UTC (format: "03z", "21z", etc.)
        - 'forecast_hour_index': int - Forecast hour index
        - 'target_datetime_local': datetime - Target time in local timezone
        - 'target_datetime_utc': datetime - Target time in UTC
        - 'cycle_start_utc': datetime - Cycle start time in UTC

    Examples
    --------
    >>> from datetime import datetime
    >>> from zoneinfo import ZoneInfo
    >>> target = datetime(2025, 10, 17, 14, 30, tzinfo=ZoneInfo("America/Los_Angeles"))
    >>> info = compute_file_for_datetime(target)
    >>> print(info['url'])
    """
    # Handle timezone-naive datetimes
    if target_datetime.tzinfo is None:
        if ZoneInfo is not None:
            tz = ZoneInfo(tz_str)
            target_local = target_datetime.replace(tzinfo=tz)
        else:
            # Fallback: treat as UTC
            target_local = target_datetime.replace(tzinfo=timezone.utc)
    else:
        target_local = target_datetime
        if ZoneInfo is not None:
            tz = ZoneInfo(tz_str)
            # Convert to the specified timezone if it's not already
            target_local = target_local.astimezone(tz)

    # Convert to UTC for cycle calculations
    target_utc = target_local.astimezone(timezone.utc)

    # Find the latest cycle that's before or at the target time
    # We want a cycle that has a forecast hour covering the target
    run_date, cycle_hour = _latest_cycle_for_time(target_utc)
    cycle_start_utc = datetime.combine(run_date, time(hour=cycle_hour), tzinfo=timezone.utc)

    # Calculate hours from cycle start to target
    hours_from_cycle = (target_utc - cycle_start_utc).total_seconds() / 3600.0

    # If target is before cycle start (shouldn't happen with _latest_cycle_for_time,
    # but just in case), go back one cycle
    if hours_from_cycle < 0:
        # Find the previous cycle
        cycles = [0, 3, 9, 15, 21]
        try:
            cycle_idx = cycles.index(cycle_hour)
            if cycle_idx > 0:
                cycle_hour = cycles[cycle_idx - 1]
            else:
                # Go to previous day, last cycle
                run_date = run_date - timedelta(days=1)
                cycle_hour = cycles[-1]
        except ValueError:
            # Shouldn't happen
            run_date = run_date - timedelta(days=1)
            cycle_hour = 21

        cycle_start_utc = datetime.combine(run_date, time(hour=cycle_hour), tzinfo=timezone.utc)
        hours_from_cycle = (target_utc - cycle_start_utc).total_seconds() / 3600.0

    # Use override if specified, otherwise calculate from target datetime
    if forecast_hour_override is not None:
        forecast_hour_index = forecast_hour_override
        # Ensure it's within valid range
        if forecast_hour_index < 0:
            forecast_hour_index = 0
        elif forecast_hour_index > max_forecast_hours:
            forecast_hour_index = max_forecast_hours
    else:
        # Round to nearest hour for forecast index
        forecast_hour_index = round(hours_from_cycle)
        
        # Ensure forecast hour is within valid range
        if forecast_hour_index < 0:
            forecast_hour_index = 0
        elif forecast_hour_index > max_forecast_hours:
            # If target is too far in future, cap at max
            forecast_hour_index = max_forecast_hours

    # Build URL
    url = build_sscofs_url(
        run_date.isoformat(),
        cycle=cycle_hour,
        forecast_hour=forecast_hour_index,
        nowcast=False
    )

    # Return comprehensive info
    # Format run_date and cycle to match the format expected by sscofs_cache
    return {
        'url': url,
        'run_date_utc': run_date.isoformat(),  # Convert date to string "YYYY-MM-DD"
        'cycle_utc': f"{cycle_hour:02d}z",  # Format as "03z", "21z", etc.
        'forecast_hour_index': forecast_hour_index,
        'target_datetime_local': target_local,
        'target_datetime_utc': target_utc,
        'cycle_start_utc': cycle_start_utc,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download and plot SSCOFS surface current speed.  You can either "
            "explicitly specify a run date, cycle and forecast hour or provide "
            "a local hour of the day to automatically select the latest run."
        )
    )
    # Group for explicit parameters
    parser.add_argument(
        "--date",
        type=str,
        help="Date of the model run in YYYY-MM-DD format (UTC).  If omitted, the script will compute the latest available run based on the current time.",
    )
    parser.add_argument(
        "--cycle",
        type=int,
        choices=[0, 3, 9, 15, 21],
        help="Model run cycle in UTC (00, 03, 09, 15 or 21).  Ignored if --hour-of-day is provided.",
    )
    parser.add_argument(
        "--forecast",
        type=int,
        help="Forecast hour index (0–72).  Ignored if --hour-of-day is provided.",
    )
    parser.add_argument(
        "--nowcast",
        action="store_true",
        help="Fetch a nowcast file instead of a forecast file when --date and --cycle are specified.",
    )
    # Group for automatic selection based on local hour
    parser.add_argument(
        "--hour-of-day",
        type=int,
        help=(
            "Local hour of day (0–23).  If provided, the script will determine the latest "
            "available SSCOFS run and the forecast hour covering that time.  This argument "
            "is mutually exclusive with --date/--cycle/--forecast."
        ),
    )
    parser.add_argument(
        "--timezone",
        type=str,
        default="America/Los_Angeles",
        help="IANA timezone name for interpreting --hour-of-day (default: America/Los_Angeles)",
    )
    # Cache management options
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Don't use cached data, always download fresh"
    )
    parser.add_argument(
        "--list-cache",
        action="store_true",
        help="List cached files and exit"
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear all cached files and exit"
    )
    args = parser.parse_args()

    # Handle cache management commands first
    if args.list_cache:
        from sscofs_cache import list_cache
        list_cache()
        return
    
    if args.clear_cache:
        from sscofs_cache import clear_cache
        clear_cache()
        return

    if args.hour_of_day is not None:
        # Automatic mode
        now_utc = datetime.now(timezone.utc)
        url, run_date, cycle, fhour = compute_latest_file_for_local_hour(
            now_utc, args.hour_of_day, tz_str=args.timezone
        )
        print(
            f"Latest available cycle: {run_date} at {cycle:02d}z (UTC).\n"
            f"Forecast hour index covering {args.hour_of_day:02d}:00 {args.timezone}: {fhour:03d}.\n"
            f"URL: {url}"
        )
        # Attempt to plot using the computed parameters
        try:
            example_plot(run_date.isoformat(), cycle, fhour, nowcast=False, use_cache=not args.no_cache)
        except Exception as exc:
            # Catch any errors from example_plot for better reporting
            print(f"Unable to plot data: {exc}")
        return

    # Explicit mode requires date, cycle and forecast
    if args.date is None or args.cycle is None or args.forecast is None:
        parser.error(
            "When not using --hour-of-day, you must specify --date, --cycle and --forecast."
        )
    if args.forecast < 0:
        parser.error("Forecast hour index must be non‑negative.")
    example_plot(args.date, args.cycle, args.forecast, nowcast=args.nowcast, use_cache=not args.no_cache)


if __name__ == "__main__":
    main()