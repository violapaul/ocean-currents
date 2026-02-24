import requests, re, datetime as dt, datetime
from datetime import timezone
from zoneinfo import ZoneInfo
from xml.etree import ElementTree as ET

S3_LIST = "https://noaa-nos-ofs-pds.s3.amazonaws.com/"
CYCLES = [3, 9, 15, 21]  # observed for SSCOFS (03z, 09z, 15z, 21z)

KEY_RE = re.compile(r"sscofs\.t(\d{2})z\.(\d{8})\.fields\.([nf])(\d{3})\.nc$")

def list_keys_for_date(d: dt.date) -> list[str]:
    """Return all S3 object keys under sscofs/netcdf/YYYY/MM/DD/ for date d."""
    prefix = f"sscofs/netcdf/{d:%Y/%m/%d}/"
    params = {"list-type": "2", "prefix": prefix}
    r = requests.get(S3_LIST, params=params, timeout=30)
    r.raise_for_status()
    # S3 returns XML; parse all <Key>
    root = ET.fromstring(r.text)
    # S3 v2 XML uses {namespace}Key; handle namespaces robustly
    keys = []
    for elem in root.iter():
        if elem.tag.endswith("Key"):
            keys.append(elem.text)
    return keys

def newest_cycle_for_date(d: dt.date) -> tuple[int, list[str]]:
    """Return newest cycle (03/09/15/21) present for date d, and all keys."""
    keys = list_keys_for_date(d)
    present = set()
    for k in keys:
        m = KEY_RE.search(k)
        if m:
            present.add(int(m.group(1)))
    # intersect with known cycles, pick max
    available_cycles = sorted(c for c in present if c in CYCLES)
    return (available_cycles[-1], keys) if available_cycles else (None, keys)

def find_latest_cycle(max_days_back: int = 3) -> tuple[dt.date, int, list[str]]:
    """Search today, then back up to `max_days_back` days for a date that has at least one cycle."""
    today_utc = dt.datetime.utcnow().date()
    for i in range(max_days_back + 1):
        d = today_utc - dt.timedelta(days=i)
        try:
            cyc, keys = newest_cycle_for_date(d)
        except requests.HTTPError:
            continue
        if cyc is not None:
            return d, cyc, keys
    raise RuntimeError("No SSCOFS cycles found in the last few days.")

def pick_forecast_for_local_hour(local_hhmm: int, tz: str, run_date: dt.date, cycle_utc: int) -> int:
    """
    Given a local clock time (e.g., 1400 for 2pm) and timezone name,
    return the forecast hour index HHH relative to the chosen run cycle on run_date.

    Logic: convert local (run_date at local_hhmm) to UTC; forecast_hour = floor((target_utc - cycle_start_utc)/1h),
    clipped to [0, 72]. Negative => use 0.
    """
    # Build target local datetime on the *same calendar date as the cycle run date*
    hh = local_hhmm // 100
    mm = local_hhmm % 100
    local_dt = dt.datetime(run_date.year, run_date.month, run_date.day, hh, mm, tzinfo=ZoneInfo(tz))
    target_utc = local_dt.astimezone(ZoneInfo("UTC"))

    cycle_start_utc = dt.datetime(run_date.year, run_date.month, run_date.day, cycle_utc, 0, tzinfo=ZoneInfo("UTC"))
    lead_hours = (target_utc - cycle_start_utc).total_seconds() / 3600.0
    hhh = int(lead_hours // 1)
    if hhh < 0:
        hhh = 0
    if hhh > 72:
        hhh = 72
    return hhh

def build_url(run_date: dt.date, cycle_utc: int, is_forecast: bool, hour_index: int) -> str:
    tag = 'f' if is_forecast else 'n'
    return (
        f"https://noaa-nos-ofs-pds.s3.amazonaws.com/sscofs/netcdf/"
        f"{run_date:%Y/%m/%d}/sscofs.t{cycle_utc:02d}z.{run_date:%Y%m%d}.fields.{tag}{hour_index:03d}.nc"
    )

def latest_cycle_and_url_for_local_hour(local_hhmm: int, tz: str):
    run_date, cycle, keys = find_latest_cycle()
    # decide whether the desired time is in the future of the cycle (forecast) or within the nowcast window
    hhh = pick_forecast_for_local_hour(local_hhmm, tz, run_date, cycle)
    is_forecast = True  # SSCOFS distributes nowcast too, but local requested time is typically forecast usage
    url = build_url(run_date, cycle, is_forecast, hhh)
    return {
        "run_date_utc": f"{run_date:%Y-%m-%d}",
        "cycle_utc": f"{cycle:02d}z",
        "forecast_hour_index": hhh,
        "url": url
    }

if __name__ == "__main__":
    # Example: "latest cycle for my local 14:00 (2pm) in America/Los_Angeles"
    current_time_utc = dt.datetime.now(timezone.utc)
    current_time_local = current_time_utc.astimezone(ZoneInfo("America/Los_Angeles"))
    local_hour = current_time_local.hour * 100 + current_time_local.minute
    
    print(f"Current time (UTC):   {current_time_utc:%Y-%m-%d %H:%M:%S %Z}")
    print(f"Current time (local): {current_time_local:%Y-%m-%d %H:%M:%S %Z}")
    print(f"Local hour code:      {local_hour:04d}")
    print()
    
    info = latest_cycle_and_url_for_local_hour(local_hour, "America/Los_Angeles")
    
    # Parse the cycle info to calculate age
    run_date = dt.datetime.strptime(info["run_date_utc"], "%Y-%m-%d").date()
    cycle_hour = int(info["cycle_utc"].rstrip('z'))
    cycle_start = dt.datetime(run_date.year, run_date.month, run_date.day, cycle_hour, 0, tzinfo=timezone.utc)
    model_age_hours = (current_time_utc - cycle_start).total_seconds() / 3600.0
    
    print(f"Cycle start time:     {cycle_start:%Y-%m-%d %H:%M:%S %Z}")
    print(f"Model data age:       {model_age_hours:.1f} hours")
    print(f"Forecast hour index:  {info['forecast_hour_index']:03d}")
    print()
    print("URL:", info["url"])
