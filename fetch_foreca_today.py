"""
Fetch today's Foreca 15-day forecast directly from foreca.fi and append
to cache/forecasts.csv.

For personal / educational use only — do not run repeatedly or
aggressively. The forecast data belongs to Foreca.

If Foreca has changed their page format (the longfc_data JavaScript
variable is no longer present), the script will print a clear error
and save the raw HTML to cache/ so you can inspect the new format.

Usage:
    python3 fetch_foreca_today.py
"""

from __future__ import annotations

import math
import sys
import urllib.request
from datetime import date
from pathlib import Path

import pandas as pd

from foreca_15vrk import (
    CACHE,
    FORECA_URL,
    USER_AGENT,
    forecast_to_rows,
    parse_longfc,
)

TODAY = date.today()


def fetch_live_html() -> str:
    """Fetch the live Foreca page, cached per day to avoid repeat hits."""
    cache_file = CACHE / f"live_foreca_{TODAY.strftime('%Y%m%d')}.html"
    if cache_file.exists():
        print(f"  (using today's cached page: {cache_file.name})")
        return cache_file.read_text(encoding="utf-8")

    print(f"  fetching {FORECA_URL} ...")
    req = urllib.request.Request(
        FORECA_URL,
        headers={
            "User-Agent": USER_AGENT,
            "Accept-Language": "fi,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception as err:
        print(f"ERROR: could not fetch {FORECA_URL}: {err}")
        sys.exit(1)

    cache_file.write_text(html, encoding="utf-8")
    return html


def main() -> None:
    print(f"[1/3] fetching today's ({TODAY}) Foreca forecast...")
    html = fetch_live_html()

    print("[2/3] parsing longfc_data from page...")
    entries = parse_longfc(html)
    if not entries:
        cache_file = CACHE / f"live_foreca_{TODAY.strftime('%Y%m%d')}.html"
        print("ERROR: could not find longfc_data in the page HTML.")
        print("       Foreca may have changed their page format.")
        print(f"       Inspect {cache_file} to check the current format.")
        print("       The foreca_15vrk.py parser expects:")
        print("         var longfc_data = [...];   or   var longfc = [...];")
        sys.exit(1)

    rows = forecast_to_rows(TODAY, entries)
    if not rows:
        print("ERROR: forecast_to_rows returned no rows — check date parsing.")
        sys.exit(1)

    df_new = pd.DataFrame([r.__dict__ for r in rows])
    df_new["run_date"] = pd.to_datetime(df_new["run_date"])
    df_new["target_date"] = pd.to_datetime(df_new["target_date"])
    print(f"       parsed {len(df_new)} forecast rows (leads 0..{df_new['lead'].max()})")

    print("[3/3] appending to cache/forecasts.csv...")
    csv_path = CACHE / "forecasts.csv"
    if csv_path.exists():
        existing = pd.read_csv(csv_path, parse_dates=["run_date", "target_date"])
        already = (existing["run_date"].dt.date == TODAY).any()
        if already:
            print(f"       today ({TODAY}) already exists in forecasts.csv — skipping")
        else:
            combined = pd.concat([existing, df_new], ignore_index=True)
            combined.to_csv(csv_path, index=False)
            print(f"       appended {len(df_new)} rows → {len(combined)} total in forecasts.csv")
    else:
        df_new.to_csv(csv_path, index=False)
        print(f"       created forecasts.csv with {len(df_new)} rows")

    # Pretty-print the forecast
    print()
    print(f"=== Foreca Helsinki 15-day forecast — issued {TODAY} ===")
    print()
    print(f"  {'Lead':>4}  {'Date':>10}  {'Tmax':>6}  {'Tmin':>6}  {'mm (rd)':>8}  {'range':>12}")
    print(f"  {'----':>4}  {'----------':>10}  {'------':>6}  {'------':>6}  {'--------':>8}  {'------------':>12}")
    for r in rows:
        rd_str = f"{r.rd:5.1f}" if not math.isnan(r.rd) else "  n/a"
        print(
            f"  {r.lead:>4}  {r.target_date.strftime('%Y-%m-%d'):>10}"
            f"  {r.tmedmax:>+6.1f}  {r.tmedmin:>+6.1f}"
            f"  {rd_str:>8}  [{r.rl:.1f}..{r.rh:.1f}]"
        )


if __name__ == "__main__":
    main()
