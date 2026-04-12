"""
Show the upcoming 15 days as predicted by Foreca and by our MOS
post-processor side-by-side.

Requirements:
  - fetch_foreca_today.py should have been run first so today's Foreca
    forecast is in cache/forecasts.csv.  If it hasn't, this script tries
    to fetch it automatically.
  - cache/ must contain the historical data (run foreca_15vrk.py + then
    ml_forecast.py at least once so the caches are populated).

Usage:
    python3 mos_preview.py
"""

from __future__ import annotations

import json
import sys
import urllib.request
from datetime import date, timedelta

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from foreca_15vrk import (
    CACHE,
    FORECA_URL,
    GRAPHS,
    HELSINKI_LAT,
    HELSINKI_LON,
    OBS_START,
    ROOT,
    USER_AGENT,
    climatology_lookup,
    fetch_observations,
    forecast_to_rows,
    http_get,
    join_all,
    parse_longfc,
)
from ml_forecast import (
    EXTRA_DAILY,
    NEARBY_SITES,
    TARGETS,
    _MOS_FORECA_FEATURES,
    _seasonal_anchors,
    build_causal_frame,
    fetch_extra_daily,
    fetch_nearby,
    fetch_teleconnections,
    fit_mos_models,
)

TODAY = date.today()
TODAY_STR = TODAY.isoformat()
# Open-Meteo's archive API only goes to ~2025-12-31. For more recent data we
# use the forecast API with past_days. The API caps past_days at 92.
_FORECAST_PAST_DAYS = 92
_FORECAST_API = "https://api.open-meteo.com/v1/forecast"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _ensure_today_forecast() -> None:
    """Fetch today's Foreca forecast if it isn't already in forecasts.csv."""
    csv_path = CACHE / "forecasts.csv"
    if csv_path.exists():
        fc = pd.read_csv(csv_path, parse_dates=["run_date"])
        if (fc["run_date"].dt.date == TODAY).any():
            return
    print("  today's forecast not found — fetching from foreca.fi ...")
    cache_file = CACHE / f"live_foreca_{TODAY.strftime('%Y%m%d')}.html"
    if not cache_file.exists():
        req = urllib.request.Request(
            FORECA_URL,
            headers={"User-Agent": USER_AGENT, "Accept-Language": "fi,en;q=0.9"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                html = resp.read().decode("utf-8", errors="replace")
        except Exception as err:
            print(f"  ERROR: could not fetch {FORECA_URL}: {err}")
            print("  Run fetch_foreca_today.py manually and retry.")
            sys.exit(1)
        cache_file.write_text(html, encoding="utf-8")
    else:
        html = cache_file.read_text(encoding="utf-8")
    entries = parse_longfc(html)
    if not entries:
        print("  ERROR: could not parse longfc_data from the Foreca page.")
        print(f"  Inspect {cache_file} and update the parser if the format changed.")
        sys.exit(1)
    rows = forecast_to_rows(TODAY, entries)
    df_new = pd.DataFrame([r.__dict__ for r in rows])
    df_new["run_date"] = pd.to_datetime(df_new["run_date"])
    df_new["target_date"] = pd.to_datetime(df_new["target_date"])
    if csv_path.exists():
        existing = pd.read_csv(csv_path, parse_dates=["run_date", "target_date"])
        pd.concat([existing, df_new], ignore_index=True).to_csv(csv_path, index=False)
    else:
        df_new.to_csv(csv_path, index=False)
    print(f"  fetched and saved {len(df_new)} rows for {TODAY}")


def _forecast_api_daily(
    lat: float, lon: float, variables: tuple[str, ...],
) -> pd.DataFrame:
    """
    Fetch recent history via Open-Meteo's forecast API with past_days.
    The archive API only covers up to ~2025-12-31; this fills the gap to
    today. Returns only dates up to and including TODAY.
    """
    url = (
        f"{_FORECAST_API}"
        f"?latitude={lat}&longitude={lon}"
        f"&daily={','.join(variables)}"
        f"&timezone=Europe%2FHelsinki"
        f"&past_days={_FORECAST_PAST_DAYS}"
        f"&forecast_days=7"
    )
    data = json.loads(http_get(url))
    daily = data["daily"]
    df = pd.DataFrame({"date": pd.to_datetime(daily["time"])})
    for k in variables:
        df[k] = daily[k]
    return df[df["date"] <= pd.Timestamp(TODAY_STR)]


def _extend_obs(obs: pd.DataFrame) -> pd.DataFrame:
    """Append recent data from the forecast API to close the gap to today."""
    if obs["date"].max() >= pd.Timestamp(TODAY_STR):
        return obs
    bridge = _forecast_api_daily(
        HELSINKI_LAT, HELSINKI_LON,
        ("temperature_2m_max", "temperature_2m_min", "precipitation_sum"),
    ).rename(columns={
        "temperature_2m_max": "obs_tmax",
        "temperature_2m_min": "obs_tmin",
        "precipitation_sum": "obs_precip",
    })
    return (
        pd.concat([obs, bridge], ignore_index=True)
        .drop_duplicates(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )


def _extend_extra(extra: pd.DataFrame) -> pd.DataFrame:
    if extra["date"].max() >= pd.Timestamp(TODAY_STR):
        return extra
    bridge = _forecast_api_daily(HELSINKI_LAT, HELSINKI_LON, EXTRA_DAILY)
    return (
        pd.concat([extra, bridge], ignore_index=True)
        .drop_duplicates(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )


def _extend_nearby(nearby: pd.DataFrame) -> pd.DataFrame:
    if nearby["date"].max() >= pd.Timestamp(TODAY_STR):
        return nearby
    frames = []
    for name, (lat, lon) in NEARBY_SITES.items():
        df = _forecast_api_daily(
            lat, lon,
            ("temperature_2m_max", "temperature_2m_min", "precipitation_sum"),
        ).rename(columns={
            "temperature_2m_max": f"{name}_tmax",
            "temperature_2m_min": f"{name}_tmin",
            "precipitation_sum": f"{name}_precip",
        })
        frames.append(df)
    bridge = frames[0]
    for df in frames[1:]:
        bridge = bridge.merge(df, on="date", how="outer")
    return (
        pd.concat([nearby, bridge], ignore_index=True)
        .drop_duplicates(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )


def _extend_clim(clim: pd.DataFrame, obs: pd.DataFrame) -> pd.DataFrame:
    """
    Extend clim to cover today + 14 days (the forecast window) by copying
    the same month-day entry from the most recent available year. The
    expanding-window average barely shifts year to year, so this is a good
    approximation.
    """
    last = clim["date"].max()
    end = pd.Timestamp(TODAY) + pd.Timedelta(days=14)
    if last >= end:
        return clim

    new_rows = []
    for d in pd.date_range(last + pd.Timedelta(days=1), end, freq="D"):
        same_doy = clim[
            (clim["date"].dt.month == d.month)
            & (clim["date"].dt.day == d.day)
            & clim["clim_tmax"].notna()
        ].sort_values("date")
        if same_doy.empty:
            continue
        row = same_doy.iloc[-1].copy()
        row["date"] = d
        new_rows.append(row)

    if not new_rows:
        return clim
    return (
        pd.concat([clim, pd.DataFrame(new_rows)], ignore_index=True)
        .sort_values("date")
        .reset_index(drop=True)
    )


def _clim_for_date(clim: pd.DataFrame, target_date: pd.Timestamp) -> pd.Series:
    """Return the clim row for target_date (looks up by month-day)."""
    candidates = clim[
        (clim["date"].dt.month == target_date.month)
        & (clim["date"].dt.day == target_date.day)
        & clim["clim_tmax"].notna()
    ].sort_values("date")
    return candidates.iloc[-1] if not candidates.empty else pd.Series(dtype=float)


# --------------------------------------------------------------------------- #
# Build today's MOS prediction frame
# --------------------------------------------------------------------------- #

def build_today_X(
    today_fc_df: pd.DataFrame,
    causal: pd.DataFrame,
    clim: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    Build the feature matrix (one row per lead) for today's MOS prediction.
    today_fc_df: rows from forecasts.csv with run_date == today.
    causal: the extended causal frame (must include d == today).
    clim: the extended clim frame (must cover target dates).
    feature_cols: the exact column list the fitted MOS models expect.
    """
    today_ts = pd.Timestamp(TODAY)
    causal_row = causal[causal["d"] == today_ts]
    if causal_row.empty:
        # Fallback: use the most recent available row
        causal_row = causal.sort_values("d").iloc[[-1]]
        print(f"  WARNING: no causal row for {TODAY}, using {causal_row['d'].iloc[0].date()}")
    causal_dict = causal_row.iloc[0].to_dict()
    # drop 'd' — not a model feature
    causal_dict.pop("d", None)

    rows = []
    for _, fc_row in today_fc_df.iterrows():
        target_date = pd.Timestamp(fc_row["target_date"])
        clim_row = _clim_for_date(clim, target_date)

        row = {**causal_dict}
        # Foreca columns
        for col in list(_MOS_FORECA_FEATURES) + ["t100max_lo", "t100max_hi",
                                                   "t100min_lo", "t100min_hi"]:
            row[col] = fc_row.get(col, np.nan)
        row["t100max_width"] = fc_row["t100max_hi"] - fc_row["t100max_lo"]
        row["t100min_width"] = fc_row["t100min_hi"] - fc_row["t100min_lo"]
        # Seasonal anchors of the target date
        doy = target_date.dayofyear
        row["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
        row["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)
        row["year"] = target_date.year
        # Climatology at the target date
        for k in ("clim_tmax", "clim_tmin", "clim_precip", "clim_prain"):
            row[k] = clim_row.get(k, np.nan)
        rows.append(row)

    X = pd.DataFrame(rows)
    # Ensure every expected column is present (fill missing with NaN)
    for c in feature_cols:
        if c not in X.columns:
            X[c] = np.nan
    return X[feature_cols]


# --------------------------------------------------------------------------- #
# Saving predictions for later scoring
# --------------------------------------------------------------------------- #

HISTORY_CSV = CACHE / "mos_preview_history.csv"


def save_predictions(
    today_fc_df: pd.DataFrame,
    mos_predictions: dict[str, np.ndarray],
    clim: pd.DataFrame,
) -> None:
    """
    Append today's Foreca + MOS + climatology predictions to
    cache/mos_preview_history.csv. One row per (run_date, target_date).

    To score later, join this file with cache/observations.csv on
    target_date — the obs_tmax/obs_tmin/obs_precip columns will tell you
    what actually happened.
    """
    target_dates = pd.to_datetime(today_fc_df["target_date"].values)

    rows = []
    for i, (_, fc_row) in enumerate(today_fc_df.iterrows()):
        td = target_dates[i]
        cr = _clim_for_date(clim, td)
        rows.append({
            "run_date":       TODAY,
            "target_date":    td.date(),
            "lead":           int(fc_row["lead"]),
            "foreca_tmax":    fc_row["tmedmax"],
            "mos_tmax":       round(float(mos_predictions["tmax"][i]), 2),
            "clim_tmax":      cr.get("clim_tmax", np.nan),
            "foreca_tmin":    fc_row["tmedmin"],
            "mos_tmin":       round(float(mos_predictions["tmin"][i]), 2),
            "clim_tmin":      cr.get("clim_tmin", np.nan),
            "foreca_precip":  fc_row["rd"],
            "mos_precip":     round(float(mos_predictions["precip"][i]), 2),
            "clim_precip":    cr.get("clim_precip", np.nan),
        })

    df_new = pd.DataFrame(rows)

    if HISTORY_CSV.exists():
        existing = pd.read_csv(HISTORY_CSV, parse_dates=["run_date", "target_date"])
        # Remove any previous rows for today (idempotent re-runs)
        existing = existing[pd.to_datetime(existing["run_date"]).dt.date != TODAY]
        combined = pd.concat([existing, df_new], ignore_index=True)
    else:
        combined = df_new

    combined = combined.sort_values(["run_date", "lead"]).reset_index(drop=True)
    combined.to_csv(HISTORY_CSV, index=False)
    print(f"  saved predictions → {HISTORY_CSV.name} "
          f"({len(combined)} total rows across "
          f"{combined['run_date'].nunique()} run dates)")


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #

def plot_preview(
    today_fc_df: pd.DataFrame,
    mos_predictions: dict[str, np.ndarray],
    clim: pd.DataFrame,
    out_path,
) -> None:
    target_dates = pd.to_datetime(today_fc_df["target_date"].values)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.subplots_adjust(hspace=0.35)

    for ax, var, label, unit, fc_col in [
        (axes[0], "tmax", "Daily maximum temperature", "°C", "tmedmax"),
        (axes[1], "tmin", "Daily minimum temperature", "°C", "tmedmin"),
        (axes[2], "precip", "Precipitation", "mm", "rd"),
    ]:
        foreca_vals = today_fc_df[fc_col].values
        mos_vals = mos_predictions[var]

        # Climatology for each target date
        clim_vals = np.array([
            _clim_for_date(clim, d).get(f"clim_{var}", np.nan)
            for d in target_dates
        ])

        ax.plot(target_dates, foreca_vals, marker="o", lw=2,
                label="Foreca (raw)", zorder=3)
        ax.plot(target_dates, mos_vals, marker="D", lw=2, ls="--",
                label="MOS corrected", zorder=4)
        ax.plot(target_dates, clim_vals, marker="", lw=1.5, ls=":",
                color="grey", label="Climatology", zorder=2)

        if var == "precip":
            # Shade precipitation difference between Foreca and MOS
            ax.bar(target_dates, foreca_vals, width=0.4, alpha=0.25,
                   color="tab:blue", label="_")

        ax.set_ylabel(f"{label} ({unit})")
        ax.set_title(label)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9, loc="upper right")

        # Mark today
        ax.axvline(pd.Timestamp(TODAY), color="black", lw=0.8, ls=":")

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    axes[-1].xaxis.set_major_locator(mdates.DayLocator(interval=2))
    fig.autofmt_xdate(rotation=40)
    fig.suptitle(
        f"Helsinki 15-day forecast — issued {TODAY}\n"
        "Foreca raw vs. MOS post-processed vs. climatology",
        fontsize=11,
    )
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path.name}")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main() -> None:
    print(f"[1/5] loading historical data (issue date: {TODAY})...")
    _ensure_today_forecast()

    obs_path = CACHE / "observations.csv"
    obs = (
        pd.read_csv(obs_path, parse_dates=["date"])
        if obs_path.exists()
        else fetch_observations()
    )
    clim_path = CACHE / "climatology.csv"
    clim = (
        pd.read_csv(clim_path, parse_dates=["date"])
        if clim_path.exists()
        else climatology_lookup(obs)
    )
    fc_all = pd.read_csv(
        CACHE / "forecasts.csv", parse_dates=["run_date", "target_date"]
    )
    today_fc_df = fc_all[fc_all["run_date"].dt.date == TODAY].copy()
    if today_fc_df.empty:
        print("ERROR: no forecast rows for today found — this shouldn't happen.")
        sys.exit(1)
    print(f"       {len(today_fc_df)} Foreca rows for today (leads 0..{today_fc_df['lead'].max()})")

    # Build the merged frame for MOS training (historical data only)
    merged_path = CACHE / "merged.csv"
    if merged_path.exists():
        merged = pd.read_csv(merged_path, parse_dates=["run_date", "target_date"])
    else:
        merged = join_all(fc_all, obs, clim)

    print("[2/5] fetching bridge data to close the gap to today...")
    obs_ext = _extend_obs(obs)
    extra_base = fetch_extra_daily()
    extra_ext = _extend_extra(extra_base)
    nearby_base = fetch_nearby()
    nearby_ext = _extend_nearby(nearby_base)
    telec = fetch_teleconnections()  # always current
    clim_ext = _extend_clim(clim, obs_ext)
    print(f"       obs extended to {obs_ext['date'].max().date()}")

    print("[3/5] building extended causal feature frame...")
    causal = build_causal_frame(obs_ext, clim_ext, extra_ext, nearby_ext, telec)
    print(f"       {len(causal)} rows, latest d = {causal['d'].max().date()}")

    print("[4/5] fitting MOS models on all historical Foreca data...")
    # Use only the historical merged rows (not today's — no obs yet)
    merged_hist = merged[merged["run_date"].dt.date < TODAY]
    # Causal frame for historical issue dates
    causal_hist = causal[causal["d"].isin(pd.to_datetime(merged_hist["run_date"]))].copy()
    models, feature_cols = fit_mos_models(merged_hist, causal_hist)
    print(f"       fitted 3 models on {len(merged_hist)} rows, {len(feature_cols)} features")

    print("[5/5] predicting and plotting...")
    X_today = build_today_X(today_fc_df, causal, clim_ext, feature_cols)
    mos_preds: dict[str, np.ndarray] = {
        var: models[var].predict(X_today) for var in TARGETS
    }
    # Clip precip predictions to ≥ 0
    mos_preds["precip"] = np.clip(mos_preds["precip"], 0, None)

    out_path = GRAPHS / "mos_preview.png"
    plot_preview(today_fc_df, mos_preds, clim_ext, out_path)
    save_predictions(today_fc_df, mos_preds, clim_ext)

    # Summary table
    print()
    print(f"=== 15-day outlook for Helsinki — {TODAY} ===")
    print()
    print(f"  {'Lead':>4}  {'Date':>10}  {'Foreca Tx':>10}  {'MOS Tx':>8}  {'Foreca Tn':>10}  {'MOS Tn':>8}  {'Foreca mm':>10}  {'MOS mm':>8}")
    print(f"  {'----':>4}  {'----------':>10}  {'----------':>10}  {'--------':>8}  {'----------':>10}  {'--------':>8}  {'----------':>10}  {'--------':>8}")
    for i, (_, row) in enumerate(today_fc_df.iterrows()):
        print(
            f"  {row['lead']:>4}  {pd.Timestamp(row['target_date']).strftime('%Y-%m-%d'):>10}"
            f"  {row['tmedmax']:>+10.1f}  {mos_preds['tmax'][i]:>+8.1f}"
            f"  {row['tmedmin']:>+10.1f}  {mos_preds['tmin'][i]:>+8.1f}"
            f"  {row['rd']:>10.1f}  {mos_preds['precip'][i]:>8.1f}"
        )


if __name__ == "__main__":
    main()
