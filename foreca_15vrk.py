"""
Foreca 15-day forecast vs. reality for Helsinki.

Forecasts are scraped from the Wayback Machine (Foreca doesn't expose any
historical-forecast archive of its own). Observations come from the
Open-Meteo historical archive, which is free and keyless. We then compare:

    forecast error (per lead time)   vs.   climatological baseline error

The climatology is a simple day-of-year mean of Tmax/Tmin over the years
prior to each forecast, so it is a legitimate "what if you had just looked
up the long-term average" baseline.

All network I/O goes through a small on-disk cache so reruns are cheap and
we don't hammer Wayback or Open-Meteo.
"""

from __future__ import annotations

import gzip
import io
import json
import math
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache"
CACHE.mkdir(exist_ok=True)
GRAPHS = ROOT / "graphs"
GRAPHS.mkdir(exist_ok=True)

FORECA_URL = "https://www.foreca.fi/Finland/Helsinki/15vrk"

# Helsinki-Kaisaniemi is the reference city station, but Open-Meteo's ERA5
# grid point for (60.17, 24.94) is close enough for a first-pass study.
HELSINKI_LAT, HELSINKI_LON = 60.1699, 24.9384

# We need observations from somewhat before the earliest snapshot so that the
# climatology baseline always has prior years to average over.
OBS_START = "2010-01-01"
OBS_END = "2025-12-31"

USER_AGENT = "foreca-15vrk-research/0.1 (+educational comparison study)"


# --------------------------------------------------------------------------- #
# Tiny HTTP helper with disk cache
# --------------------------------------------------------------------------- #

def _cache_key(url: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", url)
    if len(safe) > 180:
        safe = safe[:80] + "__" + safe[-80:]
    return CACHE / safe

def http_get(url: str, *, retries: int = 3, sleep: float = 1.5) -> str:
    """GET with gzip handling, disk cache, and polite retry/backoff."""
    key = _cache_key(url)
    if key.exists():
        return key.read_text(encoding="utf-8")
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": USER_AGENT,
                    "Accept-Encoding": "gzip",
                },
            )
            with urllib.request.urlopen(req, timeout=90) as resp:
                raw = resp.read()
                if resp.headers.get("Content-Encoding") == "gzip":
                    raw = gzip.decompress(raw)
            text = raw.decode("utf-8", errors="replace")
            key.write_text(text, encoding="utf-8")
            return text
        except Exception as err:  # noqa: BLE001
            last_err = err
            time.sleep(sleep * (attempt + 1))
    raise RuntimeError(f"failed to fetch {url}: {last_err}")


# --------------------------------------------------------------------------- #
# Wayback CDX listing
# --------------------------------------------------------------------------- #

def list_snapshots() -> list[dict]:
    """Return list of {timestamp, original} snapshots for the 15vrk page."""
    cdx_url = (
        "http://web.archive.org/cdx/search/cdx"
        "?url=foreca.fi/Finland/Helsinki/15vrk"
        "&output=json&from=2015&to=2025"
        "&filter=statuscode:200"
        "&collapse=timestamp:8"  # one per day at most
    )
    text = http_get(cdx_url)
    rows = json.loads(text)
    header, *data = rows
    return [dict(zip(header, r)) for r in data]


# --------------------------------------------------------------------------- #
# longfc_data parsing
# --------------------------------------------------------------------------- #

_LONGFC_RE = re.compile(r"var\s+longfc(?:_data)?\s*=\s*(\[[^;]*?\])\s*;")

def _js_obj_to_json(src: str) -> str:
    """Convert JS object literal (unquoted keys, single quotes) into JSON."""
    # quote object keys:  {d:..., t100max:...} -> {"d":..., "t100max":...}
    out = re.sub(
        r"([{,])\s*([A-Za-z_][A-Za-z0-9_]*)\s*:",
        r'\1"\2":',
        src,
    )
    # flip single-quoted string values to double quotes
    out = out.replace("'", '"')
    return out

def parse_longfc(html: str) -> list[dict] | None:
    m = _LONGFC_RE.search(html)
    if not m:
        return None
    try:
        return json.loads(_js_obj_to_json(m.group(1)))
    except json.JSONDecodeError:
        return None


def fetch_forecast(timestamp: str) -> list[dict] | None:
    """Fetch one Wayback snapshot and return its parsed 15-day forecast."""
    url = f"https://web.archive.org/web/{timestamp}id_/{FORECA_URL}"
    try:
        html = http_get(url)
    except RuntimeError:
        return None
    return parse_longfc(html)


# --------------------------------------------------------------------------- #
# Turning a raw forecast into a tidy frame
# --------------------------------------------------------------------------- #

@dataclass
class ForecastRow:
    run_date: date        # date the forecast was made (snapshot day)
    target_date: date     # date being forecast
    lead: int             # days ahead (0..14)
    tmedmin: float
    tmedmax: float
    # Probability intervals (lower/upper bounds of the forecast bands).
    t50max_lo: float
    t50max_hi: float
    t50min_lo: float
    t50min_hi: float
    t100max_lo: float
    t100max_hi: float
    t100min_lo: float
    t100min_hi: float
    # Precipitation: `pr` is Foreca's 1/2/3 rain-amount category,
    # `rd` is the deterministic/median mm estimate (missing on older pages),
    # `rl`/`rh` are the low/high mm bounds of the range.
    pr: int
    rd: float        # may be NaN when missing
    rl: float
    rh: float

def _parse_dt(dt: str) -> tuple[int, int]:
    """'12.5.' or '12.5' -> (12, 5)."""
    parts = [p for p in dt.strip().strip(".").split(".") if p]
    return int(parts[0]), int(parts[1])

def _pair(v) -> tuple[float, float]:
    """Normalise a Foreca interval: a list [lo, hi] or a scalar point."""
    if isinstance(v, list):
        return float(v[0]), float(v[1])
    return float(v), float(v)

def forecast_to_rows(
    run_date: date, entries: list[dict]
) -> list[ForecastRow]:
    rows: list[ForecastRow] = []
    for lead, entry in enumerate(entries):
        day, month = _parse_dt(entry["dt"])
        # Handle year rollover: forecast can cross into the next year.
        year = run_date.year
        try:
            target = date(year, month, day)
        except ValueError:
            continue
        if target < run_date:
            try:
                target = date(year + 1, month, day)
            except ValueError:
                continue
        t50max_lo, t50max_hi = _pair(entry["t50max"])
        t50min_lo, t50min_hi = _pair(entry["t50min"])
        t100max_lo, t100max_hi = _pair(entry["t100max"])
        t100min_lo, t100min_hi = _pair(entry["t100min"])
        rd_val = entry.get("rd")
        rd = float(rd_val) if rd_val is not None else float("nan")
        rows.append(
            ForecastRow(
                run_date=run_date,
                target_date=target,
                lead=lead,
                tmedmin=float(entry["tmedmin"]),
                tmedmax=float(entry["tmedmax"]),
                t50max_lo=t50max_lo, t50max_hi=t50max_hi,
                t50min_lo=t50min_lo, t50min_hi=t50min_hi,
                t100max_lo=t100max_lo, t100max_hi=t100max_hi,
                t100min_lo=t100min_lo, t100min_hi=t100min_hi,
                pr=int(entry["pr"]),
                rd=rd,
                rl=float(entry["rl"]),
                rh=float(entry["rh"]),
            )
        )
    return rows


def harvest_forecasts(snapshots: Iterable[dict]) -> pd.DataFrame:
    all_rows: list[ForecastRow] = []
    skipped = 0
    for snap in snapshots:
        ts = snap["timestamp"]
        run_date = datetime.strptime(ts[:8], "%Y%m%d").date()
        entries = fetch_forecast(ts)
        if not entries:
            skipped += 1
            continue
        all_rows.extend(forecast_to_rows(run_date, entries))
    print(f"harvested {len(all_rows)} forecast rows from "
          f"{sum(1 for _ in snapshots) - skipped} snapshots "
          f"({skipped} skipped)")
    return pd.DataFrame([r.__dict__ for r in all_rows])


# --------------------------------------------------------------------------- #
# Observed weather from Open-Meteo historical archive
# --------------------------------------------------------------------------- #

def fetch_observations(start: str = OBS_START, end: str = OBS_END) -> pd.DataFrame:
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={HELSINKI_LAT}&longitude={HELSINKI_LON}"
        f"&start_date={start}&end_date={end}"
        "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum"
        "&timezone=Europe%2FHelsinki"
    )
    data = json.loads(http_get(url))
    daily = data["daily"]
    df = pd.DataFrame({
        "date": pd.to_datetime(daily["time"]).date,
        "obs_tmax": daily["temperature_2m_max"],
        "obs_tmin": daily["temperature_2m_min"],
        "obs_precip": daily["precipitation_sum"],
    })
    df["date"] = pd.to_datetime(df["date"])
    return df


# --------------------------------------------------------------------------- #
# Climatology baseline
# --------------------------------------------------------------------------- #

def climatology_lookup(obs: pd.DataFrame) -> pd.DataFrame:
    """
    Build an expanding-window day-of-year climatology. For each date D we
    average Tmax/Tmin/precip on the same day-of-year (+/- 3 days) across all
    prior calendar years in `obs`. Expanding window avoids future leakage.

    We also compute `clim_prain` — the prior-years rain frequency on that
    day-of-year (rain defined as precip > 0.2 mm). This serves as a rain/no-
    rain climatological baseline.
    """
    obs = obs.copy()
    obs["doy"] = obs["date"].dt.dayofyear
    obs["year"] = obs["date"].dt.year
    out_rows = []
    obs_sorted = obs.sort_values("date").reset_index(drop=True)
    for _, row in obs_sorted.iterrows():
        d: pd.Timestamp = row["date"]
        window_doys = {((d.dayofyear - 1 + k) % 365) + 1 for k in range(-3, 4)}
        prior = obs_sorted[
            (obs_sorted["year"] < d.year) & (obs_sorted["doy"].isin(window_doys))
        ]
        if len(prior) == 0:
            out_rows.append((d, np.nan, np.nan, np.nan, np.nan))
        else:
            p = prior["obs_precip"]
            out_rows.append((
                d,
                prior["obs_tmax"].mean(),
                prior["obs_tmin"].mean(),
                p.mean(),
                (p > 0.2).mean(),
            ))
    return pd.DataFrame(
        out_rows,
        columns=["date", "clim_tmax", "clim_tmin", "clim_precip", "clim_prain"],
    )


# --------------------------------------------------------------------------- #
# Join + error metrics
# --------------------------------------------------------------------------- #

def join_all(fc: pd.DataFrame, obs: pd.DataFrame, clim: pd.DataFrame) -> pd.DataFrame:
    fc = fc.copy()
    fc["target_date"] = pd.to_datetime(fc["target_date"])
    fc["run_date"] = pd.to_datetime(fc["run_date"])
    merged = fc.merge(
        obs.rename(columns={"date": "target_date"}),
        on="target_date",
        how="left",
    ).merge(
        clim.rename(columns={"date": "target_date"}),
        on="target_date",
        how="left",
    )
    merged = merged.dropna(
        subset=["obs_tmax", "obs_tmin", "clim_tmax", "clim_tmin"]
    )
    # Temperature errors (signed and absolute are computed later as needed).
    merged["err_fc_tmax"] = merged["tmedmax"] - merged["obs_tmax"]
    merged["err_fc_tmin"] = merged["tmedmin"] - merged["obs_tmin"]
    merged["err_cl_tmax"] = merged["clim_tmax"] - merged["obs_tmax"]
    merged["err_cl_tmin"] = merged["clim_tmin"] - merged["obs_tmin"]
    # Interval coverage flags.
    merged["in_t50max"] = (
        (merged["obs_tmax"] >= merged["t50max_lo"])
        & (merged["obs_tmax"] <= merged["t50max_hi"])
    )
    merged["in_t100max"] = (
        (merged["obs_tmax"] >= merged["t100max_lo"])
        & (merged["obs_tmax"] <= merged["t100max_hi"])
    )
    merged["in_t50min"] = (
        (merged["obs_tmin"] >= merged["t50min_lo"])
        & (merged["obs_tmin"] <= merged["t50min_hi"])
    )
    merged["in_t100min"] = (
        (merged["obs_tmin"] >= merged["t100min_lo"])
        & (merged["obs_tmin"] <= merged["t100min_hi"])
    )
    # Precipitation: use `rd` when available, otherwise mid-range of [rl, rh].
    merged["fc_precip"] = merged["rd"].fillna(
        (merged["rl"] + merged["rh"]) / 2
    )
    merged["in_precip_range"] = (
        (merged["obs_precip"] >= merged["rl"])
        & (merged["obs_precip"] <= merged["rh"])
    )
    RAIN_THR = 0.2
    merged["obs_rain"] = merged["obs_precip"] > RAIN_THR
    # Foreca "yes, rain" = point forecast (rd or [rl,rh] midpoint)
    # exceeds the threshold. Using rh instead makes POD trivially 1.0.
    merged["fc_rain"] = merged["fc_precip"] > RAIN_THR
    merged["clim_rain"] = merged["clim_prain"] > 0.5
    merged["err_fc_precip"] = merged["fc_precip"] - merged["obs_precip"]
    merged["err_cl_precip"] = merged["clim_precip"] - merged["obs_precip"]
    return merged


def _metrics(err: pd.Series) -> dict[str, float]:
    return {
        "mae": err.abs().mean(),
        "rmse": math.sqrt((err ** 2).mean()),
        "bias": err.mean(),
        "n": len(err),
    }

def summarise_by_lead(m: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lead, g in m.groupby("lead"):
        rows.append({
            "lead": lead,
            "n": len(g),
            "fc_mae_tmax": g["err_fc_tmax"].abs().mean(),
            "cl_mae_tmax": g["err_cl_tmax"].abs().mean(),
            "fc_mae_tmin": g["err_fc_tmin"].abs().mean(),
            "cl_mae_tmin": g["err_cl_tmin"].abs().mean(),
            "fc_bias_tmax": g["err_fc_tmax"].mean(),
            "fc_bias_tmin": g["err_fc_tmin"].mean(),
        })
    out = pd.DataFrame(rows).sort_values("lead").reset_index(drop=True)
    # Murphy skill score vs climatology, per variable.
    out["skill_tmax"] = 1 - (out["fc_mae_tmax"] / out["cl_mae_tmax"])
    out["skill_tmin"] = 1 - (out["fc_mae_tmin"] / out["cl_mae_tmin"])
    return out


def interval_coverage_by_lead(m: pd.DataFrame) -> pd.DataFrame:
    """
    For each lead time, what fraction of observations fell inside each
    probability band? A well-calibrated 50% band should read ~0.50 and a
    well-calibrated 100% band should read near 1.0.

    We also report the mean width of each band, so one can see whether
    Foreca achieves coverage by honest skill or by widening the envelope.
    """
    rows = []
    for lead, g in m.groupby("lead"):
        rows.append({
            "lead": lead,
            "n": len(g),
            "cov_t50max": g["in_t50max"].mean(),
            "cov_t100max": g["in_t100max"].mean(),
            "cov_t50min": g["in_t50min"].mean(),
            "cov_t100min": g["in_t100min"].mean(),
            "w_t50max": (g["t50max_hi"] - g["t50max_lo"]).mean(),
            "w_t100max": (g["t100max_hi"] - g["t100max_lo"]).mean(),
            "w_t50min": (g["t50min_hi"] - g["t50min_lo"]).mean(),
            "w_t100min": (g["t100min_hi"] - g["t100min_lo"]).mean(),
        })
    return pd.DataFrame(rows).sort_values("lead").reset_index(drop=True)


def precipitation_by_lead(m: pd.DataFrame) -> pd.DataFrame:
    """
    Per-lead-time precipitation metrics:
      - MAE of the point precipitation forecast in mm
      - MAE of the climatology baseline in mm
      - Accuracy / hit rate / false-alarm for rain yes/no (threshold 0.2 mm)
      - Coverage of the [rl, rh] envelope
    """
    rows = []
    for lead, g in m.groupby("lead"):
        obs_rain = g["obs_rain"]
        fc_rain = g["fc_rain"]
        clim_rain = g["clim_rain"]
        hits = ((obs_rain) & (fc_rain)).sum()
        misses = ((obs_rain) & (~fc_rain)).sum()
        false_alarms = ((~obs_rain) & (fc_rain)).sum()
        rain_events = max(obs_rain.sum(), 1)
        rows.append({
            "lead": lead,
            "n": len(g),
            "fc_mae_precip": g["err_fc_precip"].abs().mean(),
            "cl_mae_precip": g["err_cl_precip"].abs().mean(),
            "fc_bias_precip": g["err_fc_precip"].mean(),
            "cov_precip_range": g["in_precip_range"].mean(),
            "rain_accuracy": (fc_rain == obs_rain).mean(),
            "clim_rain_accuracy": (clim_rain == obs_rain).mean(),
            "pod": hits / rain_events,                  # prob. of detection
            "far": false_alarms / max(fc_rain.sum(), 1),  # false alarm ratio
            "csi": hits / max(hits + misses + false_alarms, 1),  # threat
        })
    out = pd.DataFrame(rows).sort_values("lead").reset_index(drop=True)
    out["skill_precip"] = 1 - (out["fc_mae_precip"] / out["cl_mae_precip"])
    return out


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #

def plot_error_vs_lead(summary: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, var, label in [
        (axes[0], "tmax", "Daily Tmax"),
        (axes[1], "tmin", "Daily Tmin"),
    ]:
        ax.plot(summary["lead"], summary[f"fc_mae_{var}"],
                marker="o", label="Foreca 15vrk")
        ax.plot(summary["lead"], summary[f"cl_mae_{var}"],
                marker="s", label="Climatology")
        ax.set_xlabel("Lead time (days)")
        ax.set_title(label)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Mean absolute error (\u00b0C)")
    axes[0].legend()
    fig.suptitle("Foreca Helsinki 15-day forecast vs. climatology")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)

def plot_skill(summary: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axhline(0, color="black", lw=0.8)
    ax.plot(summary["lead"], summary["skill_tmax"], marker="o", label="Tmax")
    ax.plot(summary["lead"], summary["skill_tmin"], marker="s", label="Tmin")
    ax.set_xlabel("Lead time (days)")
    ax.set_ylabel("Skill vs. climatology (1 - MAE_fc / MAE_clim)")
    ax.set_title("Forecast skill decay — positive = beats climatology")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_interval_coverage(cov: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, var, label in [
        (axes[0], "max", "Daily Tmax bands"),
        (axes[1], "min", "Daily Tmin bands"),
    ]:
        ax.axhline(0.5, color="tab:blue", lw=0.8, ls=":")
        ax.axhline(1.0, color="tab:orange", lw=0.8, ls=":")
        ax.plot(cov["lead"], cov[f"cov_t50{var}"], marker="o",
                label="50% band actual cov.")
        ax.plot(cov["lead"], cov[f"cov_t100{var}"], marker="s",
                label="100% band actual cov.")
        ax.set_xlabel("Lead time (days)")
        ax.set_ylim(0, 1.05)
        ax.set_title(label)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Fraction of days observation fell in band")
    axes[0].legend(loc="lower right")
    fig.suptitle(
        "Foreca probability-interval calibration (dotted = nominal)"
    )
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_interval_widths(cov: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(cov["lead"], cov["w_t50max"], marker="o",
            label="Tmax 50% band width")
    ax.plot(cov["lead"], cov["w_t100max"], marker="s",
            label="Tmax 100% band width")
    ax.plot(cov["lead"], cov["w_t50min"], marker="o", ls="--",
            label="Tmin 50% band width")
    ax.plot(cov["lead"], cov["w_t100min"], marker="s", ls="--",
            label="Tmin 100% band width")
    ax.set_xlabel("Lead time (days)")
    ax.set_ylabel("Mean band width (\u00b0C)")
    ax.set_title("Foreca forecast bands widen with lead time")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_precipitation(psum: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    # Left: MAE mm
    ax = axes[0]
    ax.plot(psum["lead"], psum["fc_mae_precip"], marker="o",
            label="Foreca (rd / mid)")
    ax.plot(psum["lead"], psum["cl_mae_precip"], marker="s",
            label="Climatology")
    ax.set_xlabel("Lead time (days)")
    ax.set_ylabel("Precipitation MAE (mm)")
    ax.set_title("Daily precipitation MAE")
    ax.grid(alpha=0.3)
    ax.legend()
    # Right: rain/no-rain accuracy + POD + FAR
    ax = axes[1]
    ax.plot(psum["lead"], psum["rain_accuracy"], marker="o",
            label="Foreca rain yes/no accuracy")
    ax.plot(psum["lead"], psum["clim_rain_accuracy"], marker="s",
            label="Climatology accuracy")
    ax.plot(psum["lead"], psum["pod"], marker="^", ls="--",
            color="tab:green", label="Foreca POD (hit rate)")
    ax.plot(psum["lead"], psum["far"], marker="v", ls="--",
            color="tab:red", label="Foreca false-alarm ratio")
    ax.set_xlabel("Lead time (days)")
    ax.set_ylabel("Fraction")
    ax.set_ylim(0, 1.05)
    ax.set_title("Rain / no-rain classification")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    fig.suptitle("Foreca Helsinki precipitation forecast vs. climatology")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main() -> None:
    print("[1/5] listing Wayback snapshots...")
    snapshots = list_snapshots()
    print(f"       {len(snapshots)} snapshots found "
          f"({snapshots[0]['timestamp']} .. {snapshots[-1]['timestamp']})")

    print("[2/5] fetching forecasts...")
    fc = harvest_forecasts(snapshots)
    fc.to_csv(CACHE / "forecasts.csv", index=False)

    print("[3/5] fetching observed weather (Open-Meteo archive)...")
    obs = fetch_observations()
    obs.to_csv(CACHE / "observations.csv", index=False)

    print("[4/5] building climatology baseline...")
    clim_cache = CACHE / "climatology.csv"
    if clim_cache.exists():
        clim = pd.read_csv(clim_cache, parse_dates=["date"])
    else:
        clim = climatology_lookup(obs)
        clim.to_csv(clim_cache, index=False)

    print("[5/5] joining + computing errors...")
    merged = join_all(fc, obs, clim)
    merged.to_csv(CACHE / "merged.csv", index=False)
    summary = summarise_by_lead(merged)
    summary.to_csv(CACHE / "summary_by_lead.csv", index=False)
    cov = interval_coverage_by_lead(merged)
    cov.to_csv(CACHE / "interval_coverage.csv", index=False)
    psum = precipitation_by_lead(merged)
    psum.to_csv(CACHE / "precipitation_by_lead.csv", index=False)

    fmt = lambda x: f"{x:6.2f}"  # noqa: E731
    print()
    print("=== MAE by lead time (°C) ===")
    print(summary[[
        "lead", "n",
        "fc_mae_tmax", "cl_mae_tmax", "skill_tmax",
        "fc_mae_tmin", "cl_mae_tmin", "skill_tmin",
    ]].to_string(index=False, float_format=fmt))

    print()
    print("=== Temperature interval coverage by lead time ===")
    print("   (nominal: 50% band should be 0.50, 100% band should be ~1.00)")
    print(cov[[
        "lead", "n",
        "cov_t50max", "cov_t100max", "cov_t50min", "cov_t100min",
        "w_t50max", "w_t100max",
    ]].to_string(index=False, float_format=fmt))

    print()
    print("=== Precipitation by lead time ===")
    print("   (mm MAE, rain accuracy, POD=hit rate, FAR=false alarm, CSI=threat)")
    print(psum[[
        "lead", "n",
        "fc_mae_precip", "cl_mae_precip", "skill_precip",
        "rain_accuracy", "clim_rain_accuracy",
        "pod", "far", "csi", "cov_precip_range",
    ]].to_string(index=False, float_format=fmt))

    plot_error_vs_lead(summary, GRAPHS / "mae_vs_lead.png")
    plot_skill(summary, GRAPHS / "skill_vs_lead.png")
    plot_interval_coverage(cov, GRAPHS / "interval_coverage.png")
    plot_interval_widths(cov, GRAPHS / "interval_widths.png")
    plot_precipitation(psum, GRAPHS / "precipitation.png")
    print()
    print("wrote graphs/mae_vs_lead.png, graphs/skill_vs_lead.png, "
          "graphs/interval_coverage.png, graphs/interval_widths.png, "
          "graphs/precipitation.png")


if __name__ == "__main__":
    main()
