"""
Score past MOS preview predictions against observed weather.

Joins cache/mos_preview_history.csv with cache/observations.csv on
target_date and reports MAE for Foreca, MOS, and climatology — overall
and broken down by lead time.

Usage:
    python3 score_predictions.py
"""

from __future__ import annotations

import json
import urllib.request
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache"
GRAPHS = ROOT / "graphs"
GRAPHS.mkdir(exist_ok=True)

HISTORY_CSV = CACHE / "mos_preview_history.csv"
OBS_CSV = CACHE / "observations.csv"


def _extend_observations(obs: pd.DataFrame) -> pd.DataFrame:
    """Extend observations to ~today via Open-Meteo forecast API (past_days)."""
    today = pd.Timestamp(date.today())
    if obs["date"].max() >= today - pd.Timedelta(days=1):
        return obs
    url = (
        "https://api.open-meteo.com/v1/forecast"
        "?latitude=60.1699&longitude=24.9384"
        "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum"
        "&timezone=Europe%2FHelsinki"
        "&past_days=92&forecast_days=1"
    )
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "foreca-15vrk-research/0.1"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
    except Exception as err:
        print(f"  WARNING: could not fetch recent observations: {err}")
        return obs
    daily = data["daily"]
    bridge = pd.DataFrame({
        "date": pd.to_datetime(daily["time"]),
        "obs_tmax": daily["temperature_2m_max"],
        "obs_tmin": daily["temperature_2m_min"],
        "obs_precip": daily["precipitation_sum"],
    })
    bridge = bridge[bridge["date"] <= today]
    return (
        pd.concat([obs, bridge], ignore_index=True)
        .drop_duplicates(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )


def load_scored() -> pd.DataFrame:
    if not HISTORY_CSV.exists():
        raise FileNotFoundError(
            f"{HISTORY_CSV} not found — run mos_preview.py at least once first."
        )
    if not OBS_CSV.exists():
        raise FileNotFoundError(
            f"{OBS_CSV} not found — run foreca_15vrk.py first."
        )

    hist = pd.read_csv(HISTORY_CSV, parse_dates=["run_date", "target_date"])
    obs  = _extend_observations(pd.read_csv(OBS_CSV, parse_dates=["date"]))

    scored = hist.merge(
        obs.rename(columns={"date": "target_date"}),
        on="target_date",
        how="inner",
    )
    if scored.empty:
        raise ValueError(
            "No overlap between prediction history and observations yet.\n"
            "Target dates from your predictions haven't passed — check back in a few days."
        )
    return scored


def overall_summary(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for var, obs_col in [("tmax", "obs_tmax"), ("tmin", "obs_tmin"), ("precip", "obs_precip")]:
        fc_err  = scored[f"foreca_{var}"] - scored[obs_col]
        mos_err = scored[f"mos_{var}"]    - scored[obs_col]
        cl_err  = scored[f"clim_{var}"]   - scored[obs_col]
        n = fc_err.notna().sum()
        rows.append({
            "variable":  var,
            "n":         n,
            "foreca_mae": fc_err.abs().mean(),
            "mos_mae":    mos_err.abs().mean(),
            "clim_mae":   cl_err.abs().mean(),
            "foreca_bias": fc_err.mean(),
            "mos_bias":    mos_err.mean(),
        })
    df = pd.DataFrame(rows)
    df["mos_skill"]    = 1 - df["mos_mae"]    / df["clim_mae"]
    df["foreca_skill"] = 1 - df["foreca_mae"] / df["clim_mae"]
    return df


def by_lead_summary(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lead, g in scored.groupby("lead"):
        row = {"lead": int(lead), "n": len(g)}
        for var, obs_col in [("tmax", "obs_tmax"), ("tmin", "obs_tmin"), ("precip", "obs_precip")]:
            fc_err  = g[f"foreca_{var}"] - g[obs_col]
            mos_err = g[f"mos_{var}"]    - g[obs_col]
            cl_err  = g[f"clim_{var}"]   - g[obs_col]
            row[f"foreca_mae_{var}"] = fc_err.abs().mean()
            row[f"mos_mae_{var}"]    = mos_err.abs().mean()
            row[f"clim_mae_{var}"]   = cl_err.abs().mean()
        rows.append(row)
    return pd.DataFrame(rows).sort_values("lead").reset_index(drop=True)


def plot_scores(by_lead: pd.DataFrame, n_run_dates: int, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, var, label, unit in [
        (axes[0], "tmax",   "Tmax",        "°C"),
        (axes[1], "tmin",   "Tmin",        "°C"),
        (axes[2], "precip", "Precipitation", "mm"),
    ]:
        ax.plot(by_lead["lead"], by_lead[f"foreca_mae_{var}"],
                marker="o", label="Foreca")
        ax.plot(by_lead["lead"], by_lead[f"mos_mae_{var}"],
                marker="D", ls="--", label="MOS")
        ax.plot(by_lead["lead"], by_lead[f"clim_mae_{var}"],
                marker="", ls=":", color="grey", label="Climatology")
        ax.set_xlabel("Lead time (days)")
        ax.set_ylabel(f"MAE ({unit})")
        ax.set_title(label)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle(
        f"MOS vs. Foreca vs. Climatology — scored on {n_run_dates} preview run(s)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"saved {out.name}")


def main() -> None:
    scored = load_scored()
    n_run_dates = scored["run_date"].nunique()
    n_target    = scored["target_date"].nunique()
    date_range  = f"{scored['target_date'].min().date()} .. {scored['target_date'].max().date()}"

    print(f"Scored {len(scored)} rows  "
          f"({n_run_dates} run date(s), {n_target} unique target dates: {date_range})")
    print()

    summary = overall_summary(scored)
    fmt = lambda x: f"{x:6.3f}"  # noqa: E731
    print("=== Overall MAE ===")
    print(summary[[
        "variable", "n",
        "foreca_mae", "mos_mae", "clim_mae",
        "foreca_skill", "mos_skill",
        "foreca_bias", "mos_bias",
    ]].to_string(index=False, float_format=fmt))

    print()
    by_lead = by_lead_summary(scored)
    print("=== MAE by lead time ===")
    print(by_lead.to_string(index=False, float_format=fmt))

    out = GRAPHS / "score_predictions.png"
    plot_scores(by_lead, n_run_dates, out)


if __name__ == "__main__":
    main()
