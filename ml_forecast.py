"""
ML weather forecasting for Helsinki — companion to foreca_15vrk.py.

Two heads:

  - **Standalone**. HistGradientBoostingRegressor trained on ERA5 daily
    observations + a few extra Open-Meteo daily variables + nearby-station
    temperatures + NOAA daily NAO/AO indices. Predicts Tmax / Tmin / precip
    1..14 days ahead. Compared against the same day-of-year climatology
    baseline that foreca_15vrk.py uses, and against Foreca on the dates
    where Wayback snapshots happen to fall in our test window.

  - **MOS post-processor**. HistGradientBoostingRegressor trained on the
    merged Foreca rows, with Foreca's published forecast as input features.
    One model per variable spanning all leads (~1000 rows total — too small
    to per-lead train). Compares MOS-corrected forecast against raw Foreca
    on held-out snapshots.

Data sources reused from foreca_15vrk.py:
  - cache/observations.csv  (Open-Meteo ERA5 daily Helsinki)
  - cache/forecasts.csv     (Wayback-recovered Foreca 15-day forecasts)
  - cache/climatology.csv   (day-of-year expanding-window climatology)
  - cache/merged.csv        (joined frame from foreca_15vrk.join_all)

New data (all cached on first call via foreca_15vrk.http_get):
  - Extra Open-Meteo daily variables for Helsinki
  - Open-Meteo daily Tmax/Tmin/precip for Stockholm, Tallinn, St Petersburg
  - NOAA CPC daily NAO and AO indices

Note: We considered upper-air features (500 hPa geopotential, 850 hPa T)
but Open-Meteo's historical-forecast-api only exposes them back to 2022,
which doesn't cover the 2011-2020 standalone training window or most
Foreca snapshots. Skipped in v1; would be a natural follow-up if a longer
upper-air archive becomes available.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from foreca_15vrk import (
    CACHE,
    GRAPHS,
    HELSINKI_LAT,
    HELSINKI_LON,
    OBS_END,
    OBS_START,
    ROOT,
    climatology_lookup,
    fetch_observations,
    http_get,
    join_all,
)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

LAGS = (0, 1, 2, 3, 5, 7)  # 0 = the issue date itself ("today's obs")
LEADS = list(range(1, 15))             # 1..14 days ahead
TARGETS = ("tmax", "tmin", "precip")

# Standalone head temporal split (on the issue date d, with target_date
# constrained to stay in the same window so that no train label leaks past
# the cutoff).
STANDALONE_TRAIN_START = pd.Timestamp("2011-01-08")  # 7-day lag warm-up
STANDALONE_TRAIN_END = pd.Timestamp("2020-12-31")
STANDALONE_TEST_START = pd.Timestamp("2021-01-01")
STANDALONE_TEST_END = pd.Timestamp("2025-12-31")

# MOS head temporal split, on run_date.
MOS_TRAIN_END = pd.Timestamp("2022-12-31")
MOS_TEST_START = pd.Timestamp("2023-01-01")

NEARBY_SITES = {
    "stockholm": (59.33, 18.07),
    "tallinn": (59.44, 24.75),
    "stpetersburg": (59.94, 30.32),
}

EXTRA_DAILY = (
    "pressure_msl_mean",
    "wind_speed_10m_max",
    "wind_direction_10m_dominant",
    "relative_humidity_2m_mean",
    "cloud_cover_mean",
    "shortwave_radiation_sum",
)

NAO_URL = "https://ftp.cpc.ncep.noaa.gov/cwlinks/norm.daily.nao.index.b500101.current.ascii"
AO_URL = "https://ftp.cpc.ncep.noaa.gov/cwlinks/norm.daily.ao.index.b500101.current.ascii"

GBM_KWARGS = dict(
    max_depth=6,
    learning_rate=0.05,
    max_iter=400,
    early_stopping=True,
    random_state=42,
)


# --------------------------------------------------------------------------- #
# New data fetchers (all share foreca_15vrk's http_get / cache/)
# --------------------------------------------------------------------------- #

def fetch_extra_daily(
    lat: float = HELSINKI_LAT,
    lon: float = HELSINKI_LON,
    start: str = OBS_START,
    end: str = OBS_END,
    variables: tuple[str, ...] = EXTRA_DAILY,
) -> pd.DataFrame:
    """Open-Meteo archive endpoint with extended `daily=` list."""
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={start}&end_date={end}"
        f"&daily={','.join(variables)}"
        "&timezone=Europe%2FHelsinki"
    )
    data = json.loads(http_get(url))
    daily = data["daily"]
    df = pd.DataFrame({"date": pd.to_datetime(daily["time"])})
    for k in variables:
        df[k] = daily[k]
    return df


def fetch_nearby(
    sites: dict[str, tuple[float, float]] = NEARBY_SITES,
    start: str = OBS_START,
    end: str = OBS_END,
) -> pd.DataFrame:
    """Daily Tmax/Tmin/precip for each nearby site as a wide frame keyed by date."""
    out: pd.DataFrame | None = None
    for name, (lat, lon) in sites.items():
        url = (
            "https://archive-api.open-meteo.com/v1/archive"
            f"?latitude={lat}&longitude={lon}"
            f"&start_date={start}&end_date={end}"
            "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum"
            "&timezone=Europe%2FHelsinki"
        )
        data = json.loads(http_get(url))
        d = data["daily"]
        df = pd.DataFrame({
            "date": pd.to_datetime(d["time"]),
            f"{name}_tmax": d["temperature_2m_max"],
            f"{name}_tmin": d["temperature_2m_min"],
            f"{name}_precip": d["precipitation_sum"],
        })
        out = df if out is None else out.merge(df, on="date", how="outer")
    assert out is not None
    return out.sort_values("date").reset_index(drop=True)


def _parse_cpc_daily(text: str, name: str) -> pd.DataFrame:
    """NOAA CPC daily index files are 4 columns: year month day value."""
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 4:
            continue
        try:
            y, mo, d = int(parts[0]), int(parts[1]), int(parts[2])
            val = float(parts[3])
        except (ValueError, TypeError):
            continue
        try:
            rows.append((date(y, mo, d), val))
        except ValueError:
            continue
    df = pd.DataFrame(rows, columns=["date", name])
    df["date"] = pd.to_datetime(df["date"])
    return df


def fetch_teleconnections() -> pd.DataFrame:
    """Daily NAO and AO indices from NOAA CPC."""
    nao = _parse_cpc_daily(http_get(NAO_URL), "nao")
    ao = _parse_cpc_daily(http_get(AO_URL), "ao")
    return (
        nao.merge(ao, on="date", how="outer")
        .sort_values("date")
        .reset_index(drop=True)
    )


# --------------------------------------------------------------------------- #
# Causal feature frame
# --------------------------------------------------------------------------- #

def build_causal_frame(
    obs: pd.DataFrame,
    clim: pd.DataFrame,
    extra: pd.DataFrame,
    nearby: pd.DataFrame,
    telec: pd.DataFrame,
) -> pd.DataFrame:
    """
    Wide daily frame indexed by issue date `d`. Every column is information
    available *by end of day d*: lags, lagged anomalies, lagged extras,
    lagged nearby-station obs, lagged teleconnections. The day-of-year
    climatology and target-date anchors are joined later by build_xy(),
    because they depend on the *target* date, not d.
    """
    df = (
        obs.merge(clim, on="date", how="left")
        .merge(extra, on="date", how="left")
        .merge(nearby, on="date", how="left")
        .merge(telec, on="date", how="left")
        .sort_values("date")
        .reset_index(drop=True)
    )

    out = pd.DataFrame({"d": df["date"]})

    # Lagged local observations
    for col in ("obs_tmax", "obs_tmin", "obs_precip"):
        for k in LAGS:
            out[f"{col}_lag{k}"] = df[col].shift(k)
    for k in LAGS:
        out[f"obs_rain_lag{k}"] = (df["obs_precip"].shift(k) > 0.2).astype(float)

    # Lagged anomalies relative to climatology (lag 0 = today's anomaly)
    for k in (0, 1, 3, 7):
        out[f"anom_tmax_lag{k}"] = df["obs_tmax"].shift(k) - df["clim_tmax"].shift(k)
        out[f"anom_tmin_lag{k}"] = df["obs_tmin"].shift(k) - df["clim_tmin"].shift(k)
        out[f"anom_precip_lag{k}"] = df["obs_precip"].shift(k) - df["clim_precip"].shift(k)

    # Smoothed recent anomaly (regime indicator)
    out["anom_tmax_mean7"] = (df["obs_tmax"] - df["clim_tmax"]).shift(1).rolling(7).mean()
    out["anom_tmin_mean7"] = (df["obs_tmin"] - df["clim_tmin"]).shift(1).rolling(7).mean()
    out["anom_precip_mean7"] = (df["obs_precip"] - df["clim_precip"]).shift(1).rolling(7).mean()

    # Extra Open-Meteo daily variables (lag 0 = today, plus 1..3 days back)
    for col in EXTRA_DAILY:
        for k in (0, 1, 2, 3):
            out[f"{col}_lag{k}"] = df[col].shift(k)

    # Nearby-station obs (lag 0..2 days; advection from west is fast)
    nearby_cols = [c for c in nearby.columns if c != "date"]
    for col in nearby_cols:
        for k in (0, 1, 2):
            out[f"{col}_lag{k}"] = df[col].shift(k)

    # Teleconnection indices (lag 0 = today's NAO/AO, plus 1..3 days back)
    for col in ("nao", "ao"):
        for k in (0, 1, 2, 3):
            out[f"{col}_lag{k}"] = df[col].shift(k)

    return out


def _seasonal_anchors(target_date: pd.Series) -> pd.DataFrame:
    """sin/cos doy + raw year, indexed identically to target_date."""
    doy = target_date.dt.dayofyear
    return pd.DataFrame({
        "sin_doy": np.sin(2 * np.pi * doy / 365.25),
        "cos_doy": np.cos(2 * np.pi * doy / 365.25),
        "year": target_date.dt.year,
    }, index=target_date.index)


def build_xy(
    causal: pd.DataFrame,
    obs: pd.DataFrame,
    clim: pd.DataFrame,
    lead: int,
    var: str,
) -> pd.DataFrame:
    """
    For (lead, var): pull `causal` features at d, attach climatology +
    seasonal anchors of d+lead, attach obs[var] at d+lead as label `y`.
    Returns one row per d. Caller is responsible for splitting into
    train/test by `d`.
    """
    rows = causal.copy()
    rows["target_date"] = rows["d"] + pd.Timedelta(days=lead)
    # Climatology at the target date (the value the model is adjusting away from)
    rows = rows.merge(
        clim.rename(columns={"date": "target_date"}),
        on="target_date", how="left",
    )
    # Seasonal encoding of the target date
    anchors = _seasonal_anchors(rows["target_date"])
    rows = pd.concat([rows, anchors], axis=1)
    # Label
    var_col = f"obs_{var}"
    obs_slim = obs[["date", var_col]].rename(
        columns={"date": "target_date", var_col: "y"}
    )
    rows = rows.merge(obs_slim, on="target_date", how="left")
    return rows


# --------------------------------------------------------------------------- #
# Standalone head
# --------------------------------------------------------------------------- #

# Columns excluded from the feature matrix (metadata or label)
_NON_FEATURE_COLS = {"d", "target_date", "y"}


def _split_features_label(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    feat_cols = [c for c in frame.columns if c not in _NON_FEATURE_COLS]
    return frame[feat_cols], frame["y"]


def train_standalone(
    causal: pd.DataFrame,
    obs: pd.DataFrame,
    clim: pd.DataFrame,
) -> pd.DataFrame:
    """
    Train one HGBR per (variable, lead) on 2011-2020, predict on 2021-2025.
    Returns a long predictions frame: columns d, target_date, lead, var,
    pred, y.
    """
    all_rows = []
    for var in TARGETS:
        for lead in LEADS:
            xy = build_xy(causal, obs, clim, lead, var)
            xy = xy.dropna(subset=["y"])

            train = xy[
                (xy["d"] >= STANDALONE_TRAIN_START)
                & (xy["target_date"] <= STANDALONE_TRAIN_END)
            ]
            test = xy[
                (xy["d"] >= STANDALONE_TEST_START)
                & (xy["target_date"] <= STANDALONE_TEST_END)
            ]

            X_train, y_train = _split_features_label(train)
            X_test, y_test = _split_features_label(test)

            model = HistGradientBoostingRegressor(**GBM_KWARGS)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)

            preds = test[["d", "target_date"]].copy()
            preds["lead"] = lead
            preds["var"] = var
            preds["pred"] = pred
            preds["y"] = y_test.values
            all_rows.append(preds)
    return pd.concat(all_rows, ignore_index=True)


# --------------------------------------------------------------------------- #
# MOS head — Foreca-aware post-processor
# --------------------------------------------------------------------------- #

# Foreca-side feature whitelist
_MOS_FORECA_FEATURES = (
    "tmedmax", "tmedmin",
    "t50max_lo", "t50max_hi", "t50min_lo", "t50min_hi",
    "t100max_lo", "t100max_hi", "t100min_lo", "t100min_hi",
    "pr", "rd", "rl", "rh",
    "lead",
)


def _build_mos_frame(
    merged: pd.DataFrame,
    causal: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join the foreca merged rows with the causal feature frame at d=run_date.
    Adds Foreca's own forecast as features and the same target-date anchors
    used by the standalone head.
    """
    m = merged.copy()
    m["target_date"] = pd.to_datetime(m["target_date"])
    m["run_date"] = pd.to_datetime(m["run_date"])
    m = m.rename(columns={"run_date": "d"})
    m = m.merge(causal, on="d", how="left")

    anchors = _seasonal_anchors(m["target_date"])
    extras = pd.DataFrame({
        "t100max_width": m["t100max_hi"] - m["t100max_lo"],
        "t100min_width": m["t100min_hi"] - m["t100min_lo"],
    }, index=m.index)
    return pd.concat([m, anchors, extras], axis=1)


def _mos_feature_cols(causal: pd.DataFrame, full: pd.DataFrame) -> list[str]:
    """Deduplicated list of feature columns used by every MOS model."""
    causal_cols = [c for c in causal.columns if c != "d"]
    raw = (
        causal_cols
        + ["sin_doy", "cos_doy", "year"]
        + list(_MOS_FORECA_FEATURES) + ["t100max_width", "t100min_width"]
        + ["clim_tmax", "clim_tmin", "clim_precip", "clim_prain"]
    )
    seen: set[str] = set()
    out: list[str] = []
    for c in raw:
        if c not in seen and c in full.columns:
            seen.add(c)
            out.append(c)
    return out


def fit_mos_models(
    merged: pd.DataFrame,
    causal: pd.DataFrame,
) -> tuple[dict, list[str]]:
    """
    Fit one HGBR per variable on ALL available Foreca data (no held-out
    test set). Intended for live prediction in mos_preview.py, where
    maximising training coverage matters more than evaluating held-out
    skill. Returns (models_dict, feature_cols).
    """
    full = _build_mos_frame(merged, causal)
    feature_cols = _mos_feature_cols(causal, full)
    models: dict[str, HistGradientBoostingRegressor] = {}
    for var in TARGETS:
        df = full.dropna(subset=[f"obs_{var}"])
        model = HistGradientBoostingRegressor(**GBM_KWARGS)
        model.fit(df[feature_cols], df[f"obs_{var}"])
        models[var] = model
    return models, feature_cols


def train_mos(
    merged: pd.DataFrame,
    causal: pd.DataFrame,
) -> pd.DataFrame:
    """
    Train one HGBR per variable on the merged Foreca frame, with `lead` as
    a feature so a single model spans all leads. Returns predictions on
    the held-out snapshots.
    """
    full = _build_mos_frame(merged, causal)
    feature_cols = _mos_feature_cols(causal, full)

    train = full[full["d"] <= MOS_TRAIN_END]
    test = full[full["d"] >= MOS_TEST_START]

    pred_rows = []
    for var in TARGETS:
        target_col = f"obs_{var}"
        train_v = train.dropna(subset=[target_col])
        test_v = test.dropna(subset=[target_col])

        X_train = train_v[feature_cols]
        y_train = train_v[target_col]
        X_test = test_v[feature_cols]
        y_test = test_v[target_col]

        model = HistGradientBoostingRegressor(**GBM_KWARGS)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        out = test_v[["d", "target_date", "lead"]].copy()
        out["var"] = var
        out["pred"] = pred
        out["y"] = y_test.values
        # Also carry the raw Foreca forecast for the same row, for
        # head-to-head MAE without re-merging.
        if var == "tmax":
            out["foreca"] = test_v["tmedmax"].values
        elif var == "tmin":
            out["foreca"] = test_v["tmedmin"].values
        else:  # precip
            out["foreca"] = test_v["fc_precip"].values
        pred_rows.append(out)
    return pd.concat(pred_rows, ignore_index=True)


# --------------------------------------------------------------------------- #
# Evaluation helpers
# --------------------------------------------------------------------------- #

def summarise_standalone(
    preds: pd.DataFrame,
    clim: pd.DataFrame,
) -> pd.DataFrame:
    """
    Per-lead MAE for ML and climatology, three variables. preds must have
    columns d, target_date, lead, var, pred, y.
    """
    wide = (
        preds.pivot_table(
            index=["d", "target_date", "lead"],
            columns="var",
            values=["pred", "y"],
        )
        .reset_index()
    )
    wide.columns = ["_".join(c).rstrip("_") for c in wide.columns.values]
    wide = wide.merge(
        clim.rename(columns={"date": "target_date"}),
        on="target_date", how="left",
    )

    rows = []
    for lead, g in wide.groupby("lead"):
        row = {"lead": int(lead), "n": len(g)}
        for v in TARGETS:
            err_ml = (g[f"pred_{v}"] - g[f"y_{v}"]).abs()
            err_cl = (g[f"clim_{v}"] - g[f"y_{v}"]).abs()
            row[f"ml_mae_{v}"] = err_ml.mean()
            row[f"cl_mae_{v}"] = err_cl.mean()
            denom = row[f"cl_mae_{v}"]
            row[f"skill_{v}"] = (
                1 - row[f"ml_mae_{v}"] / denom if denom and denom > 0 else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("lead").reset_index(drop=True)


def summarise_standalone_vs_foreca(
    preds: pd.DataFrame,
    merged: pd.DataFrame,
) -> pd.DataFrame:
    """
    Head-to-head: ML vs Foreca on the dates where a Foreca snapshot exists
    in the standalone test window. Joined on (d=run_date, lead).
    """
    p = preds.copy()
    p["d"] = pd.to_datetime(p["d"])
    p["target_date"] = pd.to_datetime(p["target_date"])

    m = merged.copy()
    m["run_date"] = pd.to_datetime(m["run_date"])
    m["target_date"] = pd.to_datetime(m["target_date"])
    m = m[(m["run_date"] >= STANDALONE_TEST_START) & (m["run_date"] <= STANDALONE_TEST_END)]
    m = m.rename(columns={"run_date": "d"})

    # Pivot ML preds wide
    ml_wide = p.pivot_table(
        index=["d", "lead"], columns="var", values="pred",
    ).reset_index()
    ml_wide.columns = ["d", "lead"] + [f"ml_{v}" for v in ml_wide.columns[2:]]
    # Match foreca rows
    j = m.merge(ml_wide, on=["d", "lead"], how="inner")

    rows = []
    for lead, g in j.groupby("lead"):
        row = {"lead": int(lead), "n": len(g)}
        # Tmax / Tmin
        for v, fc_col in (("tmax", "tmedmax"), ("tmin", "tmedmin")):
            row[f"ml_mae_{v}"] = (g[f"ml_{v}"] - g[f"obs_{v}"]).abs().mean()
            row[f"fc_mae_{v}"] = (g[fc_col] - g[f"obs_{v}"]).abs().mean()
            row[f"cl_mae_{v}"] = (g[f"clim_{v}"] - g[f"obs_{v}"]).abs().mean()
        # Precipitation: ML uses pred, Foreca uses fc_precip
        row["ml_mae_precip"] = (g["ml_precip"] - g["obs_precip"]).abs().mean()
        row["fc_mae_precip"] = (g["fc_precip"] - g["obs_precip"]).abs().mean()
        row["cl_mae_precip"] = (g["clim_precip"] - g["obs_precip"]).abs().mean()
        rows.append(row)
    return pd.DataFrame(rows).sort_values("lead").reset_index(drop=True)


def summarise_mos(mos_preds: pd.DataFrame) -> pd.DataFrame:
    """Per-lead MAE for MOS prediction vs raw Foreca on the same rows."""
    rows = []
    for lead, g in mos_preds.groupby("lead"):
        row = {"lead": int(lead), "n_rows": len(g) // len(TARGETS)}
        for v in TARGETS:
            gv = g[g["var"] == v]
            if len(gv) == 0:
                continue
            row[f"mos_mae_{v}"] = (gv["pred"] - gv["y"]).abs().mean()
            row[f"fc_mae_{v}"] = (gv["foreca"] - gv["y"]).abs().mean()
        rows.append(row)
    return pd.DataFrame(rows).sort_values("lead").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #

def plot_ml_mae_vs_lead(
    standalone: pd.DataFrame,
    head_to_head: pd.DataFrame,
    mos: pd.DataFrame,
    out: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, var, label in [
        (axes[0], "tmax", "Daily Tmax"),
        (axes[1], "tmin", "Daily Tmin"),
    ]:
        ax.plot(standalone["lead"], standalone[f"ml_mae_{var}"],
                marker="o", label="ML standalone (full test set)")
        ax.plot(standalone["lead"], standalone[f"cl_mae_{var}"],
                marker="s", label="Climatology (full test set)")
        if not head_to_head.empty:
            ax.plot(head_to_head["lead"], head_to_head[f"fc_mae_{var}"],
                    marker="^", ls="--", label="Foreca (snapshot subset)")
            ax.plot(head_to_head["lead"], head_to_head[f"ml_mae_{var}"],
                    marker="v", ls="--", label="ML standalone (snapshot subset)")
        if not mos.empty:
            ax.plot(mos["lead"], mos[f"mos_mae_{var}"],
                    marker="D", ls=":", label="MOS post-processor (snapshot subset)")
        ax.set_xlabel("Lead time (days)")
        ax.set_title(label)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Mean absolute error (\u00b0C)")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Helsinki temperature: ML vs. climatology vs. Foreca vs. MOS")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_ml_skill(standalone: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axhline(0, color="black", lw=0.8)
    ax.plot(standalone["lead"], standalone["skill_tmax"],
            marker="o", label="Tmax")
    ax.plot(standalone["lead"], standalone["skill_tmin"],
            marker="s", label="Tmin")
    ax.plot(standalone["lead"], standalone["skill_precip"],
            marker="^", label="Precipitation")
    ax.set_xlabel("Lead time (days)")
    ax.set_ylabel("Skill vs. climatology (1 - MAE_ml / MAE_clim)")
    ax.set_title("ML standalone forecast skill — positive = beats climatology")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_mos_improvement(mos: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, var, label, unit in [
        (axes[0], "tmax", "Tmax", "\u00b0C"),
        (axes[1], "tmin", "Tmin", "\u00b0C"),
        (axes[2], "precip", "Precip", "mm"),
    ]:
        delta = mos[f"mos_mae_{var}"] - mos[f"fc_mae_{var}"]
        colors = ["tab:green" if d < 0 else "tab:red" for d in delta]
        ax.bar(mos["lead"], delta, color=colors)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xlabel("Lead time (days)")
        ax.set_ylabel(f"\u0394 MAE ({unit}, MOS - Foreca)")
        ax.set_title(label)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle(
        "MOS post-processor vs. raw Foreca (negative bar = MOS improves)"
    )
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_ml_precipitation(
    standalone: pd.DataFrame,
    head_to_head: pd.DataFrame,
    out: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.plot(standalone["lead"], standalone["ml_mae_precip"],
            marker="o", label="ML standalone (full test set)")
    ax.plot(standalone["lead"], standalone["cl_mae_precip"],
            marker="s", label="Climatology (full test set)")
    if not head_to_head.empty:
        ax.plot(head_to_head["lead"], head_to_head["fc_mae_precip"],
                marker="^", ls="--", label="Foreca (snapshot subset)")
        ax.plot(head_to_head["lead"], head_to_head["ml_mae_precip"],
                marker="v", ls="--", label="ML (snapshot subset)")
    ax.set_xlabel("Lead time (days)")
    ax.set_ylabel("Precipitation MAE (mm)")
    ax.set_title("Helsinki precipitation: ML vs. climatology vs. Foreca")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Data loading (use foreca_15vrk caches if present)
# --------------------------------------------------------------------------- #

def load_or_build_observations() -> pd.DataFrame:
    cached = CACHE / "observations.csv"
    if cached.exists():
        df = pd.read_csv(cached, parse_dates=["date"])
    else:
        df = fetch_observations()
        df.to_csv(cached, index=False)
    return df


def load_or_build_climatology(obs: pd.DataFrame) -> pd.DataFrame:
    cached = CACHE / "climatology.csv"
    if cached.exists():
        return pd.read_csv(cached, parse_dates=["date"])
    clim = climatology_lookup(obs)
    clim.to_csv(cached, index=False)
    return clim


def load_or_build_merged(
    obs: pd.DataFrame, clim: pd.DataFrame
) -> pd.DataFrame:
    cached = CACHE / "merged.csv"
    if cached.exists():
        return pd.read_csv(cached, parse_dates=["run_date", "target_date"])
    fc = pd.read_csv(
        CACHE / "forecasts.csv", parse_dates=["run_date", "target_date"]
    )
    merged = join_all(fc, obs, clim)
    merged.to_csv(cached, index=False)
    return merged


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main() -> None:
    print("[1/8] loading observations + climatology + merged forecasts...")
    obs = load_or_build_observations()
    clim = load_or_build_climatology(obs)
    merged = load_or_build_merged(obs, clim)
    print(f"       {len(obs)} obs days, {len(merged)} merged forecast rows")

    print("[2/8] fetching extra Open-Meteo daily variables for Helsinki...")
    extra = fetch_extra_daily()

    print("[3/8] fetching nearby station observations...")
    nearby = fetch_nearby()

    print("[4/8] fetching NOAA CPC daily NAO and AO indices...")
    telec = fetch_teleconnections()

    print("[5/8] building causal feature frame...")
    causal = build_causal_frame(obs, clim, extra, nearby, telec)
    print(f"       {len(causal)} rows, {causal.shape[1] - 1} feature columns")

    print("[6/8] training standalone HGBR head (3 vars x 14 leads = 42 models)...")
    standalone_preds = train_standalone(causal, obs, clim)
    standalone_preds.to_csv(CACHE / "ml_standalone_predictions.csv", index=False)
    standalone_summary = summarise_standalone(standalone_preds, clim)
    standalone_summary.to_csv(CACHE / "ml_standalone_summary.csv", index=False)
    head_to_head = summarise_standalone_vs_foreca(standalone_preds, merged)
    head_to_head.to_csv(CACHE / "ml_vs_foreca_subset.csv", index=False)

    print("[7/8] training MOS post-processor head (3 vars across all leads)...")
    mos_preds = train_mos(merged, causal)
    mos_preds.to_csv(CACHE / "ml_mos_predictions.csv", index=False)
    mos_summary = summarise_mos(mos_preds)
    mos_summary.to_csv(CACHE / "ml_mos_summary.csv", index=False)

    print("[8/8] plotting + sanity checks...")
    plot_ml_mae_vs_lead(standalone_summary, head_to_head, mos_summary, GRAPHS / "ml_mae_vs_lead.png")
    plot_ml_skill(standalone_summary, GRAPHS / "ml_skill_vs_lead.png")
    plot_ml_precipitation(standalone_summary, head_to_head, GRAPHS / "ml_precipitation.png")
    plot_mos_improvement(mos_summary, GRAPHS / "ml_mos_improvement.png")

    fmt = lambda x: f"{x:6.2f}"  # noqa: E731

    print()
    print("=== Standalone ML — full test set MAE by lead ===")
    print(standalone_summary[[
        "lead", "n",
        "ml_mae_tmax", "cl_mae_tmax", "skill_tmax",
        "ml_mae_tmin", "cl_mae_tmin", "skill_tmin",
        "ml_mae_precip", "cl_mae_precip", "skill_precip",
    ]].to_string(index=False, float_format=fmt))

    print()
    print("=== Standalone ML vs. Foreca — snapshot-subset MAE by lead ===")
    if head_to_head.empty:
        print("   (no overlapping snapshots in the test window)")
    else:
        print(head_to_head[[
            "lead", "n",
            "ml_mae_tmax", "fc_mae_tmax",
            "ml_mae_tmin", "fc_mae_tmin",
            "ml_mae_precip", "fc_mae_precip",
        ]].to_string(index=False, float_format=fmt))

    print()
    print("=== MOS post-processor MAE by lead (vs. raw Foreca on same rows) ===")
    print(mos_summary.to_string(index=False, float_format=fmt))

    # Sanity checks
    print()
    print("=== Sanity checks ===")
    s1 = standalone_summary.iloc[0]
    print(f"  ML lead-1 Tmax MAE: {s1['ml_mae_tmax']:.2f}  "
          f"(should be < climatology {s1['cl_mae_tmax']:.2f})")
    if s1["ml_mae_tmax"] >= s1["cl_mae_tmax"]:
        print("  WARNING: lead-1 ML Tmax does not beat climatology — feature bug?")
    if s1["ml_mae_tmax"] < 0.5:
        print("  WARNING: lead-1 ML Tmax suspiciously low — possible leakage")
    monotone = standalone_summary["ml_mae_tmax"].diff().dropna().ge(-0.3).all()
    print(f"  ML Tmax MAE roughly monotone in lead: {monotone}")
    print()
    print("wrote graphs/ml_mae_vs_lead.png, graphs/ml_skill_vs_lead.png, "
          "graphs/ml_precipitation.png, graphs/ml_mos_improvement.png")


if __name__ == "__main__":
    main()
