# Foreca 15-day forecast vs. reality — Helsinki

How good is Foreca's 15-day forecast for Helsinki, and would you do just as
well by looking up the historical average for that date? This is a small
study that tries to answer both questions with publicly available data.

![Forecast skill decay](graphs/skill_vs_lead.png)

## The problem

Foreca publishes a 15-day forecast for Helsinki at
`foreca.fi/Finland/Helsinki/15vrk`, but — like most forecast providers — it
does **not** expose an archive of past forecasts. If you want to know how
the day-10 forecast from 18 months ago compared to what actually happened,
you have to reconstruct it yourself.

## Approach

1. **Historical forecasts** are recovered from the
   [Internet Archive's Wayback Machine](https://web.archive.org/). The CDX
   API lists every archived copy of the 15-day page; for each snapshot we
   fetch the raw HTML and parse the JavaScript variable `longfc_data` (or
   the older `longfc`) that contains the 15-day forecast Foreca emitted at
   that point in time. We never contact `foreca.fi` directly.
2. **Observed weather** for Helsinki comes from the
   [Open-Meteo historical archive](https://open-meteo.com/en/docs/historical-weather-api)
   (daily Tmax, Tmin, and precipitation sum from ERA5 reanalysis at 60.17°N,
   24.94°E). No API key required.
3. **Climatology baseline** — the "what if you just used the historical
   average?" comparison — is built as an expanding-window day-of-year mean:
   for each target date D we average Tmax, Tmin, and precipitation from the
   same day-of-year (±3 days) across **all prior calendar years** in the
   observation record. This avoids leaking future information into the
   baseline.
4. **Errors** are computed per lead time (days 0–14 ahead): Foreca's point
   forecast is compared to the observation, and so is the climatology
   baseline, so we can read a skill score `1 − MAE_forecast / MAE_clim`.
5. **Probability intervals**: Foreca publishes two ranges per day in addition
   to the point forecast — a "50%" band and a "100%" band, for both Tmax and
   Tmin. We check empirically how often the observation actually fell inside
   each band.
6. **Precipitation** gets the same treatment: MAE of the point forecast vs
   MAE of climatology, rain/no-rain accuracy vs a climatological baseline,
   POD (hit rate), false-alarm ratio, CSI (threat score), and the coverage
   of the `[rl, rh]` range.

All HTTP responses are cached to `cache/` so reruns are free and we don't
hammer Wayback or Open-Meteo.

## What you get

Running `python3 foreca_15vrk.py` produces:

- `cache/forecasts.csv` — every parsed 15-day forecast, one row per
  (run_date, target_date) pair.
- `cache/observations.csv` — daily observed Tmax, Tmin, precipitation for
  Helsinki from 2010-01-01 to 2025-12-31.
- `cache/climatology.csv` — per-date expanding-window climatology.
- `cache/merged.csv` — all forecast rows joined to observations and
  climatology, with signed errors, interval-coverage flags, and
  rain/no-rain flags.
- `cache/summary_by_lead.csv`, `cache/interval_coverage.csv`,
  `cache/precipitation_by_lead.csv` — per-lead-time summary tables.
- `graphs/mae_vs_lead.png`, `graphs/skill_vs_lead.png`,
  `graphs/interval_coverage.png`, `graphs/interval_widths.png`,
  `graphs/precipitation.png` — plots.

## Findings

### Temperature point forecast

![MAE by lead time](graphs/mae_vs_lead.png)

- At lead 0 (nowcast) the forecast has MAE ≈ 1.1–1.3 °C, versus ≈ 3.1 °C
  for a day-of-year climatology — a 60–67% reduction in error.
- Skill decays roughly linearly with lead time.
- **The break-even point is about day 9–10.** From day 10 on, the 15-day
  forecast's MAE is slightly *worse* than just using the historical average.
- The last ~5 days of the 15-day ribbon carry essentially no information
  beyond the climatological average.

### Probability intervals — are they calibrated?

![Interval coverage](graphs/interval_coverage.png)

Short answer: **no — they are severely under-dispersive at short lead**.

- At lead 0 the "50% band" contains the observed Tmax only **18%** of the
  time; the "100% band" catches it **55%** of the time.
- Coverage creeps up with lead time and reaches ≈ 0.85–0.95 for the 100%
  band from lead 4 onwards — but only because the bands widen dramatically:
  the Tmax 100% band grows from ~2 °C wide at lead 0 to ~12 °C wide at
  lead 14 (see `graphs/interval_widths.png`).
- Treat the bands as **ensemble spread**, not as calibrated probability
  intervals. The "100%" label does not mean the truth will fall inside.

### Precipitation

![Precipitation](graphs/precipitation.png)

- **Useful out to ~4 days.** At leads 0–2, MAE is 0.8–1.3 mm versus
  2.4–3.9 mm for climatology (skill 0.66–0.68), and rain/no-rain accuracy
  is 79–87%.
- By lead 5 the mm-level skill has collapsed to ~0 (Foreca MAE 2.03 mm,
  climatology 2.07 mm).
- By lead 10 the forecast is *worse* than climatology: rain/no-rain
  accuracy drops to 45%, lower than the 49% you'd get by looking up the
  historical rain frequency for that day of year.
- The `[rl, rh]` range catches the observed precipitation ≈ 70–80% of the
  time at all lead lengths, so it is not a true full-range envelope.

### Rough useful-horizon cheat sheet

| Quantity | Useful horizon |
|---|---|
| Tmax / Tmin point forecast | ~9 days |
| Precipitation amount (mm) | ~4 days |
| Rain yes/no | ~2 days clearly, ~5 days weakly |
| Temperature bands as probability | Don't — they are ensemble spread, not calibrated intervals |

## Caveats

- **Small sample.** Only 71 parsed snapshots over 2016–2025 (Wayback doesn't
  crawl on a schedule), so per-lead n = 71 is on the low side. The numbers
  are directionally solid but noisy at the decimal level.
- **Observations are ERA5**, not the FMI Helsinki-Kaisaniemi station. City
  and reanalysis temperatures differ by a few tenths of a degree, which
  slightly inflates forecast MAE for free.
- **Snapshots are not a random sample** of Helsinki weather. The Wayback
  crawl is clumpy in time and may be biased toward newsworthy weather.
- **Only the point forecasts are compared for temperature.** The bands are
  only evaluated for coverage, not as a full probabilistic scoring rule.

## Running it

```bash
python3 foreca_15vrk.py
```

Requirements: `pandas`, `numpy`, `matplotlib`. Everything else is standard
library. First run fetches ~71 Wayback snapshots and one Open-Meteo query
serially with polite backoff; subsequent runs hit the disk cache and
complete in a few seconds.

## ML baseline — can a small model beat Foreca?

[ml_forecast.py](ml_forecast.py) is a follow-up that asks: what if instead
of comparing Foreca to climatology, we trained a simple gradient-boosting
model on past observations and compared **that** to Foreca?

Two heads, both `HistGradientBoostingRegressor` from scikit-learn:

- **Standalone** — trained on 2011–2020 ERA5 daily observations only,
  tested on 2021–2025. Features: Tmax/Tmin/precip lags 0–7 days, lagged
  anomalies vs. climatology, day-of-year sin/cos, daily Open-Meteo
  pressure / wind / RH / cloud / radiation, daily Tmax/Tmin/precip from
  Stockholm / Tallinn / St Petersburg, and daily NAO / AO indices from
  NOAA CPC. One model per (variable, lead) — 42 in total. Foreca is
  **not** an input.
- **MOS post-processor** — trained on the merged Foreca rows (~1000
  examples, run_date < 2023) with Foreca's own published forecast as an
  input feature on top of all the standalone features. Tested on
  snapshots with run_date ≥ 2023. One model per variable, with `lead`
  as a feature so a single model spans all leads.

### Standalone results (full 2021–2025 test set)

![ML standalone skill](graphs/ml_skill_vs_lead.png)

- **Lead 1**: Tmax MAE 1.55 °C, Tmin 1.66 °C, precip 2.29 mm — beats
  climatology by 46% / 49% / 10%.
- Skill decays smoothly to ≈ 0 around lead 7–8 for temperature, and to
  ≈ 0 by lead 2 for precipitation.
- Past lead 8 the model is essentially indistinguishable from
  climatology — adding lag features, NAO/AO, and nearby cities does not
  push the useful horizon meaningfully past where pure persistence dies.

### Standalone vs. Foreca on the snapshot subset

![ML vs Foreca temperature](graphs/ml_mae_vs_lead.png)

The same 32 Foreca snapshots in 2021–2025, head-to-head:

- **Foreca wins handily at leads 2–7** (it has access to global NWP).
- **ML matches Foreca at lead 1** for Tmax (1.41 vs 1.43 °C).
- **ML beats Foreca from lead ~8 onwards** for both Tmax and Tmin —
  exactly the regime where Foreca had already collapsed to climatology.
  This is consistent with the original finding that Foreca's last 5
  days carry no information beyond climatology.
- **Foreca wins on precipitation at every lead**. A point gradient-
  boosting model trained on lagged station data is not a competitive
  precipitation forecaster.

### MOS post-processor — can ML correct Foreca?

![MOS improvement](graphs/ml_mos_improvement.png)

Bars show MOS MAE *minus* raw Foreca MAE on the same rows; green = MOS
helps, red = MOS hurts.

- **Tmax / Tmin: clear win at most leads.** The largest gains are at
  leads 4–10, where MOS shaves 0.2–1.2 °C off Foreca's MAE. This is the
  classical operational story — gradient-boosting bias correction adds
  real skill to a physics-based forecast.
- **Precipitation: MOS mostly hurts.** With ~1000 training rows and a
  highly skewed target, the MOS model overcorrects Foreca's precip
  point forecast at most leads. This is a sample-size problem more than
  a method problem.
- **Caveat**: only 24 test snapshots, so per-lead numbers are noisy.
  Treat individual bars as suggestive, not significant.

### Headline takeaway

For a single-file ~600-line script with no NWP and no GPU:

| Question | Answer |
|---|---|
| Beat climatology at every lead? | **Tmax / Tmin yes, out to ~lead 7.** Precip barely. |
| Beat Foreca at long leads (≥10)? | **Yes, on the small snapshot subset** — ML's bias toward climatology *is* the right answer once Foreca's signal has decayed. |
| Improve Foreca at intermediate leads via MOS? | **Yes for temperature, no for precip.** Temperature MOS is the most useful artifact in this directory. |

### Running the ML baseline

```bash
python3 ml_forecast.py
```

Extra dependency: `scikit-learn`. First run fetches a few new Open-Meteo
columns + nearby-station archives + NOAA CPC daily NAO/AO files (all
cached to `cache/`). Subsequent runs are CPU-only and finish in a few
seconds.

## Live forecasting and scoring

Three additional scripts extend the study into the present day:

**`fetch_foreca_today.py`** — fetch today's Foreca forecast directly from
`foreca.fi` and append to `cache/forecasts.csv`. Run at most once per day.

**`mos_preview.py`** — show the upcoming 15 days with both the raw Foreca
forecast and the MOS-corrected version side-by-side. Saves
`graphs/mos_preview.png` and appends today's predictions to
`cache/mos_preview_history.csv` for later scoring.

**`score_predictions.py`** — once target dates have passed, join
`cache/mos_preview_history.csv` with `cache/observations.csv` to compute
who did better (Foreca, MOS, or climatology). Saves
`graphs/score_predictions.png`.

```bash
# Daily workflow
python3 fetch_foreca_today.py
python3 mos_preview.py

# After enough target dates have passed
python3 score_predictions.py

# Monthly: rebuild merged.csv with new snapshots + retrain models
rm cache/merged.csv
python3 foreca_15vrk.py
python3 ml_forecast.py
```

## Files

- [foreca_15vrk.py](foreca_15vrk.py) — historical Foreca vs. climatology pipeline.
- [ml_forecast.py](ml_forecast.py) — ML baseline (standalone + MOS evaluation).
- [fetch_foreca_today.py](fetch_foreca_today.py) — fetch today's Foreca forecast.
- [mos_preview.py](mos_preview.py) — upcoming 15-day Foreca + MOS preview.
- [score_predictions.py](score_predictions.py) — score past MOS predictions against observations.
- `cache/` — HTTP response cache plus generated CSV tables.
- `graphs/` — all generated plots.

## Data sources

- Internet Archive — Wayback Machine ([CDX API](https://archive.org/developers/wayback-cdx-server.html))
- [Open-Meteo historical weather API](https://open-meteo.com/en/docs/historical-weather-api) (ERA5)
- [Open-Meteo forecast API](https://open-meteo.com/en/docs) (recent bridge data)
- [NOAA CPC daily NAO/AO indices](https://www.cpc.ncep.noaa.gov/products/precip/CWlink/)
- Forecasts originate from [Foreca](https://www.foreca.fi/), recovered
  via the Wayback Machine and directly for current forecasts.
