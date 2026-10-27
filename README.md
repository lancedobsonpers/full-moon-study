# Does the full moon make people act up?

The code and data behind two pages on lancedobson.com:

- **The plan, published before any data was compared to the moon:** https://lancedobson.com/essays/full-moon-pre-registration/
- **The results:** https://lancedobson.com/essays/full-moon-results/

**Verdict:** not supported. Mood, assaults, police calls and ER visits showed no full-moon difference beyond chance, the direction flipped from year to year, and the full moon calendar did not help forecast any of them in held-out years. One secondary result survived correction in the opposite direction: about 1.5% fewer police calls around the full moon than around the new moon, mostly in New Orleans.

## Reproduce it

Requires Python 3.13 with `numpy`, `pandas`, `scipy`, `statsmodels`, `requests` and `skyfield`.

```
python scripts/pull_socrata.py       # Chicago, NYC, Seattle, New Orleans daily counts (aggregated server-side)
python scripts/ephemeris.py          # needs JPL de440s.bsp in raw/ephemeris/ (~32 MB, not committed)
python scripts/analyze.py            # pre-registered tests -> out/results.json
PYTHONPATH=scripts python scripts/explore_police.py   # exploratory, NOT pre-registered
PYTHONPATH=scripts python scripts/selftest.py         # method check on simulated data
```

## What's here

- **`raw/chicago_assault.csv`, `raw/nyc_assault.csv`** — assaults and batteries per day (all hours, and 18:00–05:59), by date of occurrence.
- **`raw/seattle_calls.csv`** — distinct calls per day where the call type is 911, other telephone or in person.
- **`raw/nola_calls.csv`** — distinct calls per day with `selfinitiated = 'N'`.
- **`raw/neiss_daily.csv`** — NEISS cases per treatment date, 2006–2025, counted from CPSC's archived annual TSV files (unweighted). CPSC blocks scripted downloads, so these were counted in a web browser; the `night` column is unused.
- **`raw/hedonometer_*.json`** — Hedonometer daily happiness (`en_all` is the primary series).
- **`raw/US_births_*.csv`** — FiveThirtyEight births by date (CDC 1994–2003, SSA 2004–2014).
- **`out/moon_phases.csv`, `out/moon_<zone>.csv`** — full and new moons, and nightly illumination, by local date.
- **`out/results.json`** — every pre-registered number. **`out/exploratory_police.csv`** — the exploratory split.

## Implementation details the plan didn't specify

- **Random windows** are placed anywhere in each lunar cycle except the last two days before the next full moon, so windows never overlap.
- **Holidays:** 16 US holidays, including Easter.
- **Forecast baseline:** the previous 28 days' mean plus weekday and holiday effects learned on the training years.
- **Brightness placebo:** illumination is rotated within each cycle by a random offset.
- **Hedonometer days** are assumed to be UTC.
- **Data quirks kept as-is:** New Orleans 2023-11-28 and 2023-11-29 are unusually low (likely a recording gap); NYC 2010-12-27 is the blizzard.
