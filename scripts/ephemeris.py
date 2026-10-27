"""Full and new moon instants from JPL DE440s, and nightly illumination, for each time zone used.
Writes out/moon_phases.csv (one row per full/new moon, UTC) and out/moon_<zone>.csv (one row per local date)."""
from zoneinfo import ZoneInfo
import numpy as np, pandas as pd
from skyfield import almanac
from skyfield.api import Loader
load = Loader("raw/ephemeris")
eph = load("de440s.bsp"); ts = load.timescale()
ZONES = {"utc": "UTC", "central": "America/Chicago", "eastern": "America/New_York", "pacific": "America/Los_Angeles"}
SIGNS = ["Aries","Taurus","Gemini","Cancer","Leo","Virgo","Libra","Scorpio","Sagittarius","Capricorn","Aquarius","Pisces"]

t, phase = almanac.find_discrete(ts.utc(1993, 6, 1), ts.utc(2027, 1, 1), almanac.moon_phases(eph))
from skyfield.framelib import ecliptic_frame
moon_lon = eph["earth"].at(t).observe(eph["moon"]).apparent().frame_latlon(ecliptic_frame)[1].degrees
ph = pd.DataFrame({"utc": pd.to_datetime(t.utc_iso()), "phase": phase, "moon_sign": [SIGNS[int(l // 30)] for l in moon_lon]})
ph = ph[ph.phase.isin([0, 2])].assign(kind=lambda x: np.where(x.phase == 2, "full", "new")).drop(columns="phase")
ph.to_csv("out/moon_phases.csv", index=False)

days = pd.date_range("1994-01-01", "2026-12-31", freq="D")
for key, zone in ZONES.items():
    z = ZoneInfo(zone)
    cal = pd.DataFrame({"date": days.date})
    for kind in ("full", "new"):
        local = ph[ph.kind == kind].utc.dt.tz_convert(z).dt.date
        cal[f"{kind}_day"] = cal.date.isin(set(local))
    # Illumination at local midnight that ends each date's evening, i.e. the night of that date.
    mid = [pd.Timestamp(d, tz=z) + pd.Timedelta(days=1) for d in days]
    tt = ts.from_datetimes([m.to_pydatetime() for m in mid])
    cal["illumination"] = almanac.fraction_illuminated(eph, "moon", tt)
    cal.to_csv(f"out/moon_{key}.csv", index=False)
    print(key, "full days", int(cal.full_day.sum()), "new days", int(cal.new_day.sum()))
print(ph[ph.utc.dt.year == 2024].head(4).to_string())
