"""Pull daily counts (all hours, and night only 18:00-05:59) from city open-data portals.

Counts are aggregated server-side by Socrata, so no row-level data is downloaded.
Output: raw/<source>.csv with columns date,all,night (local calendar date of the event).
"""
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests

RAW = Path(__file__).resolve().parent.parent / "raw"

NOLA_IDS = {
    2014: "jsyu-nz5r", 2015: "w68y-xmk6", 2016: "wgrp-d3ma", 2017: "bqmt-f3jk",
    2018: "9san-ivhk", 2019: "qf6q-pp4b", 2020: "hp7u-i9hf", 2021: "3pha-hum9",
    2022: "nci8-thrr", 2023: "pc5d-tvaw", 2024: "2zcj-b6ts", 2025: "4xwx-sfte",
    2026: "es9j-6y5d",
}
SEATTLE_PUBLIC = "('911','TELEPHONE OTHER, NOT 911','IN PERSON COMPLAINT')"


def soql(url, select, where, group):
    for attempt in range(5):
        r = requests.get(url, params={"$select": select, "$where": where, "$group": group,
                                      "$limit": 100000}, timeout=600)
        if r.ok:
            return r.json()
        time.sleep(10 * (attempt + 1))
    r.raise_for_status()


def daily(url, ts, where, count="count(*)", night_where=None):
    """Return {date: [all, night]} for a timestamp field `ts`."""
    out = defaultdict(lambda: [0, 0])
    night_where = night_where or f"(date_extract_hh({ts}) >= 18 or date_extract_hh({ts}) < 6)"
    for i, extra in enumerate([None, night_where]):
        w = where if extra is None else f"({where}) and {extra}"
        for row in soql(url, f"date_trunc_ymd({ts}) as d, {count} as n", w, "d"):
            if row.get("d"):
                out[row["d"][:10]][i] += int(row["n"])
    return out


def write(name, data):
    RAW.mkdir(exist_ok=True)
    with open(RAW / f"{name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "all", "night"])
        for d in sorted(data):
            w.writerow([d, *data[d]])
    print(name, len(data), "days", file=sys.stderr)


def chicago():
    url = "https://data.cityofchicago.org/resource/ijzp-q8t2.json"
    data = {}
    for y in range(2001, 2027):  # year by year keeps each query small
        end = "2026-09-30T23:59:59" if y == 2026 else f"{y}-12-31T23:59:59"
        data.update(daily(url, "date",
                          f"primary_type in('ASSAULT','BATTERY') and date between '{y}-01-01' and '{end}'"))
    write("chicago_assault", data)


def nyc():
    data = defaultdict(lambda: [0, 0])
    offs = "ofns_desc in('FELONY ASSAULT','ASSAULT 3 & RELATED OFFENSES')"
    night = "(cmplnt_fr_tm >= '18:00:00' or cmplnt_fr_tm < '06:00:00')"
    for ds in ["qgea-i56i", "5uac-w243"]:
        url = f"https://data.cityofnewyork.us/resource/{ds}.json"
        for y in range(2006, 2027):
            end = "2026-03-31T23:59:59" if y == 2026 else f"{y}-12-31T23:59:59"
            part = daily(url, "cmplnt_fr_dt",
                         f"{offs} and cmplnt_fr_dt between '{y}-01-01' and '{end}'", night_where=night)
            for d, v in part.items():
                data[d][0] += v[0]
                data[d][1] += v[1]
    write("nyc_assault", data)


def seattle():
    url = "https://data.seattle.gov/resource/33kz-ixgy.json"
    ts = "cad_event_original_time_queued"
    data = {}
    for y in range(2010, 2027):
        end = "2026-09-30T23:59:59" if y == 2026 else f"{y}-12-31T23:59:59"
        data.update(daily(url, ts, f"call_type in {SEATTLE_PUBLIC} and {ts} between '{y}-01-01' and '{end}'",
                          count="count(distinct cad_event_number)"))
    write("seattle_calls", data)


def nola():
    data = {}
    for y, ds in NOLA_IDS.items():
        url = f"https://data.nola.gov/resource/{ds}.json"
        end = "2026-09-30T23:59:59" if y == 2026 else f"{y}-12-31T23:59:59"
        # Each file is restricted to its own year, which drops the 2019/2020 overlap.
        # From 2025 the portal stores timecreate as text, so it is cast.
        ts = "timecreate::floating_timestamp" if y >= 2025 else "timecreate"
        data.update(daily(url, ts,
                          f"selfinitiated = 'N' and {ts} between '{y}-01-01' and '{end}'",
                          count="count(distinct nopd_item)"))
    write("nola_calls", data)


if __name__ == "__main__":
    for name in sys.argv[1:] or ["chicago", "nyc", "seattle", "nola"]:
        globals()[name]()
