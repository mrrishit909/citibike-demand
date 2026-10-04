"""Step 1: Citi Bike trips, counted per day as each monthly file is downloaded, plus NOAA daily weather for Central Park.

    ./venv/bin/python download.py   -> results/daily/YYYYMM.csv (37 per-month daily count files)  results/meta/YYYYMM.json  results/schema.csv  results/pull.json
                                       data/USW00094728.csv.gz  data/ghcnd-stations.txt

Citi Bike system data: https://s3.amazonaws.com/tripdata/index.html (2023 comes inside the yearly file 2023-citibike-tripdata.zip, 2024 and 2025 as monthly zips 2024MM-citibike-tripdata.zip, plus 202601 because the monthly file
holds the trips that END in that month, so the last trips of 31 December 2025 sit in it). Published by Lyft Bikes and Scooters, LLC under the Citi Bike Data License Agreement https://citibikenyc.com/data-sharing-policy :
a "non-exclusive, royalty-free, limited, perpetual license to access, reproduce, analyze, copy, modify, distribute in your product or service and use the Data for any lawful purpose", but you may not "(b) Host, stream, publish, distribute, sublicense, or sell
the Data as a stand-alone dataset; provided, however, you may include the Data as source material, as applicable, in analyses, reports, or studies published or distributed for non-commercial purposes" and not "(f) Attempt to correlate the Data with names, addresses, or other
information of customers or subscribers". So this repo publishes only per-day counts (trips, member, casual, classic, electric) and per-month audit counts. No trip row, ride_id, station, coordinate or rider detail is kept, and nothing about any rider is looked at. This project is not
affiliated with, endorsed by or sponsored by Citi Bike, Lyft or Citigroup.

Each month's big zip (0.35 to 1.6 GB, 20 GB in all) is downloaded, counted with DuckDB (two threads, 2 GB memory limit), cross-checked against a plain line count of the CSV files, and deleted. The counts are by the local date in started_at.

Weather: NOAA National Centers for Environmental Information, GHCN-Daily, New York Central Park (USW00094728), https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_station/USW00094728.csv.gz . US government data; citation requested by NOAA:
Menne, M.J., I. Durre, R.S. Vose, B.E. Gleason and T.G. Houston, 2012, "An overview of the Global Historical Climatology Network-Daily Database", Journal of Atmospheric and Oceanic Technology 29, 897-910, doi:10.1175/JTECH-D-11-00103.1; dataset doi:10.7289/V5D21VHZ.
The weather file (1.8 MB, whole station history since 1869, rebuilt daily by NOAA) is fetched and never committed; results/pull.json records its checksum.
"""
import hashlib
import json
import shutil
import ssl
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from datetime import date
from pathlib import Path

import certifi
import duckdb

HERE = Path(__file__).parent
D, R, W = HERE / "data", HERE / "results", HERE / "data" / "work"
S3 = "https://s3.amazonaws.com/tripdata/"
MONTHS = [f"{y}{m:02d}" for y in (2023, 2024, 2025) for m in range(1, 13)] + ["202601"]
WX = {"USW00094728.csv.gz": "https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_station/USW00094728.csv.gz", "ghcnd-stations.txt": "https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt"}
HEADER = "ride_id,rideable_type,started_at,ended_at,start_station_name,start_station_id,end_station_name,end_station_id,start_lat,start_lng,end_lat,end_lng,member_casual"
UA = {"User-Agent": "research-portfolio (Rishit Raj Mathur)"}
CTX = ssl.create_default_context(cafile=certifi.where())


def fetch(url, dest, tries=4):
    """stream url to dest (skipped if a complete file of the right size is already there); returns (bytes, sha256)"""
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, context=CTX, timeout=600) as r:
                size = int(r.headers.get("Content-Length", -1))
                if dest.exists() and dest.stat().st_size == size:
                    break
                part = dest.with_name(dest.name + ".part")
                with open(part, "wb") as f:
                    shutil.copyfileobj(r, f, 1 << 20)
                if size >= 0 and part.stat().st_size != size:
                    raise OSError("short read")
                part.rename(dest)
            break
        except Exception as e:
            print("retry", url, e, flush=True); time.sleep(5 * (k + 1))
    else:
        raise RuntimeError("download failed " + url)
    return sha256_of(dest)


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return path.stat().st_size, h.hexdigest()


def line_count(path):
    n = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            n += chunk.count(b"\n")
    return n


def count_month(month, csvs, zip_bytes, zip_sha):
    """one pass of DuckDB over the month's CSV files -> daily counts and audit counts; asserts agreement with a plain-Python line count"""
    heads = Counter(open(p, encoding="utf-8").readline().strip() for p in csvs)
    lines = sum(line_count(p) - 1 for p in csvs)                       # data rows, header excluded
    con = duckdb.connect(); con.execute("pragma memory_limit='2GB'; pragma threads=2")
    src = "read_csv(" + repr([str(p) for p in csvs]) + ", all_varchar=true, header=true, union_by_name=true)"
    dur = "date_diff('second', try_cast(started_at as timestamp), try_cast(ended_at as timestamp))"
    ok = "regexp_matches(started_at, '^[0-9]{4}-[0-9]{2}-[0-9]{2} ')"
    daily = con.execute(f"""select substr(started_at,1,10) d, count(*) trips, sum((member_casual='member')::int) member, sum((member_casual='casual')::int) casual,
        sum((rideable_type='classic_bike')::int) classic, sum((rideable_type='electric_bike')::int) electric,
        sum((coalesce({dur},0)<60)::int) short, sum((coalesce({dur},0)>86400)::int) over_day, sum((coalesce(start_station_id,'')='')::int) no_start_station
        from {src} where {ok} group by 1 order by 1""").fetchall()
    a = con.execute(f"""select count(*), count(distinct ride_id), sum((not {ok})::int), sum((try_cast(ended_at as timestamp) is null)::int), sum(({dur}<0)::int), min(started_at), max(started_at),
        sum((substr(started_at,1,7) <> '{month[:4]}-{month[4:]}')::int) from {src}""").fetchone()
    types = dict(con.execute(f"select rideable_type, count(*) from {src} group by 1 order by 1").fetchall()); users = dict(con.execute(f"select member_casual, count(*) from {src} group by 1 order by 1").fetchall())
    lens = dict(con.execute(f"select length(started_at), count(*) from {src} group by 1 order by 1").fetchall()); con.close()
    assert a[0] == lines == sum(r[1] for r in daily) + a[2], (month, a[0], lines)
    R_daily = R / "daily"; R_daily.mkdir(parents=True, exist_ok=True)
    with open(R_daily / f"{month}.csv", "w") as f:
        f.write("date,trips,member,casual,classic,electric,short,over_day,no_start_station\n")
        for r in daily:
            f.write(",".join(str(v) for v in r) + "\n")
    meta = {"month": month, "csv_files": len(csvs), "headers": sorted(heads), "header_is_standard": list(heads) == [HEADER], "rows": a[0], "line_count_rows": lines, "distinct_ride_ids": a[1], "bad_started_at": a[2], "bad_ended_at": a[3],
            "negative_duration": a[4], "first_start": a[5], "last_start": a[6], "started_outside_month": a[7], "rideable_types": types, "member_casual_values": users, "started_at_lengths": lens, "zip_bytes": zip_bytes, "zip_sha256": zip_sha}
    (R / "meta").mkdir(exist_ok=True); (R / "meta" / f"{month}.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(month, a[0], "rows", len(daily), "days", flush=True)


def unzip_csvs(zpath, out):
    out.mkdir(exist_ok=True); paths = []
    with zipfile.ZipFile(zpath) as z:
        for n in z.namelist():
            if n.endswith(".csv") and not n.startswith("__MACOSX"):
                z.extract(n, out); paths.append(out / n)
    return sorted(paths)


def main():
    D.mkdir(exist_ok=True); W.mkdir(exist_ok=True); R.mkdir(exist_ok=True)
    info = {"pulled": date.today().isoformat(), "files": {}}
    for name, url in WX.items():
        n, h = fetch(url, D / name); info["files"][name] = {"url": url, "bytes": n, "sha256": h}
    y23 = None; pool = ThreadPoolExecutor(max_workers=10)          # one connection delivers only about 1 MB/s here, so the 2024-2026 zips are fetched ten at a time and counted one at a time
    todo = [m for m in MONTHS if not (R / "meta" / f"{m}.json").exists()]
    futs = {m: pool.submit(fetch, S3 + f"{m}-citibike-tripdata.zip", W / f"{m}-citibike-tripdata.zip") for m in todo if not m.startswith("2023")}
    for month in todo:
        out = W / "csv"; shutil.rmtree(out, ignore_errors=True)
        if month.startswith("2023"):
            if y23 is None:
                y23 = W / "2023-citibike-tripdata.zip"; y23_bytes, y23_sha = fetch(S3 + "2023-citibike-tripdata.zip", y23)
            inner = W / f"{month}-citibike-tripdata.zip"
            with zipfile.ZipFile(y23) as z, open(inner, "wb") as f:
                shutil.copyfileobj(z.open(f"2023-citibike-tripdata/{month}-citibike-tripdata.zip"), f, 1 << 20)
            zb, zs = sha256_of(inner)
        else:
            inner = W / f"{month}-citibike-tripdata.zip"; zb, zs = futs[month].result()
        count_month(month, unzip_csvs(inner, out), zb, zs)
        inner.unlink(); shutil.rmtree(out, ignore_errors=True)
    if y23 is not None and y23.exists():
        y23.unlink()
    metas = [json.loads((R / "meta" / f"{m}.json").read_text()) for m in MONTHS]
    for m in metas:
        info["files"][m["month"] + "-citibike-tripdata.zip"] = {"url": S3 + (("2023-citibike-tripdata.zip (inside it: " + m["month"] + "-citibike-tripdata.zip)") if m["month"].startswith("2023") else m["month"] + "-citibike-tripdata.zip"), "bytes": m["zip_bytes"], "sha256": m["zip_sha256"]}
    (R / "pull.json").write_text(json.dumps(info, indent=2) + "\n")
    import csv
    with open(R / "schema.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["month", "csv_files", "header_is_standard", "rows", "distinct_ride_ids", "bad_started_at", "bad_ended_at", "negative_duration", "started_outside_month", "rideable_types", "member_casual_values", "started_at_lengths", "first_start", "last_start"])
        for m in metas:
            w.writerow([m["month"], m["csv_files"], int(m["header_is_standard"]), m["rows"], m["distinct_ride_ids"], m["bad_started_at"], m["bad_ended_at"], m["negative_duration"], m["started_outside_month"],
                        "|".join(f"{k}:{v}" for k, v in m["rideable_types"].items()), "|".join(f"{k}:{v}" for k, v in m["member_casual_values"].items()), "|".join(f"{k}:{v}" for k, v in m["started_at_lengths"].items()), m["first_start"], m["last_start"]])
    print("done", len(metas), "months")


if __name__ == "__main__":
    main()
