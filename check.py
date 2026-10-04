"""Step 5: the project's check. Recompute with different code, in plain Python (csv, gzip, json, math and loops: no pandas, numpy, scipy or sklearn) from the stored per-month daily count files (results/daily/*.csv, results/meta/*.json) and the NOAA file in data/, and compare with results/:

    ./venv/bin/python check.py

the audit counts; the federal-holiday list (found by walking the calendar); the daily series built from the 37 monthly count files, the monthly and annual tables; the weather normals (1993-2022, +-7 days); every walk-forward forecast of the 24 test months for the three targets (seasonal naive, seasonal naive with level change,
and the 10 regression configurations, each fitted by normal equations solved by Gauss-Jordan elimination, 240 fits of up to 19 columns on up to 1060 days); the scores, ablation, weather-value, fold, segment, Diebold-Mariano, worst-day, wx_skill tables; the Newey-West sensitivity regressions on all 1096 days; and ols_predictions.csv (to its rounding of 0.05 trips).
The boosted forecasts (gbm_predictions.csv, scikit-learn) cannot be refitted without the library: they are checked by properties (every test day and target present, finite, positive, same count of values) and every score, ablation, segment, test and worst-day number that uses them is recomputed from the stored predictions.
Comparisons are at a relative tolerance of 1e-8 (counts exact).
"""
import csv
import glob
import gzip
import json
import math
from datetime import date, timedelta
from pathlib import Path

HERE = Path(__file__).parent
D, R = HERE / "data", HERE / "results"
START, TEST0, END = date(2023, 1, 1), date(2024, 1, 1), date(2025, 12, 31)
TARGETS = ("total", "member", "casual")
CFG = {"cal": "", "T": "T", "R": "R", "S": "S", "all": "TRS", "noT": "RS", "noR": "TS", "noS": "TR", "all_clim": "TRS", "all_week": "TRS"}
REGIME = {"all_clim": "clim", "all_week": "week"}
GROUP_COLS = {"T": [13, 14], "R": [15, 16], "S": [17, 18]}          # columns of the 19-column design (13 calendar + 6 weather)
n_ok = 0


def same(label, got, want, rel=1e-8):
    global n_ok
    n_ok += 1; want = float(want)
    assert abs(got - want) <= rel * abs(want) + 1e-9, f"{label}: recount {got} != results {want}"


def same_int(label, got, want):
    global n_ok
    n_ok += 1; assert int(got) == int(float(want)), f"{label}: recount {got} != results {want}"


def same_str(label, got, want):
    global n_ok
    n_ok += 1; assert str(got) == str(want), f"{label}: recount {got} != results {want}"


def table(name):
    return list(csv.DictReader(open(R / f"{name}.csv")))


def mean(v):
    return sum(v) / len(v)


def inverse(a):
    n = len(a); m = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(a)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(m[r][c])); m[c], m[p] = m[p], m[c]; d = m[c][c]; m[c] = [x / d for x in m[c]]
        for r in range(n):
            if r != c:
                f = m[r][c]
                if f != 0.0:
                    m[r] = [x - f * y for x, y in zip(m[r], m[c])]
    return [row[n:] for row in m]


def nth_weekday(y, m, wd, n):
    k = 0
    for day in range(1, 32):
        d = date(y, m, day)
        if d.weekday() == wd:
            k += 1
            if k == n:
                return d


def last_weekday(y, m, wd):
    best = None
    for day in range(1, 32):
        try:
            d = date(y, m, day)
        except ValueError:
            break
        if d.weekday() == wd:
            best = d
    return best


def holiday_rows():
    out = set()
    for y in (2023, 2024, 2025):
        for name, d in (("New Year's Day", date(y, 1, 1)), ("Juneteenth", date(y, 6, 19)), ("Independence Day", date(y, 7, 4)), ("Veterans Day", date(y, 11, 11)), ("Christmas Day", date(y, 12, 25))):
            out.add((d, name, "actual"))
            if d.weekday() == 5:
                out.add((date.fromordinal(d.toordinal() - 1), name + " (observed)", "observed"))
            if d.weekday() == 6:
                out.add((date.fromordinal(d.toordinal() + 1), name + " (observed)", "observed"))
        tg = nth_weekday(y, 11, 3, 4)
        for d, name in ((nth_weekday(y, 1, 0, 3), "Martin Luther King Jr. Day"), (nth_weekday(y, 2, 0, 3), "Presidents Day"), (last_weekday(y, 5, 0), "Memorial Day"), (nth_weekday(y, 9, 0, 1), "Labor Day"), (nth_weekday(y, 10, 0, 2), "Columbus Day"),
                        (tg, "Thanksgiving"), (date.fromordinal(tg.toordinal() + 1), "Day after Thanksgiving")):
            out.add((d, name, "actual"))
    return sorted(r for r in out if START <= r[0] <= END)


def iso(s):
    return date(int(s[:4]), int(s[5:7]), int(s[8:10]))


def metrics(a, p):
    n = len(a); e = [p[i] - a[i] for i in range(n)]
    return {"n": n, "mae": sum(abs(v) for v in e) / n, "rmse": math.sqrt(sum(v * v for v in e) / n), "mape": 100 * sum(abs(e[i]) / a[i] for i in range(n)) / n, "bias_pct": 100 * sum(e[i] / a[i] for i in range(n)) / n}


def dm(d, lags):
    n = len(d); m = mean(d); e = [v - m for v in d]; g = lambda l: sum(e[i] * e[i - l] for i in range(l, n)) / n
    var = (g(0) + 2 * sum((1 - l / (lags + 1)) * g(l) for l in range(1, lags + 1))) / n; t = m / math.sqrt(var); return t, math.erfc(abs(t) / math.sqrt(2))


def main():
    global n_ok
    HOL = holiday_rows(); hol = {h[0] for h in HOL}
    got = {(r["date"], r["name"], r["kind"]) for r in table("holidays")}
    assert got == {(d.isoformat(), n, k) for d, n, k in HOL}; same_int("holiday rows", len(got), len(HOL))
    # ---- the daily series from the per-month count files
    cols = ["trips", "member", "casual", "classic", "electric", "short", "over_day", "no_start_station"]; day = {}
    for f in sorted(glob.glob(str(R / "daily" / "*.csv"))):
        for r in csv.DictReader(open(f)):
            c = day.setdefault(r["date"], dict.fromkeys(cols, 0))
            for k in cols:
                c[k] += int(r[k])
    meta = [json.load(open(f)) for f in sorted(glob.glob(str(R / "meta" / "*.json")))]
    for m in meta:                                                                   # every month's daily files add up to its audited row count
        s = sum(int(r["trips"]) for r in csv.DictReader(open(R / "daily" / f"{m['month']}.csv"))); same_int("month rows " + m["month"], s + m["bad_started_at"], m["rows"]); same_int("line count " + m["month"], m["line_count_rows"], m["rows"])
    days = [START + timedelta(i) for i in range((END - START).days + 1)]; assert len(days) == 1096 and all(d.isoformat() in day for d in days)
    series = {k: [day[d.isoformat()][k] for d in days] for k in cols}
    # ---- weather
    el = {k: {} for k in ("TMAX", "TMIN", "PRCP", "SNOW", "SNWD")}; flagged_study = flagged_hist = 0
    for line in csv.reader(gzip.open(D / "USW00094728.csv.gz", "rt")):
        if line[2] in el:
            ds = line[1]
            if line[5] != "":
                if "20230101" <= ds <= "20251231": flagged_study += 1
                if "19930101" <= ds <= "20221231": flagged_hist += 1
                continue
            el[line[2]][ds] = float(line[3])
    key = lambda d: d.strftime("%Y%m%d")
    wx = {}
    for d in [date(1993, 1, 1) + timedelta(i) for i in range((END - date(1993, 1, 1)).days + 1)]:
        k = key(d)
        if k in el["TMAX"] and k in el["TMIN"] and k in el["PRCP"] and k in el["SNOW"]:
            wx[d] = (el["TMAX"][k] / 10, el["TMIN"][k] / 10, el["PRCP"][k] / 10, el["SNOW"][k], el["SNWD"].get(k, 0.0))
    assert all(d in wx for d in days)
    snwd_gap = [d for d in days if key(d) not in el["SNWD"]]
    # ---- audit
    A = {"monthly files counted": len(meta), "CSV files inside them": sum(m["csv_files"] for m in meta), "trip rows in all files": sum(m["rows"] for m in meta), "months whose header is not the standard 13 columns": sum(not m["header_is_standard"] for m in meta),
         "rows with duplicate ride_id": sum(m["rows"] - m["distinct_ride_ids"] for m in meta), "rows with unreadable start time": sum(m["bad_started_at"] for m in meta), "rows with unreadable end time": sum(m["bad_ended_at"] for m in meta),
         "rows with end before start": sum(m["negative_duration"] for m in meta), "rows lasting 24 hours or more": sum(c["over_day"] for c in day.values()), "rows under 60 seconds or without an end time": sum(c["short"] for c in day.values()),
         "rows started outside the file's month": sum(m["started_outside_month"] for m in meta), "rows with no start station": sum(c["no_start_station"] for c in day.values()), "distinct bike types in 2023-2025 files": len({k for m in meta for k in m["rideable_types"]}),
         "distinct rider types": len({k for m in meta for k in m["member_casual_values"]}), "months where started_at has a different length": sum(len(m["started_at_lengths"]) > 1 for m in meta), "trips 2023-01-01 to 2025-12-31": sum(series["trips"]), "days in the study window": len(days), "days with no count": 0,
         "days where member + casual differs from trips": sum(series["member"][i] + series["casual"][i] != series["trips"][i] for i in range(len(days))), "days where classic + electric differs from trips": sum(series["classic"][i] + series["electric"][i] != series["trips"][i] for i in range(len(days))),
         "test days": sum(d >= TEST0 for d in days), "test months": 24, "holiday dates in the test window": sum(1 for d in days if d >= TEST0 and d in hol), "storm days in the test window": sum(1 for d in days if d >= TEST0 and (wx[d][2] >= 25.4 or wx[d][3] >= 25.4)),
         "weather values missing in the study window (TMAX, TMIN, PRCP, SNOW)": 0, "snow-depth values missing in the study window": len(snwd_gap), "quality-flagged weather values in the study window": flagged_study, "quality-flagged values 1993-2022 left out of the normals": flagged_hist}
    tab = {r["check"]: r["value"] for r in table("audit")}; assert set(tab) == set(A), set(tab) ^ set(A)
    for k, v in A.items():
        same_int("audit " + k, v, tab[k])
    # ---- monthly and annual
    ym = lambda d: d.strftime("%Y-%m")
    for r in table("monthly"):
        idx = [i for i, d in enumerate(days) if ym(d) == r["ym"]]
        for k in ("trips", "member", "casual", "classic", "electric"):
            same_int(f"monthly {r['ym']} {k}", sum(series[k][i] for i in idx), r[k])
        same_int("days", len(idx), r["days"]); same("tavg", mean([(wx[days[i]][0] + wx[days[i]][1]) / 2 for i in idx]), r["tavg"]); same("prcp", sum(wx[days[i]][2] for i in idx), r["prcp_mm"]); same("snow", sum(wx[days[i]][3] for i in idx), r["snow_mm"])
        t = sum(series["trips"][i] for i in idx); same("per_day", t / len(idx), r["per_day"]); same("electric_share", sum(series["electric"][i] for i in idx) / t, r["electric_share"]); same("member_share", sum(series["member"][i] for i in idx) / t, r["member_share"])
    prev = None
    for r in table("annual"):
        y = int(r["yr"]); idx = [i for i, d in enumerate(days) if d.year == y]; t = sum(series["trips"][i] for i in idx)
        same_int("annual days", len(idx), r["days"]); same_int("annual trips", t, r["trips"]); same_int("annual member", sum(series["member"][i] for i in idx), r["member"]); same_int("annual electric", sum(series["electric"][i] for i in idx), r["electric"])
        same("annual per_day", t / len(idx), r["per_day"]); same("annual member share", sum(series["member"][i] for i in idx) / t, r["member_share"]); same("annual electric share", sum(series["electric"][i] for i in idx) / t, r["electric_share"])
        if prev is not None: same("annual growth", 100 * ((t / len(idx)) / prev - 1), r["growth_pct"])
        prev = t / len(idx)
    # ---- features
    def f6(a):
        t = (a[0] + a[1]) / 20 - 1.5
        return [t, t * t, math.log1p(a[2]), 1.0 if a[2] >= 1.0 else 0.0, math.log1p(a[3]), 1.0 if a[4] > 0 else 0.0]
    def calrow(d):
        x = 2 * math.pi * d.timetuple().tm_yday / 365.25; r = [1.0] + [1.0 if d.weekday() == w else 0.0 for w in range(1, 7)]
        return r + [1.0 if d in hol else 0.0, math.sin(x), math.cos(x), math.sin(2 * x), math.cos(2 * x), (d - START).days / 365.25 - 1.5]
    # normals: for each calendar day of a 365-day year (Feb 29 skipped), the +-7 day mean over 1993-2022 of the raw values and of the six terms
    def cal_key(d):
        return (date(2001, d.month, d.day) - date(2001, 1, 1)).days
    sr, cr, sf = [[0.0] * 5 for _ in range(365)], [0] * 365, [[0.0] * 6 for _ in range(365)]
    for d, a in wx.items():
        if d.year > 2022 or (d.month == 2 and d.day == 29):
            continue
        k = cal_key(d); cr[k] += 1; f = f6(a)
        for j in range(5): sr[k][j] += a[j]
        for j in range(6): sf[k][j] += f[j]
    norm_raw, norm_f = [], []
    for k in range(365):
        ks = [(k + o) % 365 for o in range(-7, 8)]; c = sum(cr[q] for q in ks)
        norm_raw.append([sum(sr[q][j] for q in ks) / c for j in range(5)]); norm_f.append([sum(sf[q][j] for q in ks) / c for j in range(6)])
    def normal(d, tabl):
        return tabl[cal_key(date(d.year, 2, 28) if (d.month == 2 and d.day == 29) else d)]
    # ---- walk-forward OLS, seasonal naive, climatology and last-week inputs
    test_days = [d for d in days if d >= TEST0]; idx_of = {d: i for i, d in enumerate(days)}
    full = [calrow(d) + f6(wx[d]) for d in days]; tcol = {"total": series["trips"], "member": series["member"], "casual": series["casual"]}
    models = ["sn", "snl"] + [f"ols_{c}" for c in CFG]; pred = {t: {m: {} for m in models} for t in TARGETS}; clim_raw, week_raw = {}, {}
    for (y, mth) in sorted({(d.year, d.month) for d in test_days}):
        te = [d for d in test_days if (d.year, d.month) == (y, mth)]; f0 = idx_of[te[0]]
        xtx = [[0.0] * 19 for _ in range(19)]; xty = {t: [0.0] * 19 for t in TARGETS}; ly = {t: [math.log(tcol[t][i]) for i in range(f0)] for t in TARGETS}
        for i in range(f0):
            xi = full[i]
            for a in range(19):
                xa = xi[a]; row = xtx[a]
                for b in range(a, 19): row[b] += xa * xi[b]
                for t in TARGETS: xty[t][a] += xa * ly[t][i]
        for a in range(19):
            for b in range(a): xtx[a][b] = xtx[b][a]
        wkf = [mean([f6(wx[days[i]])[j] for i in range(f0 - 7, f0)]) for j in range(6)]; wkr = [mean([wx[days[i]][j] for i in range(f0 - 7, f0)]) for j in range(5)]
        for d in te:
            clim_raw[d] = normal(d, norm_raw); week_raw[d] = wkr
        for t in TARGETS:
            a = tcol[t]
            for d in te:
                pred[t]["sn"][d] = float(a[idx_of[d] - 364])
            r = sum(a[f0 - 28:f0]) / sum(a[f0 - 28 - 364:f0 - 364]) if f0 - 28 - 364 >= 0 else 1.0
            for d in te:
                pred[t]["snl"][d] = a[idx_of[d] - 364] * r
        for cfg, groups in CFG.items():
            cs = list(range(13)) + [c for g in groups for c in GROUP_COLS[g]]; inv = inverse([[xtx[a][b] for b in cs] for a in cs])
            for t in TARGETS:
                rhs = [xty[t][a] for a in cs]; b = [sum(inv[p][q] * rhs[q] for q in range(len(cs))) for p in range(len(cs))]
                for d in te:
                    wxf = f6(wx[d]) if cfg not in REGIME else (normal(d, norm_f) if REGIME[cfg] == "clim" else wkf)
                    x = (calrow(d) + wxf); pred[t][f"ols_{cfg}"][d] = math.exp(sum(b[p] * x[cs[p]] for p in range(len(cs))))
    # ---- boosted predictions: properties
    gb = {t: {} for t in TARGETS}; gcols = [f"gbm_{c}" for c in CFG]
    rows = table("gbm_predictions"); same_int("gbm rows", len(rows), 3 * len(test_days))
    for r in rows:
        for c in gcols:
            v = float(r[c]); assert math.isfinite(v) and v > 0, (r["date"], c)
            gb[r["target"]].setdefault(c, {})[iso(r["date"])] = v
    for t in TARGETS:
        for c in gcols:
            assert set(gb[t][c]) == set(test_days); pred[t][c] = gb[t][c]
    models += gcols; same_int("models", len(models), 22)
    act = {t: [float(tcol[t][idx_of[d]]) for d in test_days] for t in TARGETS}; P = {t: {m: [pred[t][m][d] for d in test_days] for m in models} for t in TARGETS}
    # ---- ols_predictions.csv (rounded to 0.1)
    opr = table("ols_predictions"); same_int("ols pred rows", len(opr), 3 * len(test_days))
    for r in opr:
        d = iso(r["date"]); t = r["target"]; same_int("actual", tcol[t][idx_of[d]], r["actual"])
        for m in ["sn", "snl"] + [f"ols_{c}" for c in CFG]:
            assert abs(pred[t][m][d] - float(r[m])) <= 0.0501, (r["date"], t, m, pred[t][m][d], r[m])
    n_ok += len(opr) * 12
    # ---- scores
    yrs = [d.year for d in test_days]; masks = {"all": [True] * len(test_days), "2024": [y == 2024 for y in yrs], "2025": [y == 2025 for y in yrs]}
    sub = lambda v, mk: [x for x, k in zip(v, mk) if k]; S = {}
    for r in table("scores"):
        t, m, per = r["target"], r["model"], r["period"]; mt = metrics(sub(act[t], masks[per]), sub(P[t][m], masks[per])); S[(t, m, per)] = mt
        same_int("score n", mt["n"], r["n"])
        for k in ("mae", "rmse", "mape", "bias_pct"):
            same(f"score {t} {m} {per} {k}", mt[k], r[k])
    assert len(S) == 3 * 22 * 3
    for r in table("ablation"):
        fam, t, c = r["family"], r["target"], r["config"]; same(f"ablation {fam} {t} {c}", S[(t, f"{fam}_{c}", "all")]["mape"], r["mape"]); same(f"ablation change {fam} {t} {c}", S[(t, f"{fam}_{c}", "all")]["mape"] - S[(t, f"{fam}_cal", "all")]["mape"], r["mape_change_vs_calendar"])
    lab = {"none (calendar only)": "cal", "observed on the day (upper bound)": "all", "climatology (honest)": "all_clim", "last week held flat (honest)": "all_week"}
    for r in table("weather_value"):
        s_ = S[(r["target"], f"{r['family']}_{lab[r['weather_input']]}", "all")]; same("wv mape", s_["mape"], r["mape"]); same("wv mae", s_["mae"], r["mae"]); same("wv bias", s_["bias_pct"], r["bias_pct"])
    ymon = [d.strftime("%Y-%m") for d in test_days]
    for r in table("folds"):
        mk = [x == r["month"] for x in ymon]; same_int("fold days", sum(mk), r["days"]); same("fold actual", mean(sub(act["total"], mk)), r["actual_per_day"])
        for m in ("sn", "snl", "ols_cal", "ols_all", "ols_all_clim", "gbm_cal", "gbm_all", "gbm_all_clim"):
            same(f"fold {r['month']} {m}", metrics(sub(act["total"], mk), sub(P["total"][m], mk))["mape"], r[m])
    storm = {d for d in test_days if wx[d][2] >= 25.4 or wx[d][3] >= 25.4}; isH = [d in hol for d in test_days]; isS = [d in storm for d in test_days]
    seg = {"holiday": isH, "storm (>= 1 inch rain or snow)": [s and not h for s, h in zip(isS, isH)], "other days": [not s and not h for s, h in zip(isS, isH)], "all test days": [True] * len(test_days)}
    for r in table("segments"):
        mk = seg[r["segment"]]; mt = metrics(sub(act[r["target"]], mk), sub(P[r["target"]][r["model"]], mk)); same_int("seg n", mt["n"], r["n"])
        for k in ("mae", "rmse", "mape", "bias_pct"):
            same(f"segment {r['target']} {r['segment']} {r['model']} {k}", mt[k], r[k])
    ae = lambda m: [abs(P["total"][m][i] - act["total"][i]) for i in range(len(test_days))]
    for r in table("dm"):
        d_ = [x - y for x, y in zip(ae(r["reference"]), ae(r["candidate"]))]; t_, p_ = dm(d_, 30); same("dm gain", mean(d_), r["mean_abs_error_gain"]); same("dm t", t_, r["dm_t"], 1e-7); same("dm p", p_, r["dm_p"], 1e-6)
    e = [(P["total"]["ols_all_clim"][i] - act["total"][i]) / act["total"][i] for i in range(len(test_days))]; order = sorted(range(len(test_days)), key=lambda k: (-round(abs(e[k]), 9), k))[:12]
    wrows = table("worst_days"); assert len(wrows) == 12
    for r, k in zip(wrows, order):
        d = test_days[k]; same_str("worst date", d.isoformat(), r["date"]); same_str("worst weekday", d.strftime("%a"), r["weekday"]); same_int("worst actual", act["total"][k], r["actual"]); same("worst pred", P["total"]["ols_all_clim"][k], r["predicted_honest"]); same("worst err", 100 * e[k], r["error_pct_honest"])
        same("worst pred obs", P["total"]["ols_all"][k], r["predicted_observed_weather"]); same("worst err obs", 100 * (P["total"]["ols_all"][k] - act["total"][k]) / act["total"][k], r["error_pct_observed"]); same("worst tmax", wx[d][0], r["tmax"]); same("worst prcp", wx[d][2], r["prcp_mm"]); same("worst snow", wx[d][3], r["snow_mm"]); same_int("worst hol", d in hol, r["holiday"])
    # ---- weather skill
    for r in table("wx_skill"):
        f = {"mean temperature (C)": lambda a: (a[0] + a[1]) / 2, "rain (mm)": lambda a: a[2], "snow (mm)": lambda a: a[3]}[r["variable"]]
        o = [f(wx[d]) for d in test_days]; c_ = [f(clim_raw[d]) for d in test_days]; w_ = [f(week_raw[d]) for d in test_days]; mo = mean(o); ss = sum((v - mo) ** 2 for v in o)
        same("wx sd", math.sqrt(ss / len(o)), r["sd_observed"]); same("wx rmse clim", math.sqrt(sum((x - y) ** 2 for x, y in zip(c_, o)) / len(o)), r["rmse_climatology"]); same("wx rmse week", math.sqrt(sum((x - y) ** 2 for x, y in zip(w_, o)) / len(o)), r["rmse_last_week"])
        same("wx r2 clim", 1 - sum((x - y) ** 2 for x, y in zip(c_, o)) / ss, r["r2_climatology"]); same("wx r2 week", 1 - sum((x - y) ** 2 for x, y in zip(w_, o)) / ss, r["r2_last_week"])
    # ---- sensitivity: all 1096 days, 19 columns, Newey-West
    xs = lambda T: T / 10 - 1.5; con = {}
    for lo, hi in ((0, 10), (10, 20), (20, 30)):
        c = [0.0] * 19; c[13], c[14] = xs(hi) - xs(lo), xs(hi) ** 2 - xs(lo) ** 2; con[f"mean temperature {lo} to {hi} C"] = c
    for mm in (10, 25):
        c = [0.0] * 19; c[15], c[16] = math.log1p(mm), 1.0; con[f"rain day of {mm} mm vs dry"] = c
    c = [0.0] * 19; c[17], c[18] = math.log1p(25.4), 1.0; con["snow day of 25 mm, snow on the ground, vs none"] = c
    c = [0.0] * 19; c[5] = 1.0; con["Saturday vs Monday"] = c
    c = [0.0] * 19; c[7] = 1.0; con["federal holiday vs ordinary day"] = c
    n = len(days); xtx = [[sum(full[i][a] * full[i][b] for i in range(n)) for b in range(19)] for a in range(19)]; inv = inverse(xtx); sens = {(r["target"], r["effect"]): r for r in table("sensitivity")}; assert len(sens) == 3 * len(con)
    for t in TARGETS:
        y = [math.log(v) for v in tcol[t]]; b = [sum(inv[a][c_] * sum(full[i][c_] * y[i] for i in range(n)) for c_ in range(19)) for a in range(19)]; res = [y[i] - sum(full[i][a] * b[a] for a in range(19)) for i in range(n)]
        u = [[full[i][a] * res[i] for a in range(19)] for i in range(n)]; Sg = [[sum(u[i][a] * u[i][b_] for i in range(n)) for b_ in range(19)] for a in range(19)]
        for l in range(1, 15):
            w = 1 - l / 15
            for a in range(19):
                for b_ in range(19):
                    Sg[a][b_] += w * (sum(u[i][a] * u[i - l][b_] for i in range(l, n)) + sum(u[i][b_] * u[i - l][a] for i in range(l, n)))
        V = [[sum(inv[a][p] * Sg[p][q] * inv[q][b_] for p in range(19) for q in range(19)) * n / (n - 19) for b_ in range(19)] for a in range(19)]
        for name, c in con.items():
            est = sum(c[a] * b[a] for a in range(19)); se = math.sqrt(sum(c[a] * V[a][b_] * c[b_] for a in range(19) for b_ in range(19))); r = sens[(t, name)]
            same(f"sens {t} {name}", 100 * (math.exp(est) - 1), r["pct_change"], 1e-7); same("sens lo", 100 * (math.exp(est - 1.96 * se) - 1), r["ci_low_pct"], 1e-7); same("sens hi", 100 * (math.exp(est + 1.96 * se) - 1), r["ci_high_pct"], 1e-7)
    print(f"check.py: {n_ok} recomputed numbers agree with results/")


if __name__ == "__main__":
    main()
