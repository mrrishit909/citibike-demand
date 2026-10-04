"""Steps 2-4: audit the counts, then forecast New York's daily Citi Bike ridership month by month, from the past only, and ask what weather adds to the calendar.

    ./venv/bin/python analyze.py
        -> results/audit.csv  holidays.csv  monthly.csv  annual.csv  scores.csv  folds.csv  ablation.csv  weather_value.csv  segments.csv  dm.csv  worst_days.csv  sensitivity.csv  wx_skill.csv  gbm_predictions.csv  ols_predictions.csv

Data: trips per day (all rows in the monthly Citi Bike files, by local start date; total, member, casual), 2023-01-01 to 2025-12-31 (1096 days), and daily TMAX, TMIN, PRCP, SNOW, SNWD at Central Park (GHCN-Daily USW00094728; 1993 to 2022 only for the climatology).

Walk-forward by month: for each of the 24 test months (2024-01 .. 2025-12) every model is fitted on all days before the month starts (an expanding window, 365 days for the first month) and predicts every day of the month, 1 to 31 days ahead.
Targets are fitted in logs: total, member and casual trips. Models:
  sn    seasonal naive: the same weekday one year earlier (day - 364)
  snl   seasonal naive times the level change, last 28 days before the month over the same 28 days a year earlier (the fix for system growth)
  ols_* least squares on the calendar (weekday, holiday, two yearly harmonics, linear trend) plus weather terms: temperature (mean of TMAX and TMIN, and its square), rain (log(1+mm), wet day >= 1 mm), snow (log(1+mm), snow on the ground)
  gbm_* sklearn GradientBoostingRegressor (300 trees, depth 3, learning rate 0.05, subsample 0.8, seed 0, no tuning) on weekday, holiday, day of year and the raw weather values (TMAX, TMIN / PRCP / SNOW, SNWD). Trees cannot extrapolate growth, so it is fitted to
        the log trips minus the calendar regression's trend and the trend is added back.
Weather configurations (the ablation): cal (none), T, R, S (one group added), all, noT, noR, noS (one group dropped from all). What the weather input is when predicting is the whole point:
  obs   observed weather on the day (the upper bound: nobody knows next month's rain today), cfg names cal, T, R, S, all, noT, noR, noS
  clim  all_clim: the 1993-2022 normal for that calendar day (mean over +-7 days), declared in advance as THE honest case
  week  all_week: the mean of the 7 days before the month started, held flat (persistence)
The models are always trained on observed weather; only the weather fed in at prediction time changes (plug-in). Errors: MAE, RMSE, MAPE and mean signed % error over the 731 test days. Diebold-Mariano on the daily absolute errors with 30 Bartlett lags (forecasts are up to 31 days ahead).
Segments: holiday days (federal holidays, the observed day if it fell on a weekend, and the day after Thanksgiving), storm days (PRCP >= 25.4 mm or SNOW >= 25.4 mm), and the rest. Sensitivity: the all-weather regression on all 1096 days (descriptive, in sample) with Newey-West 14-lag errors.
"""
import csv
import gzip
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor

HERE = Path(__file__).parent
D, R = HERE / "data", HERE / "results"
START, TEST0, END = date(2023, 1, 1), date(2024, 1, 1), date(2025, 12, 31)
TARGETS = ("total", "member", "casual")
CFG = {"cal": "", "T": "T", "R": "R", "S": "S", "all": "TRS", "noT": "RS", "noR": "TS", "noS": "TR", "all_clim": "TRS", "all_week": "TRS"}
INFO = {"all_clim": "clim", "all_week": "week"}
OLS_G, RAW_G = {"T": [0, 1], "R": [2, 3], "S": [4, 5]}, {"T": [0, 1], "R": [2], "S": [3, 4]}
GBM_PARAMS = dict(n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8, min_samples_leaf=5, random_state=0)
MODELS = ["sn", "snl"] + [f"ols_{c}" for c in CFG] + [f"gbm_{c}" for c in CFG]
STORM_MM, DM_LAGS, NW_LAGS = 25.4, 30, 14


def nth(y, m, wd, n):
    d = date(y, m, 1); d += timedelta((wd - d.weekday()) % 7); return d + timedelta(7 * (n - 1))


def last_wd(y, m, wd):
    d = date(y, m + 1, 1) - timedelta(1) if m < 12 else date(y, 12, 31); return d - timedelta((d.weekday() - wd) % 7)


def holidays():
    rows = []
    for y in (2023, 2024, 2025):
        for name, d in (("New Year's Day", date(y, 1, 1)), ("Juneteenth", date(y, 6, 19)), ("Independence Day", date(y, 7, 4)), ("Veterans Day", date(y, 11, 11)), ("Christmas Day", date(y, 12, 25))):
            rows.append((d, name, "actual"))
            if d.weekday() == 5: rows.append((d - timedelta(1), name + " (observed)", "observed"))
            if d.weekday() == 6: rows.append((d + timedelta(1), name + " (observed)", "observed"))
        tg = nth(y, 11, 3, 4)
        rows += [(nth(y, 1, 0, 3), "Martin Luther King Jr. Day", "actual"), (nth(y, 2, 0, 3), "Presidents Day", "actual"), (last_wd(y, 5, 0), "Memorial Day", "actual"), (nth(y, 9, 0, 1), "Labor Day", "actual"),
                 (nth(y, 10, 0, 2), "Columbus Day", "actual"), (tg, "Thanksgiving", "actual"), (tg + timedelta(1), "Day after Thanksgiving", "actual")]
    return sorted(set(r for r in rows if START <= r[0] <= END))


def load_trips():
    parts = [pd.read_csv(f) for f in sorted((R / "daily").glob("*.csv"))]
    d = pd.concat(parts).groupby("date").sum().sort_index(); d.index = pd.to_datetime(d.index).date
    return d


def load_weather():
    w = pd.read_csv(D / "USW00094728.csv.gz", header=None, names=["id", "date", "el", "v", "m", "q", "s", "t"], dtype={"date": str, "q": str, "m": str}); w = w[w["el"].isin(["TMAX", "TMIN", "PRCP", "SNOW", "SNWD"])]
    flagged = w[w["q"].notna()]; w = w[w["q"].isna()]
    p = w.pivot(index="date", columns="el", values="v"); p.index = pd.to_datetime(p.index).date
    wx = pd.DataFrame({"tmax": p["TMAX"] / 10, "tmin": p["TMIN"] / 10, "prcp": p["PRCP"] / 10, "snow": p["SNOW"].astype(float), "snwd_raw": p["SNWD"].astype(float)})
    wx["snwd"] = wx["snwd_raw"].fillna(0.0)
    return wx, flagged


def feats_ols(a):
    """a: n x 5 raw (tmax, tmin, prcp, snow, snwd) -> n x 6 regression terms (temperature centred at 15 C and in units of 10 C)"""
    t = (a[:, 0] + a[:, 1]) / 20 - 1.5
    return np.c_[t, t * t, np.log1p(a[:, 2]), (a[:, 2] >= 1.0).astype(float), np.log1p(a[:, 3]), (a[:, 4] > 0).astype(float)]


def cal_design(days):
    hol = {h[0] for h in HOL}; n = len(days); X = np.zeros((n, 13)); X[:, 0] = 1
    for i, d in enumerate(days):
        if d.weekday() > 0: X[i, d.weekday()] = 1
        X[i, 7] = d in hol; x = 2 * np.pi * d.timetuple().tm_yday / 365.25
        X[i, 8:12] = np.sin(x), np.cos(x), np.sin(2 * x), np.cos(2 * x); X[i, 12] = (d - START).days / 365.25 - 1.5
    return X


def gbm_cal(days):
    hol = {h[0] for h in HOL}; return np.array([[d.weekday(), float(d in hol), d.timetuple().tm_yday] for d in days])


def ols_fit(X, y):
    return np.linalg.lstsq(X, y, rcond=None)[0]


def climatology(wx):
    """normal for each of the 365 calendar days (Feb 29 skipped): mean over 1993-2022 and over +-7 days, of the raw values and of the regression terms"""
    h = wx.loc[date(1993, 1, 1):date(2022, 12, 31)]; h = h[h[["tmax", "tmin", "prcp", "snow"]].notna().all(axis=1)]
    raw = h[["tmax", "tmin", "prcp", "snow", "snwd"]].to_numpy(); fo = feats_ols(raw); key = np.array([date(2001, d.month, d.day).timetuple().tm_yday - 1 if not (d.month == 2 and d.day == 29) else -1 for d in h.index])
    out = []
    for arr in (raw, fo):
        s, c = np.zeros((365, arr.shape[1])), np.zeros(365)
        for k in range(365):
            m = key == k; s[k] = arr[m].sum(axis=0); c[k] = m.sum()
        tab = np.array([sum(s[(k + o) % 365] for o in range(-7, 8)) / sum(c[(k + o) % 365] for o in range(-7, 8)) for k in range(365)]); out.append(tab)
    return out


def clim_rows(tab, days):
    return np.array([tab[date(2001, d.month, 28 if (d.month == 2 and d.day == 29) else d.day).timetuple().tm_yday - 1] for d in days])


def score(a, p):
    a, p = np.asarray(a, float), np.asarray(p, float); e = p - a
    return {"n": len(a), "mae": float(np.abs(e).mean()), "rmse": float(np.sqrt((e ** 2).mean())), "mape": float(100 * (np.abs(e) / a).mean()), "bias_pct": float(100 * (e / a).mean())}


def dm(d, lags):
    n = len(d); e = d - d.mean(); g = lambda l: float((e[l:] * e[:n - l]).sum() / n); var = (g(0) + 2 * sum((1 - l / (lags + 1)) * g(l) for l in range(1, lags + 1))) / n; t = d.mean() / np.sqrt(var)
    return float(t), float(2 * stats.norm.sf(abs(t)))


def hac(Xa, e, inv, lags):
    n, k = Xa.shape; u = Xa * e[:, None]; S = u.T @ u
    for l in range(1, lags + 1):
        G = u[l:].T @ u[:-l]; S += (1 - l / (lags + 1)) * (G + G.T)
    return inv @ S @ inv * n / (n - k)


def main():
    global HOL
    R.mkdir(exist_ok=True); HOL = holidays()
    pd.DataFrame([(d.isoformat(), n, k) for d, n, k in HOL], columns=["date", "name", "kind"]).to_csv(R / "holidays.csv", index=False)
    trips = load_trips(); meta = [json.loads(p.read_text()) for p in sorted((R / "meta").glob("*.json"))]
    days_all = [START + timedelta(i) for i in range((END - START).days + 1)]
    tr = trips.reindex(days_all); assert tr["trips"].notna().all() and len(days_all) == 1096
    wx, flagged = load_weather(); W = wx.reindex(days_all)
    fl = flagged[(flagged["date"] >= "20230101") & (flagged["date"] <= "20251231")]; hist_fl = flagged[(flagged["date"] >= "19930101") & (flagged["date"] <= "20221231") & flagged["el"].isin(["TMAX", "TMIN", "PRCP", "SNOW", "SNWD"])]
    wx_miss = {c: int(W[c].isna().sum()) for c in ("tmax", "tmin", "prcp", "snow", "snwd_raw")}; assert wx_miss["tmax"] == wx_miss["tmin"] == wx_miss["prcp"] == wx_miss["snow"] == 0
    snwd_gap = [d.isoformat() for d in days_all if np.isnan(W.loc[d, "snwd_raw"])]
    test_days = [d for d in days_all if d >= TEST0]; tot = tr["trips"].to_numpy()
    storm = {d for d in test_days if W.loc[d, "prcp"] >= STORM_MM or W.loc[d, "snow"] >= STORM_MM}; hol_set = {h[0] for h in HOL}
    # ---- audit
    rows_all = sum(m["rows"] for m in meta); inwin = int(tr["trips"].sum())
    audit = [("monthly files counted", len(meta)), ("CSV files inside them", sum(m["csv_files"] for m in meta)), ("trip rows in all files", rows_all), ("months whose header is not the standard 13 columns", sum(not m["header_is_standard"] for m in meta)),
             ("rows with duplicate ride_id", rows_all - sum(m["distinct_ride_ids"] for m in meta)), ("rows with unreadable start time", sum(m["bad_started_at"] for m in meta)), ("rows with unreadable end time", sum(m["bad_ended_at"] for m in meta)),
             ("rows with end before start", sum(m["negative_duration"] for m in meta)), ("rows lasting 24 hours or more", int(sum(pd.read_csv(f)["over_day"].sum() for f in (R / "daily").glob("*.csv")))), ("rows under 60 seconds or without an end time", int(sum(pd.read_csv(f)["short"].sum() for f in (R / "daily").glob("*.csv")))),
             ("rows started outside the file's month", sum(m["started_outside_month"] for m in meta)), ("rows with no start station", int(sum(pd.read_csv(f)["no_start_station"].sum() for f in (R / "daily").glob("*.csv")))),
             ("distinct bike types in 2023-2025 files", len({k for m in meta for k in m["rideable_types"]})), ("distinct rider types", len({k for m in meta for k in m["member_casual_values"]})),
             ("months where started_at has a different length", sum(len(m["started_at_lengths"]) > 1 for m in meta)),
             ("trips 2023-01-01 to 2025-12-31", inwin), ("days in the study window", len(days_all)), ("days with no count", 0), ("days where member + casual differs from trips", int(((tr["member"] + tr["casual"]) != tr["trips"]).sum())), ("days where classic + electric differs from trips", int(((tr["classic"] + tr["electric"]) != tr["trips"]).sum())),
             ("test days", len(test_days)), ("test months", 24), ("holiday dates in the test window", len(hol_set & set(test_days))), ("storm days in the test window", len(storm)),
             ("weather values missing in the study window (TMAX, TMIN, PRCP, SNOW)", 0), ("snow-depth values missing in the study window", wx_miss["snwd_raw"]), ("quality-flagged weather values in the study window", len(fl)), ("quality-flagged values 1993-2022 left out of the normals", len(hist_fl))]
    pd.DataFrame(audit, columns=["check", "value"]).to_csv(R / "audit.csv", index=False)
    # ---- monthly and annual tables
    df = tr.copy(); df["tavg"] = (W["tmax"] + W["tmin"]) / 2; df["prcp"] = W["prcp"]; df["snow"] = W["snow"]; df["ym"] = [d.strftime("%Y-%m") for d in df.index]; df["yr"] = [d.year for d in df.index]
    mo = df.groupby("ym").agg(days=("trips", "size"), trips=("trips", "sum"), member=("member", "sum"), casual=("casual", "sum"), classic=("classic", "sum"), electric=("electric", "sum"), tavg=("tavg", "mean"), prcp_mm=("prcp", "sum"), snow_mm=("snow", "sum")).reset_index()
    mo["per_day"] = mo["trips"] / mo["days"]; mo["electric_share"] = mo["electric"] / mo["trips"]; mo["member_share"] = mo["member"] / mo["trips"]; mo.to_csv(R / "monthly.csv", index=False)
    an = df.groupby("yr").agg(days=("trips", "size"), trips=("trips", "sum"), member=("member", "sum"), electric=("electric", "sum")).reset_index(); an["per_day"] = an["trips"] / an["days"]; an["member_share"] = an["member"] / an["trips"]; an["electric_share"] = an["electric"] / an["trips"]; an["growth_pct"] = 100 * (an["per_day"] / an["per_day"].shift(1) - 1)
    an.to_csv(R / "annual.csv", index=False)
    # ---- walk-forward
    rawW = W[["tmax", "tmin", "prcp", "snow", "snwd"]].to_numpy(); olsW = feats_ols(rawW); calX, gbmX = cal_design(days_all), gbm_cal(days_all); ctab_raw, ctab_ols = climatology(wx)
    idx = {d: i for i, d in enumerate(days_all)}; yrs_all = np.array([(d - START).days for d in days_all]) / 365.25; CLIM_RAW, WEEK_RAW = np.zeros((len(days_all), 5)), np.zeros((len(days_all), 5)); tcol = {"total": tot, "member": tr["member"].to_numpy(float), "casual": tr["casual"].to_numpy(float)}
    months = sorted({(d.year, d.month) for d in test_days}); pred = {t: {m: np.full(len(days_all), np.nan) for m in MODELS} for t in TARGETS}
    for (y, m) in months:
        te = [d for d in test_days if (d.year, d.month) == (y, m)]; f0 = idx[te[0]]; tei = np.array([idx[d] for d in te]); tri = np.arange(f0)
        wk = slice(f0 - 7, f0); info = {"obs": (olsW[tei], rawW[tei]), "clim": (clim_rows(ctab_ols, te), clim_rows(ctab_raw, te)),
                                        "week": (np.tile(olsW[wk].mean(axis=0), (len(te), 1)), np.tile(rawW[wk].mean(axis=0), (len(te), 1)))}
        CLIM_RAW[tei], WEEK_RAW[tei] = info["clim"][1], info["week"][1]
        for t in TARGETS:
            a = tcol[t]; ly = np.log(a[tri])
            for i in tei:
                pred[t]["sn"][i] = a[i - 364]
            r = a[f0 - 28:f0].sum() / a[f0 - 28 - 364:f0 - 364].sum() if f0 - 28 - 364 >= 0 else 1.0       # January 2024: the year-earlier window (December 2022) is not in the data, so no level change is applied
            for i in tei:
                pred[t]["snl"][i] = a[i - 364] * r
            b_cal = ols_fit(calX[tri], ly); trend = b_cal[12]
            for cfg, groups in CFG.items():
                o_te, r_te = info[INFO.get(cfg, "obs")]; oc = [c for g in groups for c in OLS_G[g]]; rc = [c for g in groups for c in RAW_G[g]]
                Xtr, Xte = np.c_[calX[tri], olsW[tri][:, oc]], np.c_[calX[tei], o_te[:, oc]]; b = ols_fit(Xtr, ly); pred[t][f"ols_{cfg}"][tei] = np.exp(Xte @ b)
                Gtr, Gte = np.c_[gbmX[tri], rawW[tri][:, rc]], np.c_[gbmX[tei], r_te[:, rc]]; yr = (np.array([(d - START).days for d in days_all]) / 365.25)
                g = GradientBoostingRegressor(**GBM_PARAMS).fit(Gtr, ly - trend * yr[tri]); pred[t][f"gbm_{cfg}"][tei] = np.exp(g.predict(Gte) + trend * yr[tei])
    ti = np.array([idx[d] for d in test_days]); P = {t: {m: pred[t][m][ti] for m in MODELS} for t in TARGETS}
    for t in TARGETS:
        for m in MODELS:
            if m.startswith("gbm"): P[t][m] = np.round(P[t][m], 2)
    A = {t: tcol[t][ti] for t in TARGETS}
    gcols = [f"gbm_{c}" for c in CFG]; gb = []
    for t in TARGETS:
        for k, d in enumerate(test_days):
            gb.append([d.isoformat(), t] + [repr(float(P[t][c][k])) for c in gcols])
    pd.DataFrame(gb, columns=["date", "target"] + gcols).to_csv(R / "gbm_predictions.csv", index=False)
    ob = []
    for t in TARGETS:
        for k, d in enumerate(test_days):
            ob.append([d.isoformat(), t, int(A[t][k]), round(float(P[t]["sn"][k]), 1), round(float(P[t]["snl"][k]), 1)] + [round(float(P[t][f"ols_{c}"][k]), 1) for c in CFG])
    pd.DataFrame(ob, columns=["date", "target", "actual", "sn", "snl"] + [f"ols_{c}" for c in CFG]).to_csv(R / "ols_predictions.csv", index=False)
    # ---- scores
    yrs = np.array([d.year for d in test_days]); sc = []
    for t in TARGETS:
        for m in MODELS:
            for per, mk in (("all", np.ones(len(ti), bool)), ("2024", yrs == 2024), ("2025", yrs == 2025)):
                sc.append({"target": t, "model": m, "period": per, **score(A[t][mk], P[t][m][mk])})
    scd = pd.DataFrame(sc); scd.to_csv(R / "scores.csv", index=False)
    g = lambda t, m, per="all", k="mape": float(scd[(scd.target == t) & (scd.model == m) & (scd.period == per)][k].iloc[0])
    ab = []
    for fam in ("ols", "gbm"):
        for t in TARGETS:
            base = g(t, f"{fam}_cal")
            for c in ("cal", "T", "R", "S", "all", "noT", "noR", "noS", "all_clim", "all_week"):
                ab.append({"family": fam, "target": t, "config": c, "mape": g(t, f"{fam}_{c}"), "mape_change_vs_calendar": g(t, f"{fam}_{c}") - base})
    pd.DataFrame(ab).to_csv(R / "ablation.csv", index=False)
    wv = []
    for fam in ("ols", "gbm"):
        for t in TARGETS:
            for info_, c in (("none (calendar only)", "cal"), ("observed on the day (upper bound)", "all"), ("climatology (honest)", "all_clim"), ("last week held flat (honest)", "all_week")):
                wv.append({"family": fam, "target": t, "weather_input": info_, "mape": g(t, f"{fam}_{c}"), "mae": g(t, f"{fam}_{c}", k="mae"), "bias_pct": g(t, f"{fam}_{c}", k="bias_pct")})
    pd.DataFrame(wv).to_csv(R / "weather_value.csv", index=False)
    # ---- folds
    fo = []; ym = np.array([d.strftime("%Y-%m") for d in test_days])
    for mth in sorted(set(ym)):
        mk = ym == mth; r_ = {"month": mth, "days": int(mk.sum()), "actual_per_day": float(A["total"][mk].mean())}
        for m in ("sn", "snl", "ols_cal", "ols_all", "ols_all_clim", "gbm_cal", "gbm_all", "gbm_all_clim"):
            r_[m] = score(A["total"][mk], P["total"][m][mk])["mape"]
        fo.append(r_)
    pd.DataFrame(fo).to_csv(R / "folds.csv", index=False)
    # ---- segments
    isH = np.array([d in hol_set for d in test_days]); isS = np.array([d in storm for d in test_days]); seg = {"holiday": isH, "storm (>= 1 inch rain or snow)": isS & ~isH, "other days": ~isH & ~isS, "all test days": np.ones(len(ti), bool)}; sg = []
    for t in TARGETS:
        for sname, mk in seg.items():
            for m in ("snl", "ols_cal", "ols_all", "ols_all_clim", "gbm_cal", "gbm_all", "gbm_all_clim"):
                sg.append({"target": t, "segment": sname, "model": m, **score(A[t][mk], P[t][m][mk])})
    pd.DataFrame(sg).to_csv(R / "segments.csv", index=False)
    # ---- Diebold-Mariano on daily absolute errors, total trips
    ae = lambda m: np.abs(P["total"][m] - A["total"]); dmr = []
    for a_, b_, label in (("snl", "ols_cal", "calendar regression vs seasonal naive with level"), ("ols_cal", "ols_all", "observed weather vs calendar only (regression)"), ("ols_cal", "ols_all_clim", "climatology vs calendar only (regression)"),
                          ("ols_cal", "ols_all_week", "last week vs calendar only (regression)"), ("gbm_cal", "gbm_all", "observed weather vs calendar only (boosting)"), ("gbm_cal", "gbm_all_clim", "climatology vs calendar only (boosting)"), ("ols_all", "gbm_all", "boosting vs regression, observed weather"),
                          ("ols_all_clim", "gbm_all_clim", "boosting vs regression, climatology"), ("ols_cal", "gbm_cal", "boosting vs regression, calendar only")):
        t_, p_ = dm(ae(a_) - ae(b_), DM_LAGS); dmr.append({"comparison": label, "reference": a_, "candidate": b_, "mean_abs_error_gain": float((ae(a_) - ae(b_)).mean()), "dm_t": t_, "dm_p": p_})
    pd.DataFrame(dmr).to_csv(R / "dm.csv", index=False)
    # ---- worst days of the honest regression
    e = (P["total"]["ols_all_clim"] - A["total"]) / A["total"]; order = sorted(range(len(ti)), key=lambda k: (-round(abs(e[k]), 9), k))[:12]; wd = []
    for k in order:
        d = test_days[k]; wd.append({"date": d.isoformat(), "weekday": d.strftime("%a"), "actual": int(A["total"][k]), "predicted_honest": float(P["total"]["ols_all_clim"][k]), "error_pct_honest": float(100 * e[k]), "predicted_observed_weather": float(P["total"]["ols_all"][k]),
                                     "error_pct_observed": float(100 * (P["total"]["ols_all"][k] - A["total"][k]) / A["total"][k]), "tmax": float(W.loc[d, "tmax"]), "prcp_mm": float(W.loc[d, "prcp"]), "snow_mm": float(W.loc[d, "snow"]), "holiday": int(d in hol_set)})
    pd.DataFrame(wd).to_csv(R / "worst_days.csv", index=False)
    # ---- sensitivity: all 1096 days, all weather terms, Newey-West
    X = np.c_[calX, olsW]; sens = []
    xs = lambda T: T / 10 - 1.5
    contrasts = {}
    for lo, hi in ((0, 10), (10, 20), (20, 30)):
        c = np.zeros(19); c[13], c[14] = xs(hi) - xs(lo), xs(hi) ** 2 - xs(lo) ** 2; contrasts[f"mean temperature {lo} to {hi} C"] = c
    for mm in (10, 25):
        c = np.zeros(19); c[15], c[16] = np.log1p(mm), 1; contrasts[f"rain day of {mm} mm vs dry"] = c
    c = np.zeros(19); c[17], c[18] = np.log1p(25.4), 1; contrasts["snow day of 25 mm, snow on the ground, vs none"] = c
    c = np.zeros(19); c[5] = 1; contrasts["Saturday vs Monday"] = c
    c = np.zeros(19); c[7] = 1; contrasts["federal holiday vs ordinary day"] = c
    for t in TARGETS:
        y = np.log(tcol[t]); inv = np.linalg.inv(X.T @ X); b = inv @ X.T @ y; res = y - X @ b; V = hac(X, res, inv, NW_LAGS)
        for name, c in contrasts.items():
            est, se = float(c @ b), float(np.sqrt(c @ V @ c)); sens.append({"target": t, "effect": name, "pct_change": 100 * (np.exp(est) - 1), "ci_low_pct": 100 * (np.exp(est - 1.96 * se) - 1), "ci_high_pct": 100 * (np.exp(est + 1.96 * se) - 1)})
    pd.DataFrame(sens).to_csv(R / "sensitivity.csv", index=False)
    # ---- how well do the honest weather inputs know the weather
    ws = []
    tr_, cl, wkv = rawW[ti], CLIM_RAW[ti], WEEK_RAW[ti]
    for name, f in (("mean temperature (C)", lambda a: (a[:, 0] + a[:, 1]) / 2), ("rain (mm)", lambda a: a[:, 2]), ("snow (mm)", lambda a: a[:, 3])):
        o, c_, w_ = f(tr_), f(cl), f(wkv)
        ws.append({"variable": name, "sd_observed": float(o.std()), "rmse_climatology": float(np.sqrt(((c_ - o) ** 2).mean())), "rmse_last_week": float(np.sqrt(((w_ - o) ** 2).mean())), "r2_climatology": 1 - float(((c_ - o) ** 2).sum() / ((o - o.mean()) ** 2).sum()), "r2_last_week": 1 - float(((w_ - o) ** 2).sum() / ((o - o.mean()) ** 2).sum())})
    pd.DataFrame(ws).to_csv(R / "wx_skill.csv", index=False)
    for fn in ("audit", "annual", "scores", "ablation", "weather_value", "folds", "segments", "dm", "worst_days", "sensitivity", "wx_skill"):
        d_ = pd.read_csv(R / f"{fn}.csv")
        if fn == "scores": d_ = d_[(d_.period == "all") & (d_.target == "total")]
        print("==", fn); print(d_.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
