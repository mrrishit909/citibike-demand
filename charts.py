"""Step 6: charts.   ./venv/bin/python charts.py -> charts/00_cover.png, 01_level.png, 02_models.png, 03_ablation.png, 04_segments.png, 05_riders.png"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
R = HERE / "results"
INK, DIM, GRID, BG, BLUE, ORANGE, GRAY, GREEN = "#f2f2f0", "#8a8a87", "#1d1d1d", "#0b0b0b", "#3987e5", "#d95926", "#9a9a96", "#3aa876"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "text.color": INK,
                     "axes.edgecolor": "#3a3a3a", "axes.labelcolor": DIM, "xtick.color": DIM, "ytick.color": DIM,
                     "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 1,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 11, "axes.titlesize": 13,
                     "axes.titlelocation": "left", "axes.titlepad": 12})
SC = pd.read_csv(R / "scores.csv"); FO = pd.read_csv(R / "folds.csv"); SG = pd.read_csv(R / "segments.csv"); AB = pd.read_csv(R / "ablation.csv"); SE = pd.read_csv(R / "sensitivity.csv"); MO = pd.read_csv(R / "monthly.csv"); AN = pd.read_csv(R / "annual.csv")
OP = pd.read_csv(R / "ols_predictions.csv"); HO = pd.read_csv(R / "holidays.csv"); WV = pd.read_csv(R / "weather_value.csv")
LAB = {"snl": "last year, scaled", "sn": "same weekday last year", "ols_cal": "calendar", "ols_all": "calendar + observed weather", "ols_all_clim": "calendar + normal weather", "ols_all_week": "calendar + last week's weather",
       "gbm_cal": "calendar", "gbm_all": "calendar + observed weather", "gbm_all_clim": "calendar + normal weather", "gbm_all_week": "calendar + last week's weather"}
COL = {"snl": DIM, "sn": "#5a5a58", "ols_cal": GRAY, "ols_all": GREEN, "ols_all_clim": BLUE, "ols_all_week": ORANGE, "gbm_cal": GRAY, "gbm_all": GREEN, "gbm_all_clim": BLUE, "gbm_all_week": ORANGE}


def sc(t, m, k="mape", per="all"):
    return float(SC[(SC.target == t) & (SC.model == m) & (SC.period == per)][k].iloc[0])


def dates(ax):
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=3)); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %-d")); ax.tick_params(axis="x", labelsize=9)


def save(fig, name):
    fig.tight_layout(); fig.savefig(HERE / "charts" / name, dpi=150); plt.close(fig)


def level(axes):
    ax = axes[0]; d = pd.read_csv(R / "ols_predictions.csv")                                    # test days only; the full series is rebuilt from monthly.csv
    x = np.arange(len(MO)); ax.bar(x, MO["per_day"] / 1000, color=BLUE, width=0.8, label="trips per day")
    ax2 = ax.twinx(); ax2.plot(x, 100 * MO["electric_share"], color=ORANGE, linewidth=2.2, label="electric bikes, share of trips"); ax2.set_ylim(0, 100); ax2.grid(False); ax2.spines["right"].set_visible(True); ax2.spines["right"].set_color("#3a3a3a"); ax2.tick_params(colors=ORANGE); ax2.set_ylabel("electric share of trips (%)", color=ORANGE)
    ax.set_xticks(x[::6]); ax.set_xticklabels([m[2:] for m in MO["ym"]][::6], fontsize=9); ax.set_ylabel("trips per day (thousands)"); ax.grid(axis="x", visible=False); ax.set_title("Season and growth, month by month", fontsize=12)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, frameon=False, labelcolor=INK, fontsize=8.5, loc="upper left")
    ax = axes[1]; x = np.arange(len(AN)); ax.bar(x, AN["per_day"] / 1000, color=BLUE, width=0.55)
    for xi, (pdv, g) in enumerate(zip(AN["per_day"], AN["growth_pct"])):
        ax.text(xi, pdv / 1000 + 4, f"{pdv / 1000:.0f}k" + ("" if pd.isna(g) else f"  ({g:+.1f}%)"), ha="center", fontsize=9.5, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(AN["yr"].astype(str)); ax.set_ylim(0, AN["per_day"].max() / 1000 * 1.18); ax.set_ylabel("trips per day (thousands)"); ax.grid(axis="x", visible=False); ax.set_title("The level moves from year to year", fontsize=12)


def models(axes):
    ax = axes[0]; keys = ["snl", "ols_cal", "ols_all_week", "ols_all_clim", "ols_all", "gbm_cal", "gbm_all_clim", "gbm_all"]; names = [("last year, scaled" if k == "snl" else ("regression: " if k.startswith("ols") else "boosting: ") + LAB[k].replace("calendar + ", "+ ").replace("calendar", "calendar only")) for k in keys]
    y = np.arange(len(keys))[::-1]; v = [sc("total", k) for k in keys]; ax.barh(y, v, color=[COL[k] for k in keys], height=0.7)
    for yi, vi in zip(y, v):
        ax.text(vi + 0.3, yi, f"{vi:.1f}%", va="center", fontsize=9, color=INK)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8.5); ax.set_xlim(0, max(v) * 1.15); ax.set_xlabel("mean absolute % error, 731 test days (lower is better)"); ax.grid(axis="y", visible=False); ax.set_title("Error of each forecast, all 24 months", fontsize=12)
    ax = axes[1]; x = np.arange(len(FO))
    for k in ("snl", "ols_cal", "ols_all_clim", "ols_all"):
        ax.plot(x, FO[k], color=COL[k], linewidth=2 if k != "snl" else 1.6, label=LAB[k], linestyle="--" if k == "snl" else "-")
    ax.set_xticks(x[::3]); ax.set_xticklabels([m[2:] for m in FO["month"]][::3], fontsize=9); ax.set_ylabel("mean absolute % error in the month"); ax.set_ylim(0, None); ax.legend(frameon=False, labelcolor=INK, fontsize=8, loc="upper left"); ax.set_ylim(0, 75); ax.grid(axis="x", visible=False); ax.set_title("Month by month, regression", fontsize=12)


def ablation(axes):
    for ax, fam, ttl in zip(axes, ("ols", "gbm"), ("Regression", "Boosting")):
        a = AB[(AB.family == fam) & (AB.target == "total")].set_index("config"); base = float(a.loc["cal", "mape"]); items = [("temperature", "T", "noT"), ("rain", "R", "noR"), ("snow", "S", "noS"), ("all three", "all", None)]; x = np.arange(len(items)); w = 0.36
        add = [base - float(a.loc[c, "mape"]) for _, c, _ in items]; drop = [float(a.loc["noT" if n == "temperature" else "noR" if n == "rain" else "noS", "mape"]) - float(a.loc["all", "mape"]) if n != "all three" else np.nan for n, _, _ in items]
        ax.bar(x - w / 2, add, width=w * 0.92, color=GREEN, label="added to the calendar"); ax.bar(x + w / 2, np.nan_to_num(drop), width=w * 0.92, color=BLUE, label="removed from all three")
        for xi, v in list(zip(x - w / 2, add)) + [(xi, v) for xi, v in zip(x + w / 2, drop) if not np.isnan(v)]:
            ax.text(xi, v + (0.15 if v >= 0 else -0.15), f"{v:.1f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=9, color=INK)
        ax.axhline(0, color=DIM, linewidth=0.9); ax.set_xticks(x); ax.set_xticklabels([n for n, _, _ in items], fontsize=9); ax.grid(axis="x", visible=False); ax.set_ylabel("fall in % error, observed weather (points)"); ax.set_title(f"{ttl}: what each weather group adds", fontsize=12)
    axes[0].legend(frameon=False, labelcolor=INK, fontsize=8.5, loc="upper left")


def segments(axes):
    ax = axes[0]; segs = ["holiday", "storm (>= 1 inch rain or snow)", "other days"]; names = ["holidays", "storm days", "other days"]; ms = ["ols_cal", "ols_all_clim", "ols_all"]; x = np.arange(len(segs)); w = 0.27
    g = SG[SG.target == "total"]
    for dx, m in zip((-w, 0, w), ms):
        v = [float(g[(g.segment == s) & (g.model == m)]["mape"].iloc[0]) for s in segs]; ax.bar(x + dx, v, width=w * 0.92, color=COL[m], label=LAB[m].replace("calendar + ", "+ "))
        for xi, vi in zip(x + dx, v):
            ax.text(xi, vi + 1, f"{vi:.0f}", ha="center", fontsize=8.5, color=INK)
    n = [int(g[(g.segment == s) & (g.model == "ols_cal")]["n"].iloc[0]) for s in segs]; ax.set_xticks(x); ax.set_xticklabels([f"{a}\n({b} days)" for a, b in zip(names, n)], fontsize=9); ax.grid(axis="x", visible=False); ax.set_ylabel("mean absolute % error"); ax.legend(frameon=False, labelcolor=INK, fontsize=8.5, loc="upper right"); ax.set_title("Regression error by kind of day", fontsize=12)
    ax = axes[1]; d = OP[OP.target == "total"].copy(); d["date"] = pd.to_datetime(d["date"]); d = d[(d.date >= "2025-03-01") & (d.date < "2025-06-01")]
    ax.plot(d["date"], d["actual"] / 1000, color=INK, linewidth=1.8, label="actual"); ax.plot(d["date"], d["ols_all_clim"] / 1000, color=BLUE, linewidth=1.6, label="normal weather"); ax.plot(d["date"], d["ols_all"] / 1000, color=GREEN, linewidth=1.6, label="observed weather")
    ax.set_ylabel("trips per day (thousands)"); ax.legend(frameon=False, labelcolor=INK, fontsize=8.5, loc="lower right", ncol=3); dates(ax); ax.set_ylim(0, None); ax.set_title("Spring 2025, regression forecasts", fontsize=12)


def riders(axes):
    ax = axes[0]; items = [("calendar only", "ols_cal"), ("normal weather", "ols_all_clim"), ("observed weather", "ols_all")]; x = np.arange(3); w = 0.27
    for dx, (nm, m), c in zip((-w, 0, w), items, (GRAY, BLUE, GREEN)):
        v = [sc(t, m) for t in ("total", "member", "casual")]; ax.bar(x + dx, v, width=w * 0.92, color=c, label=nm)
        for xi, vi in zip(x + dx, v):
            ax.text(xi, vi + 0.6, f"{vi:.0f}", ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(["all riders", "members", "casual riders"], fontsize=9); ax.grid(axis="x", visible=False); ax.set_ylabel("mean absolute % error (regression)"); ax.legend(frameon=False, labelcolor=INK, fontsize=8.5, loc="upper left"); ax.set_title("Weather matters more for casual riders", fontsize=12)
    ax = axes[1]; eff = [("mean temperature 10 to 20 C", "10 to 20 °C"), ("rain day of 10 mm vs dry", "10 mm of rain"), ("snow day of 25 mm, snow on the ground, vs none", "25 mm of snow"), ("Saturday vs Monday", "Saturday vs Monday")]; y = np.arange(len(eff))[::-1]; h = 0.3
    for dy, t, c, lab in ((h / 2 + 0.02, "member", BLUE, "members"), (-h / 2 - 0.02, "casual", ORANGE, "casual riders")):
        d = SE[SE.target == t].set_index("effect"); v = np.array([d.loc[e, "pct_change"] for e, _ in eff]); lo = np.array([d.loc[e, "ci_low_pct"] for e, _ in eff]); hi = np.array([d.loc[e, "ci_high_pct"] for e, _ in eff])
        ax.barh(y + dy, v, height=h, color=c, label=lab); ax.errorbar(v, y + dy, xerr=[v - lo, hi - v], fmt="none", ecolor=INK, elinewidth=1, capsize=2)
    ax.axvline(0, color=DIM, linewidth=0.9); ax.set_yticks(y); ax.set_yticklabels([n for _, n in eff], fontsize=9); ax.set_xlabel("change in trips, % (95% interval)"); ax.grid(axis="y", visible=False); ax.legend(frameon=False, labelcolor=INK, fontsize=8.5, loc="lower left"); ax.set_title("What the weather does to a day", fontsize=12)


def main():
    (HERE / "charts").mkdir(exist_ok=True)
    fig = plt.figure(figsize=(14, 8)); g = fig.add_gridspec(1, 2, width_ratios=[1, 1.05]); a = fig.add_subplot(g[0]); b = fig.add_subplot(g[1])
    keys = [("calendar only", "ols_cal", GRAY), ("last week's weather", "ols_all_week", ORANGE), ("normal weather", "ols_all_clim", BLUE), ("observed weather (not knowable)", "ols_all", GREEN)]; v = [sc("total", k) for _, k, _ in keys]; y = np.arange(len(keys))[::-1]
    a.barh(y, v, color=[c for *_, c in keys], height=0.62)
    for yi, vi in zip(y, v):
        a.text(vi + 0.3, yi, f"{vi:.1f}%", va="center", fontsize=13, color=INK)
    a.set_yticks(y); a.set_yticklabels([n for n, *_ in keys], fontsize=11.5); a.set_xlim(0, max(v) * 1.2); a.set_xlabel("mean absolute % error of daily trips, 24 test months"); a.grid(axis="y", visible=False); a.set_title("Regression, one month ahead, by the weather it is given", fontsize=14)
    d = OP[OP.target == "total"].copy(); d["date"] = pd.to_datetime(d["date"]); d = d[(d.date >= "2025-03-01") & (d.date < "2025-06-01")]
    b.plot(d["date"], d["actual"] / 1000, color=INK, linewidth=2, label="actual"); b.plot(d["date"], d["ols_all_clim"] / 1000, color=BLUE, linewidth=1.8, label="forecast with normal weather"); b.plot(d["date"], d["ols_all"] / 1000, color=GREEN, linewidth=1.8, label="forecast with observed weather")
    dates(b); b.set_ylabel("trips per day (thousands)"); b.set_ylim(0, None); b.legend(frameon=False, labelcolor=INK, fontsize=10, loc="lower right"); b.set_title("Spring 2025: the storms are in the weather, not the calendar", fontsize=14)
    ol, oc, oo = sc("total", "ols_cal"), sc("total", "ols_all_clim"), sc("total", "ols_all")
    fig.suptitle(f"Observed weather cuts the daily forecast error from {ol:.0f}% to {oo:.0f}%; the normal weather only to {oc:.0f}%", x=0.01, ha="left", fontsize=19, color=INK, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(HERE / "charts" / "00_cover.png", dpi=100); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9)); level(axes); save(fig, "01_level.png")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0)); models(axes); save(fig, "02_models.png")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9)); ablation(axes); save(fig, "03_ablation.png")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9)); segments(axes); save(fig, "04_segments.png")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9)); riders(axes); save(fig, "05_riders.png")


if __name__ == "__main__":
    main()
