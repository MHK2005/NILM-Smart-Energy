"""
Phase 2: cleaning, splitting, thresholds, labels and windows.

Run:  python -m src.data_preprocessing

Input : data/processed/selected_raw.csv.gz   (made by src.extract_data)
Output: data/processed/home_<id>.npz          cleaned 1-minute arrays per home
        data/processed/windows.npz            valid window end positions per split
        data/processed/preprocessing.json     every fitted setting (scaler, thresholds...)
        results/thresholds.json
        results/class_counts.csv
        results/cleaning_report.csv
        results/figures/class_distribution.png

Leakage rules:
  * thresholds, input scaler and power scale are fitted on TRAINING minutes
    of the development homes only;
  * windows never cross a split boundary, a home, or a long gap;
  * unseen homes are not used for anything except final evaluation.
"""
import numpy as np
import pandas as pd

from src.utils import (APPLIANCES, APPLIANCE_NAMES, DEV_HOMES, UNSEEN_HOMES, GRID_COL,
                       SOLAR_COLS, WINDOW, MAX_INTERP_GAP, NEG_NOISE_KW,
                       OVER_AGG_MARGIN_KW, THRESH_MIN_KW, THRESH_FRACTION, SPLIT_MODE,
                       SPLIT_RATIOS, LOCAL_TZ, PROCESSED_DIR, RESULTS_DIR, FIG_DIR,
                       REGION, ensure_dirs, save_json)

SPLITS = {0: "train", 1: "val", 2: "test", 3: "unseen"}


def interpolate_short_gaps(s, max_gap=MAX_INTERP_GAP):
    """Linear interpolation only inside NaN runs of length <= max_gap.
    Longer gaps stay NaN (invalid). Returns (filled series, number filled)."""
    isna = s.isna()
    if not isna.any():
        return s, 0
    run_id = (isna != isna.shift()).cumsum()
    run_len = isna.groupby(run_id).transform("sum")
    short = isna & (run_len <= max_gap)
    filled = s.interpolate(method="linear", limit_area="inside")
    out = s.copy()
    out[short] = filled[short]
    return out, int(out.notna().sum() - s.notna().sum())


def assign_splits(ts_utc, is_unseen):
    """Return an int8 array: 0 train, 1 val, 2 test, 3 unseen."""
    n = len(ts_utc)
    if is_unseen:
        return np.full(n, 3, dtype=np.int8)
    split = np.empty(n, dtype=np.int8)
    r_train, r_val, _ = SPLIT_RATIOS
    if SPLIT_MODE == "global":
        groups = [np.arange(n)]
    else:  # monthly blocks in local time
        month = ts_utc.tz_convert(LOCAL_TZ).month.values
        groups = [np.where(month == m)[0] for m in np.unique(month)]
    for idx in groups:
        k = len(idx)
        a, b = int(k * r_train), int(k * (r_train + r_val))
        split[idx[:a]], split[idx[a:b]], split[idx[b:]] = 0, 1, 2
    return split


def clean_home(g, home):
    """Clean one home. Returns a DataFrame on a continuous 1-minute grid and a report."""
    rep = {"home": home, "raw_rows": len(g)}
    g = g.sort_values("ts_utc")
    rep["duplicates_removed"] = int(g.ts_utc.duplicated().sum())
    g = g.drop_duplicates("ts_utc").set_index("ts_utc")
    full = pd.date_range(g.index.min(), g.index.max(), freq="1min", tz="UTC")
    g = g.reindex(full)
    rep["minutes_on_grid"] = len(full)
    rep["missing_minutes"] = int(g[GRID_COL].isna().sum())

    # ---- whole-house aggregate ----
    # grid is net import (negative when solar exports), so use = grid + solar.
    # A home whose solar column is empty all year has no PV: solar = 0.
    agg = g[GRID_COL].astype("float64").copy()
    for sc in SOLAR_COLS:
        if sc in g and g[sc].notna().any():
            agg = agg + g[sc]          # NaN solar minute -> NaN aggregate
            rep[f"has_{sc}"] = True
        else:
            rep[f"has_{sc}"] = False
    agg, rep["aggregate_interpolated"] = interpolate_short_gaps(agg)
    neg = agg < -NEG_NOISE_KW
    rep["aggregate_negative_invalid"] = int(neg.sum())
    agg[neg] = np.nan
    agg = agg.clip(lower=0)
    out = pd.DataFrame({"aggregate": agg})

    # ---- appliances ----
    for a in APPLIANCES:
        if a not in g or g[a].notna().sum() == 0:
            out[a] = np.nan
            rep[f"{a}_status"] = "not metered"
            continue
        x, n_int = interpolate_short_gaps(g[a].astype("float64"))
        too_neg = x < -NEG_NOISE_KW
        x[too_neg] = np.nan
        x = x.clip(lower=0)               # tiny negatives are sensor noise
        over = x > out["aggregate"] + OVER_AGG_MARGIN_KW
        x[over] = np.nan                  # cannot exceed the whole house
        out[a] = x
        rep[f"{a}_status"] = "ok"
        rep[f"{a}_interpolated"] = n_int
        rep[f"{a}_invalid_negative"] = int(too_neg.sum())
        rep[f"{a}_invalid_over_aggregate"] = int(over.sum())
        rep[f"{a}_valid"] = int(x.notna().sum())
    return out, rep


def valid_window_ends(agg_valid, segment, any_target_valid, w=WINDOW):
    """Window ending at t is valid when all w aggregate minutes are valid,
    the window lies inside one split segment, and some target is valid at t."""
    n = len(agg_valid)
    if n < w:
        return np.array([], dtype=np.int64)
    c = np.concatenate([[0], np.cumsum(agg_valid.astype(np.int64))])
    full_valid = (c[w:] - c[:-w]) == w           # index j -> window ending at j+w-1
    ends = np.arange(w - 1, n)
    same_seg = segment[ends] == segment[ends - w + 1]
    ok = full_valid & same_seg & any_target_valid[ends]
    return ends[ok]


def main():
    # Note: "ts" in home_<id>.npz is stored as unix seconds (UTC).
    ensure_dirs()
    raw = pd.read_csv(PROCESSED_DIR / "selected_raw.csv.gz")
    raw["ts_utc"] = pd.to_datetime(raw.ts_utc, utc=True)
    homes = DEV_HOMES + UNSEEN_HOMES

    reports, cleaned = [], {}
    for h in homes:
        g = raw[raw.dataid == h]
        df, rep = clean_home(g, h)
        df["split"] = assign_splits(df.index, h in UNSEEN_HOMES)
        df["segment"] = (df.split != df.split.shift()).cumsum().astype(np.int32)
        cleaned[h] = df
        reports.append(rep)
        print(f"home {h}: {len(df):,} minutes, missing {rep['missing_minutes']}, "
              f"aggregate interpolated {rep['aggregate_interpolated']}")
    pd.DataFrame(reports).to_csv(RESULTS_DIR / "cleaning_report.csv", index=False)

    # ---- fit on TRAINING minutes of development homes only ----
    train = pd.concat([cleaned[h][cleaned[h].split == 0] for h in DEV_HOMES])
    agg_train = train["aggregate"].dropna()
    scaler = {"mean": float(agg_train.mean()), "std": float(agg_train.std())}
    thresholds, power_scale = {}, {}
    for a in APPLIANCES:
        v = train[a].dropna()
        p99 = float(v.quantile(0.99))
        thresholds[a] = round(max(THRESH_MIN_KW, THRESH_FRACTION * p99), 4)
        power_scale[a] = max(p99, 0.05)
        print(f"{a}: training p99 = {p99:.3f} kW -> threshold {thresholds[a]:.3f} kW")

    # ---- save per-home arrays and window indices ----
    win = {s: [] for s in SPLITS.values()}
    counts = {(a, s): [0, 0] for a in APPLIANCES for s in SPLITS.values()}
    availability = {}
    for hi, h in enumerate(homes):
        df = cleaned[h]
        agg = df["aggregate"].values.astype(np.float32)
        P = df[APPLIANCES].values.astype(np.float32)
        V = ~np.isnan(P)
        on = (np.nan_to_num(P) > np.array([thresholds[a] for a in APPLIANCES])) & V
        split = df.split.values
        np.savez_compressed(
            PROCESSED_DIR / f"home_{h}.npz",
            ts=df.index.as_unit("s").asi8, aggregate=agg, power=P, valid=V, on=on,
            split=split, segment=df.segment.values)
        availability[str(h)] = {a: bool(V[:, i].any()) for i, a in enumerate(APPLIANCES)}
        ends = valid_window_ends(~np.isnan(agg), df.segment.values, V.any(axis=1))
        for code, name in SPLITS.items():
            e = ends[split[ends] == code]
            if len(e) == 0:
                continue
            win[name].append(np.stack([np.full(len(e), hi), e], axis=1))
            for i, a in enumerate(APPLIANCES):
                vv = V[e, i]
                counts[(a, name)][0] += int(on[e, i][vv].sum())
                counts[(a, name)][1] += int((~on[e, i])[vv].sum())
    win_arrays = {k: (np.concatenate(v) if v else np.zeros((0, 2), np.int64))
                  for k, v in win.items()}
    np.savez_compressed(PROCESSED_DIR / "windows.npz", **win_arrays)
    for k, v in win_arrays.items():
        print(f"{k:7s} windows: {len(v):,}")

    rows = []
    for (a, s), (n_on, n_off) in counts.items():
        rows.append({"appliance": a, "name": APPLIANCE_NAMES[a], "split": s,
                     "threshold_kW": thresholds[a], "ON": n_on, "OFF": n_off,
                     "total_valid": n_on + n_off,
                     "ON_percent": round(100 * n_on / max(n_on + n_off, 1), 2)})
    cc = pd.DataFrame(rows)
    cc.to_csv(RESULTS_DIR / "class_counts.csv", index=False)
    print(cc.to_string(index=False))

    save_json({a: {"threshold_kW": thresholds[a],
                   "rule": "max(0.02 kW, 10% of training 99th percentile)"}
               for a in APPLIANCES}, RESULTS_DIR / "thresholds.json")
    save_json({
        "region": REGION, "window": WINDOW, "appliances": APPLIANCES,
        "dev_homes": DEV_HOMES, "unseen_homes": UNSEEN_HOMES, "homes_order": homes,
        "split_mode": SPLIT_MODE, "split_ratios": list(SPLIT_RATIOS),
        "thresholds_kW": thresholds, "input_scaler": scaler,
        "power_scale_kW": power_scale, "availability": availability,
        "units": "kW internally; metrics reported in W",
    }, PROCESSED_DIR / "preprocessing.json")

    plot_class_distribution(cc)


def plot_class_distribution(cc):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(APPLIANCES), figsize=(4 * len(APPLIANCES), 3.6))
    order = ["train", "val", "test", "unseen"]
    for ax, a in zip(axes, APPLIANCES):
        d = cc[cc.appliance == a].set_index("split").loc[order]
        x = np.arange(len(order))
        ax.bar(x - 0.2, d.OFF, 0.4, label="OFF", color="#94a3b8")
        ax.bar(x + 0.2, d.ON, 0.4, label="ON", color="#0d9488")
        ax.set_xticks(x, order)
        ax.set_yscale("log")
        ax.set_title(f"{APPLIANCE_NAMES[a]}\nthreshold {d.threshold_kW.iloc[0]:.2f} kW")
        ax.set_ylabel("windows (log scale)")
    axes[0].legend()
    fig.suptitle("ON/OFF class distribution per split")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "class_distribution.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
