"""
NILM data preparation -- Pecan Street Austin 1-minute data (CT-466 CCP)

Input  : austin_slim2.pkl  (columns: dataid, ts, grid, solar, air1, furnace1, drye1, ...)
Output : prep_austin.npz  (clean arrays, split labels, valid-window indices)
         class_counts.csv (ON/OFF counts per appliance, split and home)

Run    : python data_prep.py
"""
import json
import os
import numpy as np
import pandas as pd

SEED = 42
WINDOW = int(os.environ.get("NILM_WINDOW", 60))   # previous WINDOW one-minute readings (default 60)
APPLIANCES = ["air1", "drye1", "car1", "refrigerator1"]
MIN_COVERAGE = 0.95               # a circuit is used in a home only if >=95% of its readings exist
EXCLUDE_HOMES = [7536]            # 2,895 kW spike in the whole-house signal
UNSEEN_HOMES = [4767, 8565]       # whole homes kept out of training; both have air1, drye1, refrigerator1; 4767 also has car1
MAX_FILL = 5                      # interpolate gaps up to 5 minutes only
SPLIT_FRAC = (0.70, 0.15, 0.15)   # train / val / test, single chronological cut per home
SPLIT_NAMES = ["train", "val", "test"]
THR_FRACTION = 0.10               # ON threshold = 10% of the 99th percentile of TRAIN power
THR_FLOOR_KW = 0.02

raw = pd.read_pickle("austin_slim2.pkl")
cov = raw.groupby("dataid")["grid"].count() / raw.groupby("dataid").size()
CANDIDATE_HOMES = sorted(h for h in cov.index[cov >= MIN_COVERAGE] if h not in EXCLUDE_HOMES)
raw = raw[raw.dataid.isin(CANDIDATE_HOMES)]

full_index = pd.date_range("2018-01-01 06:00", "2019-01-01 05:59", freq="min", tz="UTC")
unseen = sorted(UNSEEN_HOMES)
seen = [h for h in CANDIDATE_HOMES if h not in unseen]


def clean_home(g):
    """Sort, drop duplicates, put on a continuous 1-min grid, fill short gaps, fix units/sign."""
    g = g.drop_duplicates("ts").set_index("ts").reindex(full_index)
    total = g["grid"] + g["solar"].fillna(0.0)        # whole-house consumption (kW)
    cols = {"total": total, **{a: g[a] for a in APPLIANCES}}
    out = pd.DataFrame(cols).clip(lower=0.0)          # small negative readings = meter offset
    for a in APPLIANCES:                              # circuit missing / never used in this home -> no target
        if g[a].count() < MIN_COVERAGE * len(g) or (a == "car1" and g[a].max() < 0.2):
            out[a] = np.nan
    out = out.interpolate(limit=MAX_FILL, limit_area="inside")
    return out


def chronological_split_labels(n):
    """0/1/2 = train/val/test: first 70% / next 15% / last 15% of the year, strictly in time order."""
    pos = np.arange(n) / (n - 1)
    lab = np.where(pos < SPLIT_FRAC[0], 0, np.where(pos < SPLIT_FRAC[0] + SPLIT_FRAC[1], 1, 2))
    return lab.astype(np.int8)


homes = {h: clean_home(g) for h, g in raw.groupby("dataid")}
split = chronological_split_labels(len(full_index))

# thresholds from TRAIN minutes of TRAIN (seen) homes only
train_pw = {a: np.concatenate([homes[h][a].values[split == 0] for h in seen]) for a in APPLIANCES}
thr = {}
for a in APPLIANCES:
    x = train_pw[a][~np.isnan(train_pw[a])]
    thr[a] = float(max(THR_FRACTION * np.quantile(x, 0.99), THR_FLOOR_KW))

# scaling statistics from train only
tot_train = np.concatenate([homes[h]["total"].values[split == 0] for h in seen])
tot_train = tot_train[~np.isnan(tot_train)]
stats = {"total_mean": float(tot_train.mean()), "total_std": float(tot_train.std()),
         **{f"{a}_scale": float(np.quantile(train_pw[a][~np.isnan(train_pw[a])], 0.99)) for a in APPLIANCES}}

# window validity: all WINDOW minutes in the same split, no NaN in the input (NaN target = circuit absent, masked in the loss)
arrays, valid_idx, rows = {}, {}, []
for h, d in homes.items():
    X = d["total"].values.astype(np.float32)
    Y = np.stack([d[a].values for a in APPLIANCES], 1).astype(np.float32)
    ok_min = ~np.isnan(X)
    csum_ok = np.concatenate([[0], np.cumsum(ok_min)])
    win_ok = (csum_ok[WINDOW:] - csum_ok[:-WINDOW]) == WINDOW              # window inputs all finite
    sp = split.astype(np.int16)
    sp_c = [np.concatenate([[0], np.cumsum(sp == k)]) for k in range(3)]
    same = [(c[WINDOW:] - c[:-WINDOW]) == WINDOW for c in sp_c]            # window fully in split k
    end = np.arange(WINDOW - 1, len(X))
    arrays[f"X_{h}"], arrays[f"Y_{h}"] = X, Y
    for k, name in enumerate(SPLIT_NAMES):
        idx = end[win_ok & same[k]]
        valid_idx[f"idx_{h}_{name}"] = idx.astype(np.int32)
        for j, a in enumerate(APPLIANCES):
            y = Y[idx, j]
            y = y[~np.isnan(y)]
            on = int((y > thr[a]).sum())
            rows.append(dict(home=h, unseen_home=h in unseen, split=name, appliance=a,
                             windows=len(y), ON=on, OFF=len(y) - on))

counts = pd.DataFrame(rows)
counts.to_csv("class_counts.csv", index=False)
np.savez_compressed("prep_austin.npz", **arrays, **valid_idx, split=split)
json.dump({"seed": SEED, "window": WINDOW, "appliances": APPLIANCES, "seen_homes": seen,
           "unseen_homes": unseen, "thresholds_kw": thr, "scaling": stats},
          open("prep_meta.json", "w"), indent=2)

print("seen homes  :", seen)
print("unseen homes:", unseen)
print("ON thresholds (kW, from train only):", {a: round(v, 3) for a, v in thr.items()})
summ = counts[~counts.unseen_home].groupby(["appliance", "split"])[["windows", "ON", "OFF"]].sum()
summ["ON_%"] = (100 * summ.ON / summ.windows).round(1)
print(summ)
print("\nUnseen homes (full year, all splits pooled):")
u = counts[counts.unseen_home].groupby("appliance")[["windows", "ON", "OFF"]].sum()
u["ON_%"] = (100 * u.ON / u.windows).round(1)
print(u)
