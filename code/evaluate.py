"""
Evaluate MLP and 1D-CNN on the held-out chronological test split (seen homes) and on unseen homes.

Probability cut-off per appliance is chosen on the VALIDATION set only (maximises macro-F1).
Appliance targets are NaN (masked) in homes without that circuit. Appliances are ranked by VALIDATION macro-F1.
Output : results.csv, results_summary.txt, confusion_matrices.png, power_test.png, power_unseen.png, training_curves.png
Run    : python evaluate.py
"""
import re
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix

meta = json.load(open("prep_meta.json"))
APPS = meta["appliances"]
THR = np.array([meta["thresholds_kw"][a] for a in APPS])
NAMES = {"mlp": "MLP", "cnn1d": "1D-CNN"}
SETS = {"test": "Chronological test (seen homes)", "unseen": "Unseen homes (full year)"}
MIN_T, HIGH_T = (0.85, 0.75, 0.70), (0.90, 0.85, 0.80)

preds = {m: np.load(f"preds_{m}.npz") for m in NAMES}


def best_cutoff(prob, true_on):
    """Probability cut-off that maximises macro-F1 on validation data."""
    grid = np.arange(0.05, 0.96, 0.025)
    if true_on.min() == true_on.max():
        return 0.5
    return float(grid[np.argmax([f1_score(true_on, prob >= g, average="macro") for g in grid])])


rows, cms = [], {}
for m, p in preds.items():
    for j, a in enumerate(APPS):
        vm = ~np.isnan(p["val_true_pow"][:, j])
        cut = best_cutoff(p["val_prob"][vm, j], p["val_true_pow"][vm, j] > THR[j])
        val_f1 = f1_score(p["val_true_pow"][vm, j] > THR[j], p["val_prob"][vm, j] >= cut, average="macro")
        for s in SETS:
            ok = ~np.isnan(p[f"{s}_true_pow"][:, j])
            y_on = p[f"{s}_true_pow"][ok, j] > THR[j]
            y_hat = p[f"{s}_prob"][ok, j] >= cut
            err = p[f"{s}_pow"][ok, j] - p[f"{s}_true_pow"][ok, j]
            acc, bal, f1 = accuracy_score(y_on, y_hat), balanced_accuracy_score(y_on, y_hat), f1_score(y_on, y_hat, average="macro")
            rows.append(dict(model=NAMES[m], set=s, appliance=a, cutoff=round(cut, 3), val_macro_f1=val_f1, ON=int(y_on.sum()), OFF=int((~y_on).sum()),
                             accuracy=acc, balanced_acc=bal, macro_f1=f1,
                             MAE_W=1000 * np.abs(err).mean(), RMSE_W=1000 * np.sqrt((err ** 2).mean()),
                             meets_min=bool(acc >= MIN_T[0] and bal >= MIN_T[1] and f1 >= MIN_T[2]),
                             meets_higher=bool(acc >= HIGH_T[0] and bal >= HIGH_T[1] and f1 >= HIGH_T[2])))
            cms[(m, s, a)] = confusion_matrix(y_on, y_hat, labels=[False, True])

res = pd.DataFrame(rows)
res.round(4).to_csv("results.csv", index=False)
pd.set_option("display.width", 250)
with open("results_summary.txt", "w") as f:
    f.write("Validation ranking (macro-F1 at the validation cut-off): "
            + str(res[res.set == "test"].groupby("appliance").val_macro_f1.mean().round(3).sort_values(ascending=False).to_dict()) + "\n")
    for s, title in SETS.items():
        t = res[res.set == s].drop(columns="set").round(3)
        f.write(f"\n=== {title} ===\n{t.to_string(index=False)}\n")
print(open("results_summary.txt").read())

# ---- confusion matrices
fig, ax = plt.subplots(len(SETS) * 2, len(APPS), figsize=(14, 14))
for r, (s, m) in enumerate([(s, m) for s in SETS for m in NAMES]):
    for c, a in enumerate(APPS):
        cm = cms[(m, s, a)]
        ax[r, c].imshow(cm, cmap="Blues")
        for (i, k), v in np.ndenumerate(cm):
            ax[r, c].text(k, i, f"{v:,}", ha="center", va="center", color="black", fontsize=8)
        ax[r, c].set_xticks([0, 1], ["OFF", "ON"])
        ax[r, c].set_yticks([0, 1], ["OFF", "ON"])
        ax[r, c].set_title(f"{NAMES[m]} | {a} | {s}", fontsize=9)
        ax[r, c].set_xlabel("predicted")
        ax[r, c].set_ylabel("actual")
plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=130)
plt.close()


# ---- actual vs predicted power (2-day stretch with most ON minutes)
def power_plot(s, fname, only_home=None):
    fig, ax = plt.subplots(len(APPS), 1, figsize=(12, 11))
    for j, a in enumerate(APPS):
        p0 = preds["mlp"]
        homes, idx, true = p0[f"{s}_home"], p0[f"{s}_idx"], p0[f"{s}_true_pow"][:, j]
        best = (-1, None, None)
        for h in np.unique(homes[~np.isnan(true)]):
            if only_home is not None and h != only_home:
                continue
            mask = homes == h
            hi, ht = idx[mask], true[mask]
            for lo in range(hi.min(), hi.max() - 2880, 720):
                sel = (hi >= lo) & (hi < lo + 2880)
                score = (ht[sel] > THR[j]).sum() if sel.sum() == 2880 else -1
                if score > best[0]:
                    best = (score, h, lo)
        _, h, lo = best
        mask = (homes == h) & (idx >= lo) & (idx < lo + 2880)
        t = np.arange(mask.sum()) / 60
        ax[j].plot(t, 1000 * true[mask], "k", lw=1.3, label="actual")
        for m, col in (("mlp", "tab:blue"), ("cnn1d", "tab:red")):
            ax[j].plot(t, 1000 * preds[m][f"{s}_pow"][mask, j], col, lw=0.9, alpha=0.8, label=NAMES[m])
        ax[j].set_ylabel(f"{a} (W)")
        ax[j].set_title(f"{a} - home {h}", fontsize=9)
    ax[0].legend(loc="upper right")
    ax[-1].set_xlabel("hours")
    plt.tight_layout()
    plt.savefig(fname, dpi=130)
    plt.close()


power_plot("test", "power_test.png")
power_plot("unseen", "power_unseen.png")

# ---- training curves from log
log = open("train_log.txt").read()
fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
for k, m in enumerate(NAMES):
    ep = re.findall(rf"{m} epoch\s+(\d+)\s+train_loss=([\d.]+)\s+val_loss=([\d.]+)", log)
    e, tr, va = (np.array(x, dtype=float) for x in zip(*ep))
    ax[k].plot(e, tr, label="train")
    ax[k].plot(e, va, label="val")
    ax[k].set_title(NAMES[m])
    ax[k].set_xlabel("epoch")
    ax[k].legend()
plt.tight_layout()
plt.savefig("training_curves.png", dpi=130)
