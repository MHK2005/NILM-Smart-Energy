"""
Phase 5: evaluation of all trained models on data they never saw in training.

Run:  python -m src.evaluate                 predict with every trained model, then
                                              build all tables and figures
      python -m src.evaluate --no-unseen     skip the (slower) unseen-home predictions
      python -m src.evaluate --from-saved    rebuild tables/figures from saved predictions
                                              (no PyTorch needed)

Procedure (identical for every model):
  1. Predict validation, test and unseen-home windows.
  2. Choose the ON probability cut-off per appliance on VALIDATION only
     (the value in 0.05..0.95 with the best validation macro-F1).
     The test set is never used to choose anything.
  3. Apply it to the test set and the unseen homes and compute
     Accuracy, Balanced Accuracy, Macro-F1, Precision, Recall, confusion
     matrix, MAE and RMSE (in watts) per appliance.

Outputs in results/: results.csv, model_comparison.csv, unseen_home_results.csv,
best_models.csv, benchmark_summary.csv, prob_thresholds.json, figures/*.png
"""
import argparse

import numpy as np
import pandas as pd

from src.dataset import NILMData
from src.utils import (APPLIANCE_NAMES, BENCHMARK_APPLIANCES, MODEL_NAMES, MODEL_LABELS,
                       MIN_TARGET, HIGH_TARGET, MODELS_DIR, RESULTS_DIR, FIG_DIR, PRED_DIR,
                       LOG_DIR, ensure_dirs, save_json)

PROB_GRID = np.round(np.linspace(0.05, 0.95, 19), 2)


# ======================================================================
# Prediction (needs PyTorch)
# ======================================================================
def load_trained(name, data):
    import torch
    from src.models import build_model
    path = MODELS_DIR / f"{name}.pth"
    if not path.exists():
        return None
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if not ck.get("finished"):
        print(f"[{name}] checkpoint exists but training did not finish; "
              "it is evaluated as saved (best epoch so far).")
    model = build_model(name, len(data.appliances), data.window)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model


def predict(model, data, idx, batch=4096):
    """Return ON probability and power (kW) for each window."""
    import torch
    probs, powers = [], []
    with torch.no_grad():
        for s in range(0, len(idx), batch):
            X = torch.from_numpy(data.scale_inputs(data.raw_inputs(idx[s:s + batch])))
            logits, p = model(X)
            probs.append(torch.sigmoid(logits).numpy())
            powers.append(p.numpy() * data.power_scale)
    return np.concatenate(probs), np.concatenate(powers)


def run_predictions(data, names, splits):
    done = []
    for name in names:
        model = load_trained(name, data)
        if model is None:
            print(f"[{name}] no checkpoint in models/ -> not evaluated")
            continue
        for split in splits:
            idx = data.windows_for(split)
            prob, pkw = predict(model, data, idx)
            np.savez_compressed(PRED_DIR / f"{name}_{split}.npz", idx=idx,
                                prob=prob.astype(np.float16), power_kw=pkw.astype(np.float32))
            print(f"[{name}] predicted {split}: {len(idx):,} windows")
        done.append(name)
    return done


# ======================================================================
# Metrics (NumPy only)
# ======================================================================
def confusion(y, yhat):
    tp = int(((y == 1) & (yhat == 1)).sum()); tn = int(((y == 0) & (yhat == 0)).sum())
    fp = int(((y == 0) & (yhat == 1)).sum()); fn = int(((y == 1) & (yhat == 0)).sum())
    return tn, fp, fn, tp


def clf_metrics(y, yhat):
    tn, fp, fn, tp = confusion(y, yhat)
    n = tn + fp + fn + tp
    m = {"n": n, "n_on": tp + fn, "n_off": tn + fp, "tn": tn, "fp": fp, "fn": fn, "tp": tp,
         "accuracy": (tp + tn) / n if n else np.nan}
    prec_on = tp / (tp + fp) if tp + fp else 0.0
    rec_on = tp / (tp + fn) if tp + fn else np.nan
    prec_off = tn / (tn + fn) if tn + fn else 0.0
    rec_off = tn / (tn + fp) if tn + fp else np.nan
    m["precision"], m["recall"] = prec_on, rec_on
    if tp + fn == 0 or tn + fp == 0:
        # Only one class in the ground truth: balanced accuracy and macro-F1
        # are not meaningful, so they are left empty instead of misleading.
        m["balanced_accuracy"] = m["macro_f1"] = np.nan
        m["note"] = "only one class present in ground truth"
    else:
        f1_on = 2 * prec_on * rec_on / (prec_on + rec_on) if prec_on + rec_on else 0.0
        f1_off = 2 * prec_off * rec_off / (prec_off + rec_off) if prec_off + rec_off else 0.0
        m["balanced_accuracy"] = (rec_on + rec_off) / 2
        m["macro_f1"] = (f1_on + f1_off) / 2
        m["note"] = ""
    return m


def ground_truth(data, idx):
    """ON labels, target mask and true power (kW) at each window's last minute."""
    A = len(data.appliances)
    y_on = np.zeros((len(idx), A), np.float32)
    mask = np.zeros((len(idx), A), np.float32)
    y_kw = np.zeros((len(idx), A), np.float32)
    for hi in np.unique(idx[:, 0]):
        sel = np.where(idx[:, 0] == hi)[0]
        e = idx[sel, 1]
        y_on[sel] = data.h[hi]["on"][e]
        mask[sel] = data.h[hi]["valid"][e]
        y_kw[sel] = np.nan_to_num(data.h[hi]["power"][e])
    return y_on, mask, y_kw


def tune_prob_threshold(y, prob):
    """Best macro-F1 cut-off on validation data."""
    best_t, best_f = 0.5, -1
    for t in PROB_GRID:
        f = clf_metrics(y, (prob >= t).astype(int))["macro_f1"]
        if np.isfinite(f) and f > best_f:
            best_t, best_f = float(t), f
    return best_t


def evaluate_split(data, idx, prob, pkw, thr, model_label, split, home="all"):
    y_on, mask, y_kw = ground_truth(data, idx)
    rows = []
    for i, a in enumerate(data.appliances):
        v = mask[:, i] > 0
        if v.sum() == 0:
            continue
        y = y_on[v, i].astype(int)
        yhat = (prob[v, i].astype(np.float32) >= thr[a]).astype(int)
        m = clf_metrics(y, yhat)
        err = (pkw[v, i] - y_kw[v, i]) * 1000.0      # watts
        m.update({"model": model_label, "split": split, "home": home, "appliance": a,
                  "appliance_name": APPLIANCE_NAMES[a], "prob_threshold": thr[a],
                  "mae_W": float(np.abs(err).mean()), "rmse_W": float(np.sqrt((err ** 2).mean()))})
        m["meets_min"] = bool(all(np.nan_to_num(m[k]) >= t for k, t in MIN_TARGET.items()))
        m["meets_high"] = bool(all(np.nan_to_num(m[k]) >= t for k, t in HIGH_TARGET.items()))
        rows.append(m)
    return rows


def evaluate_from_predictions(data, names, pred_dir=PRED_DIR, out_dir=RESULTS_DIR,
                              fig_dir=FIG_DIR, labels=None, val_split="val",
                              report_splits=("test", "unseen")):
    labels = labels or MODEL_LABELS
    rows, prob_thr = [], {}
    for name in names:
        f = pred_dir / f"{name}_{val_split}.npz"
        if not f.exists():
            print(f"[{name}] no saved predictions -> skipped")
            continue
        z = np.load(f)
        y_on, mask, _ = ground_truth(data, z["idx"])
        thr = {}
        for i, a in enumerate(data.appliances):
            v = mask[:, i] > 0
            thr[a] = tune_prob_threshold(y_on[v, i].astype(int), z["prob"][v, i].astype(np.float32))
        prob_thr[name] = thr
        for split in (val_split,) + tuple(report_splits):
            fp = pred_dir / f"{name}_{split}.npz"
            if not fp.exists():
                continue
            z = np.load(fp)
            idx = z["idx"]
            rows += evaluate_split(data, idx, z["prob"], z["power_kw"], thr, labels[name], split)
            if split == "unseen":
                for hi in np.unique(idx[:, 0]):
                    s = idx[:, 0] == hi
                    rows += evaluate_split(data, idx[s], z["prob"][s], z["power_kw"][s], thr,
                                           labels[name], split, home=str(data.homes[hi]))
    if not rows:
        print("Nothing to evaluate. Train models first (python -m src.train).")
        return None
    save_json(prob_thr, out_dir / "prob_thresholds.json")
    res = pd.DataFrame(rows)
    front = ["model", "split", "home", "appliance", "appliance_name", "n", "n_on", "n_off",
             "accuracy", "balanced_accuracy", "macro_f1", "precision", "recall",
             "mae_W", "rmse_W", "meets_min", "meets_high", "prob_threshold",
             "tn", "fp", "fn", "tp", "note"]
    res = res[front]
    res.to_csv(out_dir / "results.csv", index=False)

    main_split = report_splits[0] if report_splits else val_split
    comp = res[(res.split == main_split) & (res.home == "all")]
    comp_cols = ["model", "appliance", "accuracy", "balanced_accuracy", "macro_f1",
                 "precision", "recall", "mae_W", "rmse_W", "meets_min", "meets_high"]
    comp[comp_cols].round(4).to_csv(out_dir / "model_comparison.csv", index=False)
    unseen = res[res.split == "unseen"]
    if len(unseen):
        unseen[["home"] + comp_cols + ["n_on", "n_off", "note"]].round(4).to_csv(
            out_dir / "unseen_home_results.csv", index=False)

    best = []
    for a in data.appliances:
        d = comp[comp.appliance == a]
        if d.empty:
            continue
        for metric, better in [("accuracy", "max"), ("balanced_accuracy", "max"),
                               ("macro_f1", "max"), ("mae_W", "min"), ("rmse_W", "min")]:
            r = d.loc[d[metric].idxmax() if better == "max" else d[metric].idxmin()]
            best.append({"appliance": a, "metric": metric, "best_model": r.model,
                         "value": round(float(r[metric]), 4)})
    pd.DataFrame(best).to_csv(out_dir / "best_models.csv", index=False)

    bench = comp[comp.appliance.isin(BENCHMARK_APPLIANCES)].groupby("model").agg(
        all3_meet_min=("meets_min", "all"), all3_meet_high=("meets_high", "all"),
        appliances_meeting_min=("meets_min", "sum")).reset_index()
    bench.to_csv(out_dir / "benchmark_summary.csv", index=False)

    print("\n=== Comparison on", main_split, "===")
    print(comp[comp_cols].round(3).to_string(index=False))
    print("\n", bench.to_string(index=False))

    make_figures(data, res, names, pred_dir, fig_dir, labels, main_split)
    return res


# ======================================================================
# Figures
# ======================================================================
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


COLORS = ["#2563eb", "#0d9488", "#16a34a", "#f59e0b", "#9333ea", "#dc2626"]


def choose_plot_period(data, idx, a_i, span=2 * 1440):
    """Pick a 2-day stretch in the evaluation set with the best balance of ON
    and OFF minutes for this appliance. Chosen from ground truth only, so it is
    the same period for every model."""
    y_on, mask, _ = ground_truth(data, idx)
    on = (y_on[:, a_i] * mask[:, a_i]) > 0
    best = None
    for hi in np.unique(idx[:, 0]):
        sel = np.where(idx[:, 0] == hi)[0]
        e = idx[sel, 1]
        order = np.argsort(e)
        e, onh = e[order], on[sel][order]
        c = np.concatenate([[0], np.cumsum(onh)])
        right = np.searchsorted(e, e + span)
        left = np.arange(len(e))
        n_on = c[right] - c[left]
        n_off = (right - left) - n_on
        score = np.minimum(n_on, n_off)      # most balanced mix of ON and OFF
        j = int(np.argmax(score))
        if best is None or score[j] > best[0]:
            best = (score[j], hi, e[j], e[j] + span)
    return best[1:]


def make_figures(data, res, names, pred_dir, fig_dir, labels, split):
    plt = _plt()
    fig_dir.mkdir(parents=True, exist_ok=True)
    names = [n for n in names if (pred_dir / f"{n}_{split}.npz").exists()]
    comp = res[(res.split == split) & (res.home == "all")]
    apps = data.appliances

    # 1. training vs validation loss
    logs = [(n, LOG_DIR / f"{n}_log.csv") for n in names if (LOG_DIR / f"{n}_log.csv").exists()]
    if logs:
        fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharey=False)
        for ax, (n, f) in zip(axes.flat, logs):
            h = pd.read_csv(f)
            ax.plot(h.epoch, h.train_loss, "o-", label="training", color="#2563eb")
            ax.plot(h.epoch, h.val_loss, "s-", label="validation", color="#0d9488")
            ax.axvline(h.loc[h.val_loss.idxmin(), "epoch"], ls=":", color="grey")
            ax.set_title(labels[n]); ax.set_xlabel("epoch"); ax.set_ylabel("loss")
            ax.legend()
        for ax in list(axes.flat)[len(logs):]:
            ax.axis("off")
        fig.suptitle("Training vs validation loss (dotted line = best epoch)")
        fig.tight_layout(); fig.savefig(fig_dir / "loss_curves_all_models.png", dpi=120)
        plt.close(fig)

    # 2. confusion matrices
    for n in names:
        fig, axes = plt.subplots(1, len(apps), figsize=(3.6 * len(apps), 3.4))
        for ax, a in zip(np.atleast_1d(axes), apps):
            r = comp[(comp.model == labels[n]) & (comp.appliance == a)]
            if r.empty:
                ax.axis("off"); continue
            r = r.iloc[0]
            cm = np.array([[r.tn, r.fp], [r.fn, r.tp]])
            ax.imshow(cm / cm.sum(axis=1, keepdims=True).clip(1), cmap="Blues", vmin=0, vmax=1)
            for (i, j), v in np.ndenumerate(cm):
                ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=9)
            ax.set_xticks([0, 1], ["OFF", "ON"]); ax.set_yticks([0, 1], ["OFF", "ON"])
            ax.set_xlabel("predicted"); ax.set_ylabel("actual")
            ax.set_title(APPLIANCE_NAMES[a], fontsize=10)
        fig.suptitle(f"{labels[n]}: confusion matrices ({split})")
        fig.tight_layout(); fig.savefig(fig_dir / f"confusion_{n}.png", dpi=120); plt.close(fig)

    # 3 and 4. actual vs predicted power and ON/OFF, same period for all models
    preds = {n: np.load(pred_dir / f"{n}_{split}.npz") for n in names}
    idx = preds[names[0]]["idx"] if names else None
    for i, a in enumerate(apps):
        if idx is None:
            break
        hi, t0, t1 = choose_plot_period(data, idx, i)
        H = data.h[hi]
        t = pd.to_datetime(H["ts"][t0:t1], unit="s", utc=True).tz_convert("America/Chicago")
        actual = H["power"][t0:t1, i] * 1000
        fig, axes = plt.subplots(len(names), 1, figsize=(13, 2.1 * len(names)), sharex=True)
        fig2, axes2 = plt.subplots(len(names) + 1, 1, figsize=(13, 1.3 * (len(names) + 1)),
                                   sharex=True)
        axes, axes2 = np.atleast_1d(axes), np.atleast_1d(axes2)
        axes2[0].step(t, H["on"][t0:t1, i], color="black", where="post")
        axes2[0].set_ylabel("actual", rotation=0, ha="right"); axes2[0].set_yticks([0, 1])
        thr_tab = comp.set_index(["model", "appliance"])["prob_threshold"]
        for k, n in enumerate(names):
            z = preds[n]
            s = (z["idx"][:, 0] == hi) & (z["idx"][:, 1] >= t0) & (z["idx"][:, 1] < t1)
            e = z["idx"][s, 1]; order = np.argsort(e); e = e[order]
            tp = pd.to_datetime(H["ts"][e], unit="s", utc=True).tz_convert("America/Chicago")
            ax = axes[k]
            ax.plot(t, actual, color="black", lw=1, label="actual (ground truth)")
            ax.plot(tp, z["power_kw"][s, i][order] * 1000, color=COLORS[k % 6], lw=1,
                    label="predicted")
            ax.set_ylabel("W"); ax.set_title(labels[n], fontsize=9, loc="left")
            ax.legend(loc="upper right", fontsize=8)
            thr = float(thr_tab.get((labels[n], a), 0.5))
            on_pred = (z["prob"][s, i][order].astype(np.float32) >= thr).astype(int)
            axes2[k + 1].step(tp, on_pred, color=COLORS[k % 6], where="post")
            axes2[k + 1].set_ylabel(labels[n], rotation=0, ha="right")
            axes2[k + 1].set_yticks([0, 1])
        fig.suptitle(f"{APPLIANCE_NAMES[a]}: actual vs predicted power, home "
                     f"{data.homes[hi]} ({split} data)")
        fig.tight_layout(); fig.savefig(fig_dir / f"actual_vs_pred_power_{a}.png", dpi=120)
        fig2.suptitle(f"{APPLIANCE_NAMES[a]}: actual vs predicted ON/OFF, home {data.homes[hi]}")
        fig2.tight_layout(); fig2.savefig(fig_dir / f"onoff_actual_vs_pred_{a}.png", dpi=120)
        plt.close(fig); plt.close(fig2)

    # 5. six-model comparison heatmap
    if not comp.empty:
        piv = comp.pivot_table(index="model", columns="appliance",
                               values=["accuracy", "balanced_accuracy", "macro_f1"])
        order = [labels[n] for n in names if labels[n] in piv.index]
        piv = piv.loc[order]
        fig, ax = plt.subplots(figsize=(1.0 * piv.shape[1] + 3, 0.6 * len(piv) + 2))
        ax.imshow(piv.values, cmap="viridis", vmin=0.5, vmax=1, aspect="auto")
        for (r, c), v in np.ndenumerate(piv.values):
            ax.text(c, r, f"{v:.2f}" if np.isfinite(v) else "-", ha="center", va="center",
                    color="white" if (np.isfinite(v) and v < 0.8) else "black", fontsize=8)
        ax.set_yticks(range(len(piv)), piv.index)
        ax.set_xticks(range(piv.shape[1]),
                      [f"{m.replace('balanced_accuracy', 'bal.acc').replace('accuracy', 'acc')}"
                       f"\n{a}" for m, a in piv.columns], fontsize=7)
        ax.set_title(f"Six-model comparison ({split})")
        fig.tight_layout(); fig.savefig(fig_dir / "six_model_comparison.png", dpi=130)
        plt.close(fig)

    # 6-10. bar charts per metric
    specs = [("accuracy", "Accuracy", "accuracy_comparison.png"),
             ("balanced_accuracy", "Balanced accuracy", "balanced_accuracy_comparison.png"),
             ("macro_f1", "Macro-F1", "macro_f1_comparison.png"),
             ("mae_W", "MAE (W)", "mae_comparison.png"),
             ("rmse_W", "RMSE (W)", "rmse_comparison.png")]
    for metric, title, fname in specs:
        if comp.empty:
            break
        fig, ax = plt.subplots(figsize=(11, 4))
        x = np.arange(len(apps)); w = 0.8 / max(len(names), 1)
        for k, n in enumerate(names):
            d = comp[comp.model == labels[n]].set_index("appliance").reindex(apps)
            ax.bar(x + (k - (len(names) - 1) / 2) * w, d[metric], w, label=labels[n],
                   color=COLORS[k % 6])
        if metric in MIN_TARGET:
            ax.axhline(MIN_TARGET[metric], color="#f59e0b", ls="--", label="minimum target")
            ax.axhline(HIGH_TARGET[metric], color="#16a34a", ls=":", label="higher target")
            ax.set_ylim(0.4, 1.0)
        ax.set_xticks(x, [APPLIANCE_NAMES[a] for a in apps])
        ax.set_ylabel(title); ax.set_title(f"{title} per appliance ({split})")
        ax.legend(fontsize=8, ncol=4)
        fig.tight_layout(); fig.savefig(fig_dir / fname, dpi=120); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=MODEL_NAMES, choices=MODEL_NAMES)
    p.add_argument("--no-unseen", action="store_true")
    p.add_argument("--from-saved", action="store_true")
    args = p.parse_args()
    ensure_dirs()
    data = NILMData()
    splits = ["val", "test"] + ([] if args.no_unseen else ["unseen"])
    names = args.models
    if not args.from_saved:
        names = run_predictions(data, names, splits)
    evaluate_from_predictions(data, names)


if __name__ == "__main__":
    main()
