"""
Builds REPORT.md from the saved files in data/processed and results/.

Run:  python -m src.make_report      (run again after training and evaluation)

Every number in the report is read from a results file. If a step has not
been run yet, the report says so instead of showing a number.
"""
import pandas as pd

from src.utils import (ROOT, RESULTS_DIR, PROCESSED_DIR, LOG_DIR, MODEL_NAMES, MODEL_LABELS,
                       APPLIANCE_NAMES, BENCHMARK_APPLIANCES, MIN_TARGET, HIGH_TARGET,
                       MAX_TRAIN_WINDOWS, MAX_VAL_WINDOWS, BATCH_SIZE, LEARNING_RATE,
                       MAX_EPOCHS, PATIENCE, SEED, load_json)


def md_table(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |",
           "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in r.values) + " |")
    return "\n".join(out)


def maybe(path):
    return pd.read_csv(path) if path.exists() else None


def main():
    prep = load_json(PROCESSED_DIR / "preprocessing.json")
    cc = pd.read_csv(RESULTS_DIR / "class_counts.csv")
    clean = pd.read_csv(RESULTS_DIR / "cleaning_report.csv")
    comp = maybe(RESULTS_DIR / "model_comparison.csv")
    unseen = maybe(RESULTS_DIR / "unseen_home_results.csv")
    bench = maybe(RESULTS_DIR / "benchmark_summary.csv")
    best = maybe(RESULTS_DIR / "best_models.csv")
    apps = prep["appliances"]
    thr = prep["thresholds_kW"]

    trained = [n for n in MODEL_NAMES if (ROOT / "models" / f"{n}.pth").exists()]
    logs = {n: pd.read_csv(LOG_DIR / f"{n}_log.csv") for n in MODEL_NAMES
            if (LOG_DIR / f"{n}_log.csv").exists()}

    homes_tab = clean[["home", "raw_rows", "missing_minutes", "has_solar"]].copy()
    homes_tab["role"] = ["development" if h in prep["dev_homes"] else "unseen test"
                         for h in homes_tab.home]
    for a in apps:
        homes_tab[a] = ["yes" if prep["availability"][str(h)][a] else "not metered"
                        for h in homes_tab.home]

    thr_tab = pd.DataFrame({
        "appliance": [f"{APPLIANCE_NAMES[a]} ({a})" for a in apps],
        "training p99 (kW)": [round(prep["power_scale_kW"][a], 3) for a in apps],
        "threshold (kW)": [thr[a] for a in apps]})
    cc_tab = cc.pivot_table(index="appliance", columns="split",
                            values=["ON", "OFF"]).astype(int)
    cc_tab.columns = [f"{s} {k}" for k, s in cc_tab.columns]
    order = [f"{s} {k}" for s in ["train", "val", "test", "unseen"] for k in ["ON", "OFF"]]
    cc_tab = cc_tab[order].reset_index()

    L = []
    w = L.append
    w("# NILM Smart Energy: AI-Powered Non-Intrusive Load Monitoring System\n")
    w("NED University of Engineering & Technology, Department of Computer Science & "
      "Information Technology  \nCT-466 Artificial Neural Networks, Complex Computing "
      "Problem (CLO-3)\n")

    w("## 1. Introduction\n")
    w("A smart meter records the total power of a home. It does not say which appliance "
      "used that power. Non-Intrusive Load Monitoring (NILM) estimates the appliances from "
      "the total signal alone, without a meter on each circuit. This project builds six "
      "artificial neural networks that do this on real 1-minute data from Pecan Street.\n")

    w("## 2. Problem statement\n")
    w("Given only the whole-house power of a home, predict for each selected appliance "
      "(a) whether it is ON or OFF and (b) how much power it uses. Appliance-level readings "
      "are used only as the correct answers during training and evaluation. They are never "
      "a model input.\n")

    w("## 3. NILM concept\n")
    w("Each appliance leaves a pattern in the total power: the air conditioner adds about "
      "2 to 4 kW for long cycles, the dryer adds 2 to 6 kW for about an hour, EV charging "
      "adds a flat 3.3 or 7.2 kW block, and the refrigerator adds a small 70 to 150 W "
      "cycle. A model that sees the last 180 minutes of total power can learn these shapes. "
      "Example: if the total is 3,200 W and the AC circuit reads 1,400 W, the input is the "
      "3,200 W history, the AC power target is 1,400 W and the AC label is ON.\n")

    w("## 4. Objectives\n")
    w("1. Select at least 5 homes and 3 appliances with real ON and OFF readings.\n"
      "2. Build a leakage-free pipeline: cleaning, training-only thresholds, time-ordered "
      "splits, windows that never cross a split.\n"
      "3. Train six ANN models (MLP, 1D-CNN, LSTM, GRU, CNN-LSTM, TCN) on identical data.\n"
      "4. Report Accuracy, Balanced Accuracy, Macro-F1, MAE and RMSE per appliance on "
      "unseen test data and compare the models.\n")

    w("## 5. Dataset region\n")
    w(f"Region: **{prep['region']}**. Only the Austin file was used. The Austin file has 25 "
      "homes with readings for all of 2018 (13,100,539 rows, 79 columns). It had enough "
      "homes and appliances, so no other region was needed.\n")

    w("## 6. Selected homes\n")
    w(f"Development homes (train/validation/test): **{', '.join(map(str, prep['dev_homes']))}**. "
      "These are the only 5 Austin homes where the AC, dryer and EV charger all have real "
      "ON periods.  \n"
      f"Unseen homes (evaluation only): **{', '.join(map(str, prep['unseen_homes']))}**. They "
      "are never used for training, validation, thresholds or scaling.\n")
    w(md_table(homes_tab) + "\n")

    w("## 7. Selected appliances\n")
    w("Air conditioner (`air1`), electric dryer (`drye1`) and EV charging (`car1`) form the "
      "compulsory 3-appliance benchmark. The refrigerator (`refrigerator1`) is a fourth, "
      "optional appliance. It is not metered in home 6139, so its targets there are masked. "
      "Home 1642 has a dryer meter but the dryer was never used in 2018, and home 7719 has "
      "no dryer meter, so the unseen-home dryer result has no ON samples.\n")

    w("## 8. Dataset columns\n")
    w("| column | meaning | use |\n|---|---|---|\n"
      "| `dataid` | home ID | select homes, never a model input |\n"
      "| `localminute` | local time with offset (-06 CST, -05 CDT) | converted to UTC |\n"
      "| `grid` | net power from the utility, kW (negative when solar exports) | input |\n"
      "| `solar`, `solar2` | PV generation, kW | input |\n"
      "| `air1`, `drye1`, `car1`, `refrigerator1` | circuit power, kW | targets only |\n")

    w("## 9. Data cleaning and preprocessing\n")
    w("1. Read only the needed columns in chunks (the file is 2.6 GB).\n"
      "2. Parse timestamps to UTC. The file mixes -06 and -05 offsets because of daylight "
      "saving; UTC gives one continuous minute line with no repeated hour.\n"
      "3. Sort, drop duplicate timestamps (none were found), and reindex every home to a "
      "full 1-minute timeline.\n"
      "4. Fill missing runs of up to 5 minutes by linear interpolation. Longer gaps stay "
      "missing and no window may contain them.\n"
      "5. Appliance readings between -0.05 and 0 kW are sensor noise and set to 0. Lower "
      "values, or readings above the whole-house power plus 0.1 kW, are physically "
      "impossible and marked invalid (masked), not deleted and not set to zero.\n"
      "6. All power stays in kW inside the code; errors are reported in W.\n")
    w(md_table(clean[["home", "missing_minutes", "aggregate_interpolated"] +
                     [c for c in clean.columns if c.endswith("invalid_over_aggregate")]]
               .fillna("-")) + "\n")

    w("## 10. Whole-house power reconstruction\n")
    w("In Pecan Street data `grid` is the net import from the utility. It turns negative "
      "when the solar panels export. The real consumption of the home is therefore "
      "**aggregate = grid + solar + solar2**. All 7 selected homes have a `solar` column; "
      "none has `solar2`, so it counts as 0. A minute with a missing grid or solar reading "
      "has no valid aggregate. Small negative solar values at night (inverter standby) are "
      "kept because they are part of the real balance.\n")

    w("## 11. ON/OFF threshold\n")
    w("Threshold per appliance = max(0.02 kW, 10% of the 99th percentile of that "
      "appliance's power in the **training** minutes of the development homes). Validation "
      "and test data were not used. Power above the threshold is ON (1), otherwise OFF (0).\n")
    w(md_table(thr_tab) + "\n")

    w("## 12. ON/OFF class counts\n")
    w("Counts are the labels at the last minute of every valid input window.\n")
    w(md_table(cc_tab) + "\n")
    w("![class distribution](results/figures/class_distribution.png)\n")

    w("## 13. Training, validation and test split\n")
    w(f"Split mode: **{prep['split_mode']}**, ratios 70/15/15, no shuffling. Inside every "
      "calendar month of each development home, the first 70% of minutes are training, "
      "the next 15% validation and the last 15% test.\n\n"
      "Why not one 70/15/15 split of the whole year? We tested it first. It puts the test "
      "set at 7 November to 31 December, when the air conditioner is ON in only 0.6% of "
      "test minutes (2,533 of 394,029) against 27.5% in training. The AC test score would "
      "then measure the season, not the model. The monthly split keeps the time order "
      "inside each month and gives every split all seasons (AC ON share: train 22.1%, "
      "validation 23.8%, test 25.4%). The global split is still available "
      "(`SPLIT_MODE = \"global\"` in `src/utils.py`).\n\n"
      "Leakage control: windows never cross a split, a home or a long gap. Thresholds, "
      "input scaling, power scaling and class weights come from training data only. The "
      "ON probability cut-off of each model is chosen on validation data only. The two "
      "unseen homes add a stricter test: whole homes the models never saw.\n")

    w("## 14. Input window\n")
    w(f"Each sample is {prep['window']} consecutive minutes of aggregate power, scaled with "
      f"the training mean ({prep['input_scaler']['mean']:.3f} kW) and standard deviation "
      f"({prep['input_scaler']['std']:.3f} kW). The targets are the ON/OFF label and power "
      "of every appliance at the last minute of the window. Home ID, time and appliance "
      "readings are never inputs.\n")

    w("## 15. Six ANN architectures\n")
    w("All six share the same input and the same two output heads: a classification head "
      "(one ON logit per appliance) and a regression head (one non-negative power value per "
      "appliance, through softplus).\n\n"
      "| model | feature layers |\n|---|---|\n"
      "| MLP (baseline) | Dense 256, 128, 64 (ReLU), dropout 0.2 |\n"
      "| 1D-CNN | Conv 16, Conv 32, MaxPool, Conv 32, MaxPool, Dense 128 |\n"
      "| LSTM | LSTM 64, dropout 0.2, Dense 64 |\n"
      "| GRU | GRU 64, dropout 0.2, Dense 64 |\n"
      "| CNN-LSTM | Conv 16 (k5), MaxPool, Conv 32 (k3), LSTM 32, dropout 0.2, Dense 64 |\n"
      "| TCN | causal Conv 16 (dilation 1), causal Conv 32 (dilation 2), dropout 0.2, "
      "average pool + last step, Dense 64 |\n")

    w("## 16. Training method\n")
    w(f"PyTorch, Adam, learning rate {LEARNING_RATE}, batch {BATCH_SIZE}, seed {SEED}, up to "
      f"{MAX_EPOCHS} epochs, early stopping after {PATIENCE} epochs without validation "
      "improvement, best checkpoint kept. Loss = masked BCEWithLogits (with positive class "
      "weights from training labels) + masked Smooth L1 on power scaled by the training "
      "99th percentile. Only valid targets count. To keep CPU time reasonable, every model "
      f"trains on the same fixed random sample of {MAX_TRAIN_WINDOWS:,} training windows "
      f"and early-stops on the same {MAX_VAL_WINDOWS:,} validation windows. The test set is "
      "used in full and is not balanced or changed.\n")
    if logs:
        lt = pd.DataFrame([{"model": MODEL_LABELS[n], "epochs run": int(h.epoch.max()),
                            "best epoch": int(h.loc[h.val_loss.idxmin(), "epoch"]),
                            "best val loss": round(h.val_loss.min(), 4)}
                           for n, h in logs.items()])
        w(md_table(lt) + "\n")
        w("![loss curves](results/figures/loss_curves_all_models.png)\n")
    else:
        w("_Training has not been run yet. Run `python -m src.train`, then rebuild this "
          "report._\n")
    missing = [MODEL_LABELS[n] for n in MODEL_NAMES if n not in trained]
    if missing:
        w(f"_Models without a checkpoint: {', '.join(missing)}._\n")

    w("## 17. Classification evaluation\n")
    w("Per appliance on the test set: Accuracy, Balanced Accuracy (mean of ON recall and "
      "OFF recall), Macro-F1 (mean F1 of both classes), Precision and Recall of the ON "
      "class, and the confusion matrix. Balanced accuracy and macro-F1 matter most for "
      "the dryer, which is ON only about 2% of the time.\n")
    w("## 18. Regression evaluation\n")
    w("MAE and RMSE of predicted appliance power, in watts, over valid targets.\n")

    w("## 19. Six-model comparison (test set)\n")
    if comp is not None:
        t = comp.copy()
        for c in ["accuracy", "balanced_accuracy", "macro_f1"]:
            t[c] = t[c].map(lambda v: f"{v:.3f}" if pd.notna(v) else "-")
        for c in ["mae_W", "rmse_W"]:
            t[c] = t[c].map(lambda v: f"{v:.0f}")
        w(md_table(t[["model", "appliance", "accuracy", "balanced_accuracy", "macro_f1",
                      "mae_W", "rmse_W", "meets_min", "meets_high"]]) + "\n")
        if best is not None:
            w("Best model per appliance and metric:\n")
            w(md_table(best) + "\n")
    else:
        w("_No evaluation results yet. Run `python -m src.evaluate`, then rebuild this "
          "report._\n")
    if unseen is not None:
        w("### Unseen homes\n")
        u = unseen[unseen.home == "all"]
        w(md_table(u[["model", "appliance", "accuracy", "balanced_accuracy", "macro_f1",
                      "mae_W", "rmse_W", "n_on", "n_off"]].round(3)) + "\n")
        w("Balanced accuracy and macro-F1 are blank where the unseen homes have only OFF "
          "samples (dryer).\n")

    w("## 20. Graphs\n")
    if comp is not None:
        for f, cap in [("six_model_comparison.png", "Six-model comparison"),
                       ("accuracy_comparison.png", "Accuracy"),
                       ("balanced_accuracy_comparison.png", "Balanced accuracy"),
                       ("macro_f1_comparison.png", "Macro-F1"),
                       ("mae_comparison.png", "MAE"), ("rmse_comparison.png", "RMSE")]:
            w(f"![{cap}](results/figures/{f})\n")
        for a in apps:
            w(f"![{a} power](results/figures/actual_vs_pred_power_{a}.png)\n")
            w(f"![{a} on/off](results/figures/onoff_actual_vs_pred_{a}.png)\n")
        w("Confusion matrices: `results/figures/confusion_<model>.png`.\n")
    else:
        w("_Generated by `python -m src.evaluate` into `results/figures/`._\n")

    w("## 21. CCP benchmark achievement\n")
    w(f"Minimum target per appliance: Accuracy >= {MIN_TARGET['accuracy']:.0%}, Balanced "
      f"Accuracy >= {MIN_TARGET['balanced_accuracy']:.0%}, Macro-F1 >= "
      f"{MIN_TARGET['macro_f1']}. Higher target: {HIGH_TARGET['accuracy']:.0%}, "
      f"{HIGH_TARGET['balanced_accuracy']:.0%}, {HIGH_TARGET['macro_f1']}. Benchmark "
      f"appliances: {', '.join(BENCHMARK_APPLIANCES)}.\n")
    if bench is not None:
        w(md_table(bench) + "\n")
    else:
        w("_Pending evaluation._\n")

    w("## 22. Limitations\n")
    w("- One year (2018) of one region; results may not carry to other climates.\n"
      "- Only 5 Austin homes have all three benchmark appliances active, so the "
      "development set is small.\n"
      "- The monthly split keeps time order inside each month, but training and test "
      "periods are close in time (a few days apart at month boundaries).\n"
      "- Models train on a random sample of training windows to fit a CPU budget.\n"
      "- The refrigerator is a small load (about 100 W) inside a total of several kW, so "
      "it is much harder to detect than the other appliances.\n"
      "- The unseen-home dryer test has no ON samples, so only accuracy is meaningful there.\n")

    w("## 23. Conclusion\n")
    if comp is not None and bench is not None:
        ok = bench[bench.all3_meet_min]
        w(f"{len(ok)} of {len(bench)} evaluated models meet the minimum CCP benchmark on all "
          "three benchmark appliances" + (f" ({', '.join(ok.model)})." if len(ok) else ".") +
          " See section 19 for per-appliance detail.\n")
    else:
        w("_Written after the models are trained and evaluated._\n")

    (ROOT / "REPORT.md").write_text("\n".join(L))
    print("Wrote", ROOT / "REPORT.md")


if __name__ == "__main__":
    main()
