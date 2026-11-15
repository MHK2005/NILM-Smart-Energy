# NILM Smart Energy

**AI-Powered Non-Intrusive Load Monitoring System**
CT-466 Artificial Neural Networks, Complex Computing Problem, NED University of Engineering & Technology.

## What this project does

A smart meter gives the total power of a home. It does not say which appliance used it.
This project trains six neural networks that read **only the whole-house power** of the
last 180 minutes and predict, for each appliance:

1. whether it is ON or OFF (classification), and
2. how much power it uses (regression).

Appliance circuit readings are used only as the correct answers for training and
evaluation. They are never a model input.

## Dataset

- Region: **Austin, Texas**, Pecan Street 1-minute data, all of 2018 (instructor-provided).
  25 homes, 13.1 million rows, 79 columns. No other region was needed.
- Whole-house input: `aggregate = grid + solar + solar2`. `grid` is net import and goes
  negative when solar exports, so it cannot be used alone.
- Development homes (train / validation / test): **661, 4373, 4767, 6139, 8156**.
  These are the only 5 Austin homes where the AC, dryer and EV charger all have real ON periods.
- Unseen homes (evaluation only): **1642, 7719**.
- Appliances: `air1` (air conditioner), `drye1` (electric dryer), `car1` (EV charging)
  form the CCP benchmark. `refrigerator1` is a fourth, optional appliance (not metered in home 6139).

## Preprocessing (src/data_preprocessing.py)

- Timestamps converted to UTC (the file mixes -06 and -05 offsets because of daylight saving).
- Sorted, duplicates removed, every home reindexed to a continuous 1-minute timeline.
- Gaps of up to 5 minutes are interpolated. Longer gaps stay invalid.
- Appliance readings that are impossible (below -0.05 kW, or above the whole house plus
  0.1 kW) are masked as invalid. They are never set to zero.
- ON/OFF threshold per appliance = max(0.02 kW, 10% of the training 99th percentile),
  from training data only. Results: AC 0.353 kW, dryer 0.270 kW, EV 0.370 kW, fridge 0.035 kW.
- Split: 70 / 15 / 15 in time order **inside every calendar month** (setting `SPLIT_MODE`
  in `src/utils.py`). A single split of the whole year puts the test set in November and
  December, when the AC is ON only 0.6% of the time, so the AC score would measure the
  season instead of the model. Set `SPLIT_MODE = "global"` to use the plain split.
- 180-minute windows (`WINDOW` in `src/utils.py`). A window never crosses a split, a home
  or a long gap.
- Input scaling, power scaling, thresholds and class weights are fitted on training data only.

Class counts and thresholds: `results/class_counts.csv`, `results/thresholds.json`.

## The six models (src/models.py)

| model | feature layers | heads |
|---|---|---|
| MLP (baseline) | Dense 256, 128, 64, dropout 0.2 | ON logit + power per appliance |
| 1D-CNN | Conv 16, Conv 32, MaxPool, Conv 32, MaxPool, Dense 128 | same |
| LSTM | LSTM 64, dropout 0.2, Dense 64 | same |
| GRU | GRU 64, dropout 0.2, Dense 64 | same |
| CNN-LSTM | Conv 16 (k5), MaxPool, Conv 32 (k3), LSTM 32, dropout, Dense 64 | same |
| TCN | causal Conv 16 (dil. 1), causal Conv 32 (dil. 2), dropout, pooling, Dense 64 | same |

All six use `build_model(name, n_appliances, window)` and return `(logits, power)`.
Power goes through softplus, so it is never negative.

## Training (src/train.py)

Adam, learning rate 0.001, batch 512, seed 42, up to 30 epochs, early stopping after 5
epochs without validation improvement. Loss = masked BCEWithLogits (positive class weights
from training labels) + masked Smooth L1 on scaled power. Each model trains on the same fixed
random sample of 200,000 training windows and early-stops on the same 60,000 validation windows.

Saved per model: `models/<name>.pth` (best weights), `models/checkpoints/<name>_last.pth`
(for resuming), `results/training_logs/<name>_log.csv` and `<name>_config.json`.

## Evaluation (src/evaluate.py)

1. Predict validation, test and unseen-home windows.
2. Choose the ON probability cut-off per appliance on **validation** data (best macro-F1).
3. Report on the full **test** set and on the unseen homes: Accuracy, Balanced Accuracy,
   Macro-F1, Precision, Recall, confusion matrix, MAE and RMSE in watts.

CCP targets per appliance: minimum Accuracy >= 85%, Balanced Accuracy >= 75%, Macro-F1 >= 0.70.
Higher: 90%, 85%, 0.80. Tables: `results/model_comparison.csv`, `results/results.csv`,
`results/unseen_home_results.csv`, `results/best_models.csv`, `results/benchmark_summary.csv`.
Figures: `results/figures/`.

## Installation

Python 3.10 or newer.

```bash
pip install -r requirements.txt
```

## Where to put the data

Extract the instructor's archive into `data/raw/` so the CSV ends up at:

```
data/raw/1minute_data_austin/1minute_data_austin.csv
```

The original file is only read, never changed.

## How to run

```bash
python -m src.extract_data          # inspect all homes, extract the 7 selected homes  (~1 min)
python -m src.data_preprocessing    # clean, split, thresholds, windows                 (~30 s)
python -m src.train                 # train all six models                              (CPU: hours)
python -m src.evaluate              # test-set + unseen-home metrics and all figures
python -m src.make_report           # rebuild PROJECT_REPORT.md with the real results
streamlit run app.py                # open the website
```

`data/processed/` already holds the processed files, so you can start from `python -m src.train`.

### Training time and partial runs

Training runs on CPU. The MLP, CNN and TCN take minutes. The LSTM, GRU and CNN-LSTM are
slower because they step through all 180 minutes. You can train in parts:

```bash
python -m src.train --models mlp cnn tcn         # fast models first
python -m src.train --models lstm gru cnn_lstm   # later
python -m src.train --resume                     # continue a model that was stopped mid-way
python -m src.train --models gru --force         # retrain one model from scratch
python -m src.train --max-train-windows 100000   # smaller sample if your PC is slow
python -m src.train --models mlp --epochs 1 --max-train-windows 5000   # 1-minute smoke test
```

Finished models are skipped automatically. If you change `--max-train-windows`, use the same
value for all six models so the comparison stays fair.

`python -m src.evaluate --no-unseen` skips the unseen homes (faster).
`python -m src.evaluate --from-saved` rebuilds tables and figures from saved predictions.

### Optional pipeline check without PyTorch

```bash
python -m src.sanity_check
```

Trains a scikit-learn MLP on the same windows and scores it on validation data only.
It is **not** one of the six models. It only checks that the pipeline works.
Output: `results/sanity_check/`.

## Website (app.py)

Six pages: Home Dashboard, Appliance Monitoring, Model Comparison, Prediction Demo,
Model Evaluation, Historical Analysis. The site loads the saved models and results. It never
retrains. A model without a checkpoint shows a clear "not trained" message. All data is
historical, and every page says so.

## Project structure

```
NILM_Smart_Energy/
  app.py                    Streamlit website
  src/utils.py              all settings (homes, appliances, window, split, training)
  src/extract_data.py       Phase 1: inspection and extraction
  src/data_preprocessing.py Phase 2: cleaning, thresholds, splits, windows
  src/dataset.py            batches (aggregate in, targets + masks out)
  src/models.py             the six architectures
  src/train.py              Phase 4: training
  src/evaluate.py           Phase 5: metrics and figures
  src/make_report.py        builds PROJECT_REPORT.md from the results
  src/app_utils.py          helpers for the website
  src/sanity_check.py       optional check without PyTorch
  data/raw/                 original dataset (not included)
  data/processed/           cleaned per-home arrays, windows, preprocessing.json
  models/                   trained weights
  results/                  tables, logs, figures
```

## Limitations

- One region and one year of data.
- Only 5 Austin homes have all three benchmark appliances, so the development set is small.
- Training uses a random sample of training windows to fit a CPU budget.
- The refrigerator is a small load and is much harder to detect than the others.
- In the unseen homes the dryer is never ON (home 1642) or not metered (home 7719), so only
  accuracy is meaningful for it there.
