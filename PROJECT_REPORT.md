# NILM Smart Energy: AI-Powered Non-Intrusive Load Monitoring System

NED University of Engineering & Technology, Department of Computer Science & Information Technology  
CT-466 Artificial Neural Networks, Complex Computing Problem (CLO-3)

## 1. Introduction

A smart meter records the total power of a home. It does not say which appliance used that power. Non-Intrusive Load Monitoring (NILM) estimates the appliances from the total signal alone, without a meter on each circuit. This project builds six artificial neural networks that do this on real 1-minute data from Pecan Street.

## 2. Problem statement

Given only the whole-house power of a home, predict for each selected appliance (a) whether it is ON or OFF and (b) how much power it uses. Appliance-level readings are used only as the correct answers during training and evaluation. They are never a model input.

## 3. NILM concept

Each appliance leaves a pattern in the total power: the air conditioner adds about 2 to 4 kW for long cycles, the dryer adds 2 to 6 kW for about an hour, EV charging adds a flat 3.3 or 7.2 kW block, and the refrigerator adds a small 70 to 150 W cycle. A model that sees the last 180 minutes of total power can learn these shapes. Example: if the total is 3,200 W and the AC circuit reads 1,400 W, the input is the 3,200 W history, the AC power target is 1,400 W and the AC label is ON.

## 4. Objectives

1. Select at least 5 homes and 3 appliances with real ON and OFF readings.
2. Build a leakage-free pipeline: cleaning, training-only thresholds, time-ordered splits, windows that never cross a split.
3. Train six ANN models (MLP, 1D-CNN, LSTM, GRU, CNN-LSTM, TCN) on identical data.
4. Report Accuracy, Balanced Accuracy, Macro-F1, MAE and RMSE per appliance on unseen test data and compare the models.

## 5. Dataset region

Region: **Austin, Texas (Pecan Street 1-minute, 2018)**. Only the Austin file was used. The Austin file has 25 homes with readings for all of 2018 (13,100,539 rows, 79 columns). It had enough homes and appliances, so no other region was needed.

## 6. Selected homes

Development homes (train/validation/test): **661, 4373, 4767, 6139, 8156**. These are the only 5 Austin homes where the AC, dryer and EV charger all have real ON periods.  
Unseen homes (evaluation only): **1642, 7719**. They are never used for training, validation, thresholds or scaling.

| home | raw_rows | missing_minutes | has_solar | role | air1 | drye1 | car1 | refrigerator1 |
|---|---|---|---|---|---|---|---|---|
| 661 | 525530 | 5050 | True | development | yes | yes | yes | yes |
| 4373 | 518040 | 7620 | True | development | yes | yes | yes | yes |
| 4767 | 524440 | 20 | True | development | yes | yes | yes | yes |
| 6139 | 525368 | 344 | True | development | yes | yes | yes | not metered |
| 8156 | 524820 | 840 | True | development | yes | yes | yes | yes |
| 1642 | 519780 | 8520 | True | unseen test | yes | yes | yes | yes |
| 7719 | 523260 | 2340 | True | unseen test | yes | not metered | yes | yes |

## 7. Selected appliances

Air conditioner (`air1`), electric dryer (`drye1`) and EV charging (`car1`) form the compulsory 3-appliance benchmark. The refrigerator (`refrigerator1`) is a fourth, optional appliance. It is not metered in home 6139, so its targets there are masked. Home 1642 has a dryer meter but the dryer was never used in 2018, and home 7719 has no dryer meter, so the unseen-home dryer result has no ON samples.

## 8. Dataset columns

| column | meaning | use |
|---|---|---|
| `dataid` | home ID | select homes, never a model input |
| `localminute` | local time with offset (-06 CST, -05 CDT) | converted to UTC |
| `grid` | net power from the utility, kW (negative when solar exports) | input |
| `solar`, `solar2` | PV generation, kW | input |
| `air1`, `drye1`, `car1`, `refrigerator1` | circuit power, kW | targets only |

## 9. Data cleaning and preprocessing

1. Read only the needed columns in chunks (the file is 2.6 GB).
2. Parse timestamps to UTC. The file mixes -06 and -05 offsets because of daylight saving; UTC gives one continuous minute line with no repeated hour.
3. Sort, drop duplicate timestamps (none were found), and reindex every home to a full 1-minute timeline.
4. Fill missing runs of up to 5 minutes by linear interpolation. Longer gaps stay missing and no window may contain them.
5. Appliance readings between -0.05 and 0 kW are sensor noise and set to 0. Lower values, or readings above the whole-house power plus 0.1 kW, are physically impossible and marked invalid (masked), not deleted and not set to zero.
6. All power stays in kW inside the code; errors are reported in W.

| home | missing_minutes | aggregate_interpolated | air1_invalid_over_aggregate | drye1_invalid_over_aggregate | car1_invalid_over_aggregate | refrigerator1_invalid_over_aggregate |
|---|---|---|---|---|---|---|
| 661 | 5050 | 0 | 0 | 0.0 | 0 | 21.0 |
| 4373 | 7620 | 0 | 0 | 271.0 | 218 | 10.0 |
| 4767 | 20 | 0 | 0 | 0.0 | 0 | 0.0 |
| 6139 | 344 | 344 | 0 | 0.0 | 1702 | - |
| 8156 | 840 | 0 | 0 | 29.0 | 0 | 0.0 |
| 1642 | 8520 | 0 | 0 | 0.0 | 0 | 0.0 |
| 7719 | 2340 | 0 | 0 | - | 0 | 0.0 |

## 10. Whole-house power reconstruction

In Pecan Street data `grid` is the net import from the utility. It turns negative when the solar panels export. The real consumption of the home is therefore **aggregate = grid + solar + solar2**. All 7 selected homes have a `solar` column; none has `solar2`, so it counts as 0. A minute with a missing grid or solar reading has no valid aggregate. Small negative solar values at night (inverter standby) are kept because they are part of the real balance.

## 11. ON/OFF threshold

Threshold per appliance = max(0.02 kW, 10% of the 99th percentile of that appliance's power in the **training** minutes of the development homes). Validation and test data were not used. Power above the threshold is ON (1), otherwise OFF (0).

| appliance | training p99 (kW) | threshold (kW) |
|---|---|---|
| Air Conditioner (air1) | 3.532 | 0.3532 |
| Electric Dryer (drye1) | 2.704 | 0.2704 |
| EV Charging (car1) | 3.705 | 0.3705 |
| Refrigerator (refrigerator1) | 0.354 | 0.0354 |

## 12. ON/OFF class counts

Counts are the labels at the last minute of every valid input window.

| appliance | train ON | train OFF | val ON | val OFF | test ON | test OFF | unseen ON | unseen OFF |
|---|---|---|---|---|---|---|---|---|
| air1 | 400188 | 1414754 | 90187 | 288625 | 97183 | 286106 | 196986 | 842220 |
| car1 | 144763 | 1620206 | 28049 | 338704 | 30750 | 341400 | 44800 | 994406 |
| drye1 | 35464 | 1778964 | 7939 | 370819 | 9097 | 374100 | 0 | 516483 |
| refrigerator1 | 1034135 | 387984 | 215992 | 75999 | 219717 | 80750 | 913126 | 108674 |

![class distribution](results/figures/class_distribution.png)

## 13. Training, validation and test split

Split mode: **monthly**, ratios 70/15/15, no shuffling. Inside every calendar month of each development home, the first 70% of minutes are training, the next 15% validation and the last 15% test.

Why not one 70/15/15 split of the whole year? We tested it first. It puts the test set at 7 November to 31 December, when the air conditioner is ON in only 0.6% of test minutes (2,533 of 394,029) against 27.5% in training. The AC test score would then measure the season, not the model. The monthly split keeps the time order inside each month and gives every split all seasons (AC ON share: train 22.1%, validation 23.8%, test 25.4%). The global split is still available (`SPLIT_MODE = "global"` in `src/utils.py`).

Leakage control: windows never cross a split, a home or a long gap. Thresholds, input scaling, power scaling and class weights come from training data only. The ON probability cut-off of each model is chosen on validation data only. The two unseen homes add a stricter test: whole homes the models never saw.

## 14. Input window

Each sample is 180 consecutive minutes of aggregate power, scaled with the training mean (1.767 kW) and standard deviation (1.900 kW). The targets are the ON/OFF label and power of every appliance at the last minute of the window. Home ID, time and appliance readings are never inputs.

## 15. Six ANN architectures

All six share the same input and the same two output heads: a classification head (one ON logit per appliance) and a regression head (one non-negative power value per appliance, through softplus).

| model | feature layers |
|---|---|
| MLP (baseline) | Dense 256, 128, 64 (ReLU), dropout 0.2 |
| 1D-CNN | Conv 16, Conv 32, MaxPool, Conv 32, MaxPool, Dense 128 |
| LSTM | LSTM 64, dropout 0.2, Dense 64 |
| GRU | GRU 64, dropout 0.2, Dense 64 |
| CNN-LSTM | Conv 16 (k5), MaxPool, Conv 32 (k3), LSTM 32, dropout 0.2, Dense 64 |
| TCN | causal Conv 16 (dilation 1), causal Conv 32 (dilation 2), dropout 0.2, average pool + last step, Dense 64 |

## 16. Training method

PyTorch, Adam, learning rate 0.001, batch 512, seed 42, up to 30 epochs, early stopping after 5 epochs without validation improvement, best checkpoint kept. Loss = masked BCEWithLogits (with positive class weights from training labels) + masked Smooth L1 on power scaled by the training 99th percentile. Only valid targets count. To keep CPU time reasonable, every model trains on the same fixed random sample of 200,000 training windows and early-stops on the same 60,000 validation windows. The test set is used in full and is not balanced or changed.

_Training has not been run yet. Run `python -m src.train`, then rebuild this report._

_Models without a checkpoint: MLP, 1D-CNN, LSTM, GRU, CNN-LSTM, TCN._

## 17. Classification evaluation

Per appliance on the test set: Accuracy, Balanced Accuracy (mean of ON recall and OFF recall), Macro-F1 (mean F1 of both classes), Precision and Recall of the ON class, and the confusion matrix. Balanced accuracy and macro-F1 matter most for the dryer, which is ON only about 2% of the time.

## 18. Regression evaluation

MAE and RMSE of predicted appliance power, in watts, over valid targets.

## 19. Six-model comparison (test set)

_No evaluation results yet. Run `python -m src.evaluate`, then rebuild this report._

## 20. Graphs

_Generated by `python -m src.evaluate` into `results/figures/`._

## 21. CCP benchmark achievement

Minimum target per appliance: Accuracy >= 85%, Balanced Accuracy >= 75%, Macro-F1 >= 0.7. Higher target: 90%, 85%, 0.8. Benchmark appliances: air1, drye1, car1.

_Pending evaluation._

## 22. Limitations

- One year (2018) of one region; results may not carry to other climates.
- Only 5 Austin homes have all three benchmark appliances active, so the development set is small.
- The monthly split keeps time order inside each month, but training and test periods are close in time (a few days apart at month boundaries).
- Models train on a random sample of training windows to fit a CPU budget.
- The refrigerator is a small load (about 100 W) inside a total of several kW, so it is much harder to detect than the other appliances.
- The unseen-home dryer test has no ON samples, so only accuracy is meaningful there.

## 23. Conclusion

_Written after the models are trained and evaluated._
