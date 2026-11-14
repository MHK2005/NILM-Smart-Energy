# CT-466 NILM - Pecan Street Austin 1-minute data (MLP vs 1D-CNN)

## Folder layout
- code/        the 5 Python scripts
- results/     results.csv (benchmark table), results_summary.txt, class_counts.csv, prep_meta.json, train_log.txt
- figures/     power_test.png, power_unseen.png, confusion_matrices.png, training_curves.png
- models/      trained weights (model_mlp.pt, model_cnn1d.pt)
- extra_runs/  results of the 60- and 120-minute window experiments (final run = 180 minutes)

## How to run
Put the 5 scripts from code/ in ONE working folder (they read and write files in the current folder), then:
1. python extract_austin.py   (edit the archive path at the top to your 1minute_data_austin tar.gz)
2. set NILM_WINDOW=180        (Windows: set NILM_WINDOW=180 | Linux/Mac/Colab: export NILM_WINDOW=180)
3. python data_prep.py
4. python train_models.py     (~20 min CPU; seed 42)
5. python evaluate.py
Needs: python 3.10+, pandas, numpy, scikit-learn, matplotlib, torch.

Input = whole-house power only (grid + solar, kW). Targets = ON/OFF and power of air1, drye1, car1, refrigerator1.
Split = 70/15/15 chronological per home; homes 4767 and 8565 are held out entirely.
