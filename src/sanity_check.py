"""
Optional pipeline check WITHOUT PyTorch.

Run:  python -m src.sanity_check

Trains a small scikit-learn MLP on the same training windows and pushes its
predictions through the same metric and figure code used by evaluate.py.
It only uses VALIDATION windows: one half to choose the probability cut-off,
the other half to score. The test set is not touched.

This is NOT one of the six project models and its numbers are not project
results. It only shows that the windows, labels and metric code work and
that the aggregate signal carries enough information to learn from.
Outputs go to results/sanity_check/.
"""
import numpy as np
from sklearn.neural_network import MLPClassifier, MLPRegressor

from src.dataset import NILMData
from src.evaluate import evaluate_from_predictions
from src.utils import RESULTS_DIR, SEED, set_seed

OUT = RESULTS_DIR / "sanity_check"


def main(n_train=80_000):
    set_seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    data = NILMData()
    tr = data.windows_for("train", n_train, seed=SEED)
    val = data.windows_for("val", 60_000, seed=SEED + 1)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(val))
    tune, check = val[np.sort(perm[: len(val) // 2])], val[np.sort(perm[len(val) // 2:])]

    X, y_on, y_pow, mask, _ = data.batch(tr)
    y_on = y_on * mask       # masked targets set to 0 for sklearn (stand-in only)
    y_pow = y_pow * mask
    clf = MLPClassifier(hidden_layer_sizes=(256, 128, 64), max_iter=30, random_state=SEED, verbose=True)
    clf.fit(X, y_on.astype(int))
    reg = MLPRegressor(hidden_layer_sizes=(256, 128, 64), max_iter=30, random_state=SEED, verbose=True)
    reg.fit(X, y_pow)

    for name, idx in [("tune", tune), ("check", check)]:
        Xs = data.scale_inputs(data.raw_inputs(idx))
        prob = clf.predict_proba(Xs)
        pkw = np.clip(reg.predict(Xs), 0, None) * data.power_scale
        np.savez_compressed(OUT / f"sk_mlp_{name}.npz", idx=idx,
                            prob=prob.astype(np.float16), power_kw=pkw.astype(np.float32))
    evaluate_from_predictions(data, ["sk_mlp"], pred_dir=OUT, out_dir=OUT,
                              fig_dir=OUT / "figures", labels={"sk_mlp": "sklearn MLP (check)"},
                              val_split="tune", report_splits=("check",))


if __name__ == "__main__":
    main()
