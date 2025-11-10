#!/usr/bin/env python3
import argparse, joblib, numpy as np, pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             precision_recall_fscore_support,
                             confusion_matrix)

"""
& "E:\Program Files\Python\python.exe" "E:\documents\Thesis code\test_all_with_xweight.py" `
  --in "E:\documents\Thesis code\feature_test\combined_testing_sample.csv" `
  --models_dir "E:\documents\Thesis code\artifacts_x15" `
  --outdir "E:\documents\Thesis code\reports2_x15" `
  --threshold 0.6 --save_preds
  """

# ---------- helpers ----------
def emphasize_x_features(df: pd.DataFrame, weight: float) -> pd.DataFrame:
    if weight == 1.0:
        return df
    df = df.copy()
    for col in df.columns:
        if col.startswith("x_"):
            df[col] = df[col] * weight
    return df

def select_x_only(df: pd.DataFrame) -> pd.DataFrame:
    xcols = [c for c in df.columns if c.startswith("x_")]
    if not xcols:
        raise ValueError("No columns start with 'x_'. Check your feature names.")
    return df[xcols]

def figure_cm(cm, classes, title, outpath):
    fig = plt.figure(figsize=(6,5))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, aspect='auto', cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center')
    fig.tight_layout(); fig.savefig(outpath, dpi=150); plt.close(fig)

def evaluate_with_threshold(model, X, y, classes, name, outdir, threshold, save_preds_path=None):
    """
    Applies thresholding only if model supports predict_proba.
    Returns metrics dict. Optionally writes per-row predictions CSV.
    """
    supports_proba = hasattr(model, "predict_proba")
    if supports_proba:
        probs = model.predict_proba(X)
        pred_idx = probs.argmax(axis=1)
        pmax = probs.max(axis=1)
        preds = np.array([classes[i] if p >= threshold else "uncertain"
                          for p, i in zip(pmax, pred_idx)])
        # metrics (filter out 'uncertain')
        if y is not None:
            mask = preds != "uncertain"
            if mask.sum() == 0:
                acc = bacc = prec = rec = f1m = 0.0
                cm = np.zeros((len(classes), len(classes)), dtype=int)
            else:
                acc  = accuracy_score(y[mask], preds[mask])
                bacc = balanced_accuracy_score(y[mask], preds[mask])
                prec, rec, f1m, _ = precision_recall_fscore_support(
                    y[mask], preds[mask], average="macro", zero_division=0
                )
                cm = confusion_matrix(y[mask], preds[mask], labels=classes)
            figure_cm(cm, classes, f"Confusion ({name}) threshold={threshold}",
                      outdir / f"confusion_{name}.png")
        else:
            acc = bacc = prec = rec = f1m = None

        # save per-sample predictions if requested
        if save_preds_path is not None:
            dfp = pd.DataFrame({
                "pred": preds,
                "pmax": pmax
            })
            dfp.to_csv(save_preds_path, index=False)

        return {
            "model": name,
            "threshold": threshold,
            "uncertain_rate": float((preds == "uncertain").mean()),
            "test_accuracy": acc if acc is not None else np.nan,
            "test_balanced_accuracy": bacc if bacc is not None else np.nan,
            "test_precision_macro": prec if prec is not None else np.nan,
            "test_recall_macro": rec if rec is not None else np.nan,
            "test_f1_macro": f1m if f1m is not None else np.nan,
        }

    else:
        # No probability: direct predictions, threshold not applicable
        preds = model.predict(X)
        if y is not None:
            acc  = accuracy_score(y, preds)
            bacc = balanced_accuracy_score(y, preds)
            prec, rec, f1m, _ = precision_recall_fscore_support(
                y, preds, average="macro", zero_division=0
            )
            cm = confusion_matrix(y, preds, labels=classes)
            figure_cm(cm, classes, f"Confusion ({name})", outdir / f"confusion_{name}.png")
        else:
            acc = bacc = prec = rec = f1m = None

        if save_preds_path is not None:
            pd.DataFrame({"pred": preds}).to_csv(save_preds_path, index=False)

        return {
            "model": name,
            "threshold": None,
            "uncertain_rate": None,
            "test_accuracy": acc if acc is not None else np.nan,
            "test_balanced_accuracy": bacc if bacc is not None else np.nan,
            "test_precision_macro": prec if prec is not None else np.nan,
            "test_recall_macro": rec if rec is not None else np.nan,
            "test_f1_macro": f1m if f1m is not None else np.nan,
        }

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Test/evaluate saved models with optional X-axis emphasis.")
    ap.add_argument("--in", dest="inp", required=True, help="Features CSV for testing (may include 'label')")
    ap.add_argument("--models_dir", required=True, help="Folder containing best_*.joblib models")
    ap.add_argument("--outdir", default="./test_reports", help="Where to save metrics/plots/preds")
    ap.add_argument("--x_weight", type=float, default=1.0, help="Multiply features starting with 'x_'")
    ap.add_argument("--x_only", action="store_true", help="Use only features starting with 'x_'")
    ap.add_argument("--threshold", type=float, default=0.6, help="Uncertainty threshold for proba models")
    ap.add_argument("--save_preds", action="store_true", help="Save per-sample predictions CSVs")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # Load test features
    df = pd.read_csv(args.inp)
    y = None
    if "session_id" in df.columns:
        df = df.drop(columns=["session_id"])
    if "label" in df.columns:
        y = df["label"].astype(str).values
        X = df.drop(columns=["label"])
        classes = np.unique(y)
    else:
        X = df.copy()
        classes = None  # will be inferred from models where needed

    # Feature selection/emphasis
    if args.x_only:
        X = select_x_only(X)
    X = emphasize_x_features(X, args.x_weight)

    # Load any available models
    models_dir = Path(args.models_dir)
    model_paths = {
        "knn": models_dir / "best_knn.joblib",
        "linear_svm": models_dir / "best_linear_svm.joblib",
        "svm_rbf": models_dir / "best_svm_rbf.joblib",
        "tree": models_dir / "best_tree.joblib",
    }
    available = {name: p for name, p in model_paths.items() if p.exists()}
    if not available:
        raise FileNotFoundError(f"No best_*.joblib models found in {models_dir}")

    rows = []
    for name, path in available.items():
        model = joblib.load(path)

        # infer classes from model if needed (for unlabeled test only)
        model_classes = None
        try:
            model_classes = model.named_steps["clf"].classes_
        except Exception:
            pass
        used_classes = classes if classes is not None else (model_classes if model_classes is not None else None)

        # predictions output path
        preds_path = None
        if args.save_preds:
            preds_path = outdir / f"preds_{name}.csv"

        res = evaluate_with_threshold(
            model, X, y, used_classes, name, outdir, args.threshold, save_preds_path=preds_path
        )
        rows.append(res)

    # Save metrics table
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(outdir / "metrics_summary_test.csv", index=False)

    print("\n=== Test Metrics Summary ===")
    print(metrics_df.to_string(index=False))
    print(f"\nSaved reports → {outdir.resolve()}")

if __name__ == "__main__":
    main()
