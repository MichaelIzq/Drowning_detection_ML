#!/usr/bin/env python3
import argparse, joblib, numpy as np, pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_recall_fscore_support, confusion_matrix, classification_report
)

# ---------- helper functions ----------
def emphasize_x_features(df: pd.DataFrame, weight: float) -> pd.DataFrame:
    if weight == 1.0: return df
    df = df.copy()
    for c in df.columns:
        if c.startswith("x_"):
            df[c] = df[c] * weight
    return df

def select_x_only(df: pd.DataFrame) -> pd.DataFrame:
    xcols = [c for c in df.columns if c.startswith("x_")]
    if not xcols:
        raise ValueError("No columns start with 'x_'.")
    return df[xcols]

def plot_confusion(cm, classes, title, outpath, cmap="Blues"):
    fig = plt.figure(figsize=(6,5))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, aspect='auto', cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center')
    fig.tight_layout(); fig.savefig(outpath, dpi=150); plt.close(fig)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Test trained Random Forest with threshold and X-weight options.")
    ap.add_argument("--model", required=True, help="Path to best_random_forest.joblib")
    ap.add_argument("--in", dest="inp", required=True, help="Feature CSV with labels for testing")
    ap.add_argument("--outdir", default="./rf_test_results", help="Output dir for results")
    ap.add_argument("--threshold", type=float, default=0.6, help="Uncertainty threshold on predict_proba")
    ap.add_argument("--x_weight", type=float, default=1.0, help="Multiply x_ features")
    ap.add_argument("--x_only", action="store_true", help="Use only x_ features")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    model = joblib.load(args.model)
    df = pd.read_csv(args.inp)
    if "session_id" in df.columns:
        df = df.drop(columns=["session_id"])
    assert "label" in df.columns, "Input CSV must include 'label'."

    y_true = df["label"].astype(str).values
    X = df.drop(columns=["label"])
    if args.x_only:
        X = select_x_only(X)
    X = emphasize_x_features(X, args.x_weight)

    # --- predict with uncertainty threshold ---
    probs = model.predict_proba(X)
    pmax = probs.max(axis=1)
    y_idx = probs.argmax(axis=1)
    preds = np.array([model.classes_[i] if pm >= args.threshold else "uncertain" for pm, i in zip(pmax, y_idx)])

    mask = preds != "uncertain"
    uncertain_rate = float((preds == "uncertain").mean())

    if mask.sum() == 0:
        acc = bacc = prec = rec = f1m = 0.0
        cm = np.zeros((len(model.classes_), len(model.classes_)), dtype=int)
    else:
        acc  = accuracy_score(y_true[mask], preds[mask])
        bacc = balanced_accuracy_score(y_true[mask], preds[mask])
        prec, rec, f1m, _ = precision_recall_fscore_support(
            y_true[mask], preds[mask], average="macro", zero_division=0
        )
        cm = confusion_matrix(y_true[mask], preds[mask], labels=model.classes_)

    # --- save results ---
    metrics = {
        "threshold": args.threshold,
        "uncertain_rate": uncertain_rate,
        "test_accuracy": acc,
        "test_balanced_accuracy": bacc,
        "test_precision_macro": prec,
        "test_recall_macro": rec,
        "test_f1_macro": f1m,
        "n_samples": len(y_true),
        "n_uncertain": int((preds == "uncertain").sum()),
        "n_valid": int(mask.sum())
    }
    pd.DataFrame([metrics]).to_csv(outdir / "rf_test_metrics.csv", index=False)
    np.savetxt(outdir / "rf_predictions.txt", preds, fmt="%s")
    plot_confusion(cm, model.classes_, "Random Forest - Test Confusion Matrix", outdir/"rf_confusion.png", cmap="Blues")

    # Print summary
    print("\n=== Random Forest Test Summary ===")
    for k,v in metrics.items():
        print(f"{k}: {v}")
    print("\nClassification report (filtered):")
    print(classification_report(y_true[mask], preds[mask], zero_division=0))
    print(f"\nSaved results → {outdir.resolve()}")

if __name__ == "__main__":
    main()

"""
& "E:\Program Files\Python\python.exe" "E:\documents\Thesis code\random_forest_test.py" `
  --model "E:\documents\Thesis code\rf_artifacts_x15random\best_random_forest.joblib" `
  --in "E:/documents/Thesis code/features/all_strokes_combined.csv" `
  --outdir "E:\documents\Thesis code\rf_test_results" `
  --threshold 0.6 --x_weight 1.5
  
  E:/documents/Thesis code/features/all_strokes_combined.csv

  E:\documents\Thesis code\feature_test\combined_testing_sample.csv

  """