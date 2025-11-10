#!/usr/bin/env python3
import argparse, joblib, numpy as np, pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             precision_recall_fscore_support, confusion_matrix)
from sklearn.ensemble import RandomForestClassifier

# ---------------- helpers ----------------
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

def plot_confusion(cm, classes, title, outpath):
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

def plot_importance(importances, feature_names, outpath_png, outpath_csv, top_k=20):
    idx = np.argsort(importances)[::-1][:top_k]
    top_feats = [(feature_names[i], float(importances[i])) for i in idx]
    # CSV
    pd.DataFrame(top_feats, columns=["feature","importance"]).to_csv(outpath_csv, index=False)
    # Plot
    names = [f for f,_ in top_feats][::-1]
    vals  = [v for _,v in top_feats][::-1]
    plt.figure(figsize=(8,6))
    plt.barh(names, vals)
    plt.xlabel("Gini Importance"); plt.title(f"Top {top_k} Feature Importances (Random Forest)")
    plt.tight_layout(); plt.savefig(outpath_png, dpi=150); plt.close()

def evaluate_with_threshold(model, X_test, y_test, classes, threshold, outdir, tag="rf"):
    """RF supports predict_proba => apply uncertainty threshold."""
    probs = model.predict_proba(X_test)
    pmax = probs.max(axis=1)
    y_idx = probs.argmax(axis=1)
    preds = np.array([classes[i] if pm >= threshold else "uncertain" for pm, i in zip(pmax, y_idx)])

    # metrics ignoring 'uncertain'
    mask = preds != "uncertain"
    if mask.sum() == 0:
        acc = bacc = prec = rec = f1m = 0.0
        cm = np.zeros((len(classes), len(classes)), dtype=int)
    else:
        acc  = accuracy_score(y_test[mask], preds[mask])
        bacc = balanced_accuracy_score(y_test[mask], preds[mask])
        prec, rec, f1m, _ = precision_recall_fscore_support(
            y_test[mask], preds[mask], average="macro", zero_division=0
        )
        cm = confusion_matrix(y_test[mask], preds[mask], labels=classes)

    plot_confusion(cm, classes, f"Confusion (Random Forest) thr={threshold}", outdir / f"confusion_{tag}.png")
    return {
        "model": "random_forest",
        "threshold": threshold,
        "uncertain_rate": float((preds == "uncertain").mean()),
        "test_accuracy": float(acc),
        "test_balanced_accuracy": float(bacc),
        "test_precision_macro": float(prec),
        "test_recall_macro": float(rec),
        "test_f1_macro": float(f1m),
    }

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser(description="Train Random Forest with optional X-axis emphasis/X-only and uncertainty threshold.")
    ap.add_argument("--in", dest="inp", required=True, help="Combined features CSV with 'label'")
    ap.add_argument("--outdir", default="./rf_artifacts", help="Output dir for model/plots/metrics")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--x_weight", type=float, default=1.0, help="Multiply features starting with 'x_'")
    ap.add_argument("--x_only", action="store_true", help="Use only features starting with 'x_'")
    ap.add_argument("--threshold", type=float, default=0.6, help="uncertainty threshold for predict_proba")
    ap.add_argument("--tune", action="store_true", help="Run GridSearchCV over RF hyperparameters")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # Load
    df = pd.read_csv(args.inp)
    if "session_id" in df.columns:
        df = df.drop(columns=["session_id"])
    assert "label" in df.columns, "Input must have 'label' column."
    y = df["label"].astype(str).values
    X = df.drop(columns=["label"])

    # feature selection/emphasis
    if args.x_only:
        X = select_x_only(X)
    X = emphasize_x_features(X, args.x_weight)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=args.test_size, random_state=args.random_state
    )
    classes = np.unique(y_train)

    # Model
    base_rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=1,
        class_weight="balanced_subsample",
        random_state=args.random_state,
        n_jobs=-1,
    )

    if args.tune:
        param_grid = {
            "n_estimators": [300, 500, 800],
            "max_depth": [None, 10, 20, 40],
            "min_samples_leaf": [1, 2, 3],
            "max_features": ["sqrt", "log2", 0.5],
        }
        # GridSearch across a clone of RF; build by hand to keep code simple
        gs = GridSearchCV(
            estimator=RandomForestClassifier(
                class_weight="balanced_subsample",
                random_state=args.random_state,
                n_jobs=-1
            ),
            param_grid=param_grid,
            cv=5, n_jobs=-1, scoring="f1_macro"
        )
        gs.fit(X_train, y_train)
        rf = gs.best_estimator_
        pd.DataFrame(gs.cv_results_).to_csv(outdir / "gridcv_rf_results.csv", index=False)
        print(f"Best RF params: {gs.best_params_}")
    else:
        rf = base_rf.fit(X_train, y_train)

    # Save model
    joblib.dump(rf, outdir / "best_random_forest.joblib")

    # Evaluate with threshold
    metrics = evaluate_with_threshold(rf, X_test, y_test, classes, args.threshold, outdir, tag="rf")
    pd.DataFrame([metrics]).to_csv(outdir / "metrics_summary.csv", index=False)
    print("\n=== Metrics (Random Forest) ===")
    for k,v in metrics.items():
        print(f"{k}: {v}")

    # Feature importance
    plot_importance(rf.feature_importances_, X.columns, outdir/"rf_feature_importance.png", outdir/"rf_feature_importance.csv", top_k=20)
    print(f"\nSaved artifacts → {outdir.resolve()}")

if __name__ == "__main__":
    main()

"""
no x weight
& "E:\Program Files\Python\python.exe" "E:\documents\Thesis code\random_foresttrain.py" `
  --in "E:\documents\Thesis code\features\all_strokes_combined.csv" `
  --outdir "E:\documents\Thesis code\rf_artifactsrandom"

  w x weight
  & "E:\Program Files\Python\python.exe" "E:\documents\Thesis code\random_foresttrain.py" `
  --in "E:/documents/Thesis code/features/all_strokes_combined.csv" `
  --outdir "E:\documents\Thesis code\rf_artifacts_x15random" `
  --x_weight 1.5
"""