#!/usr/bin/env python3
import argparse, joblib, numpy as np, pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_recall_fscore_support,
                             f1_score, confusion_matrix, classification_report)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
"""
& "E:\Program Files\Python\python.exe" "E:\documents\Thesis code\enhanced_x_axis_train.py" `
  --in "E:/documents/Thesis code/features/all_strokes_combined.csv" `
  --outdir "E:\documents\Thesis code\artifacts_x15" `
  --x_weight 1.5

  & "E:\Program Files\Python\python.exe" "E:\documents\Thesis code\train_all_with_xweight.py" `
  --in "E:\documents\Thesis code\features\combined_features.csv" `
  --outdir "E:\documents\Thesis code\artifacts_xonly" `
  --x_only

"""
# ---------- helpers ----------
def emphasize_x_features(df: pd.DataFrame, weight: float) -> pd.DataFrame:
    if weight == 1.0:
        return df
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

def evaluate_with_threshold(model, X_test, y_test, classes, name, outdir, threshold):
    """
    For KNN and RBF-SVM (probability=True), apply 'uncertain' threshold.
    For LinearSVC/DecisionTree (no calibrated proba by default), skip threshold.
    """
    supports_proba = hasattr(model, "predict_proba")
    if supports_proba:
        probs = model.predict_proba(X_test)
        pred_idx = probs.argmax(axis=1)
        preds = np.array([classes[i] if pmax >= threshold else "uncertain"
                          for pmax, i in zip(probs.max(axis=1), pred_idx)])
        # Filter out 'uncertain' when computing standard metrics
        mask = preds != "uncertain"
        if mask.sum() == 0:
            acc = 0.0; bacc = 0.0; prec = rec = f1m = 0.0
            cm = np.zeros((len(classes), len(classes)), dtype=int)
        else:
            acc = accuracy_score(y_test[mask], preds[mask])
            bacc = balanced_accuracy_score(y_test[mask], preds[mask])
            prec, rec, f1m, _ = precision_recall_fscore_support(y_test[mask], preds[mask],
                                                                average="macro", zero_division=0)
            cm = confusion_matrix(y_test[mask], preds[mask], labels=classes)
        # Save confusion matrix (filtered)
        figure_cm(cm, classes, f"Confusion ({name}) threshold={threshold}", outdir / f"confusion_{name}.png")
        extra = {
            "threshold": threshold,
            "uncertain_rate": float((preds == "uncertain").mean())
        }
    else:
        # No proba -> straight predictions
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        bacc = balanced_accuracy_score(y_test, preds)
        prec, rec, f1m, _ = precision_recall_fscore_support(y_test, preds, average="macro", zero_division=0)
        cm = confusion_matrix(y_test, preds, labels=classes)
        figure_cm(cm, classes, f"Confusion ({name})", outdir / f"confusion_{name}.png")
        extra = {"threshold": None, "uncertain_rate": None}

    return {
        "model": name,
        "test_accuracy": acc,
        "test_balanced_accuracy": bacc,
        "test_precision_macro": prec,
        "test_recall_macro": rec,
        "test_f1_macro": f1m,
        **extra
    }

def main():
    ap = argparse.ArgumentParser(description="Train KNN / Linear SVM / RBF SVM / Decision Tree with optional X-axis emphasis.")
    ap.add_argument("--in", dest="inp", required=True, help="Combined features CSV (must include 'label')")
    ap.add_argument("--outdir", default="./artifacts_xweight", help="Where to save models/plots/metrics")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--x_weight", type=float, default=1.0, help="Multiplier for features starting with 'x_'")
    ap.add_argument("--x_only", action="store_true", help="Train using only features starting with 'x_'")
    ap.add_argument("--threshold", type=float, default=0.6, help="Uncertainty threshold (applies to KNN and RBF-SVM only)")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # ---------- load ----------
    df = pd.read_csv(args.inp)
    if "session_id" in df.columns:
        df = df.drop(columns=["session_id"])
    assert "label" in df.columns, "Input CSV must contain a 'label' column."

    # optional x-only
    feature_df = df.drop(columns=["label"])
    if args.x_only:
        feature_df = select_x_only(feature_df)

    # emphasize x features
    feature_df = emphasize_x_features(feature_df, args.x_weight)

    X = feature_df
    y = df["label"].astype(str).values
    classes = np.unique(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=args.test_size, random_state=args.random_state
    )

    # ---------- define models ----------
    models = {}

    # KNN (with scaler)
    models["knn"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=5, weights="distance", p=2))
    ])

    # Linear SVM (no proba)
    models["linear_svm"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LinearSVC(C=1.0))
    ])

    # RBF SVM (with proba)
    models["svm_rbf"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", probability=True, C=1.0, gamma="scale"))
    ])

    # Decision Tree (scaler is harmless; consistent interface)
    models["tree"] = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", DecisionTreeClassifier(random_state=args.random_state, max_depth=None))
    ])

    # ---------- train & evaluate ----------
    rows = []
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        joblib.dump(pipe, outdir / f"best_{name}.joblib")

        # pick correct label order for CM
        eval_classes = classes

        # evaluation with threshold when available
        clf = pipe.named_steps["clf"]
        res = evaluate_with_threshold(pipe, X_test, y_test, eval_classes, name, outdir, args.threshold)
        rows.append(res)

    # ---------- save metrics ----------
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(outdir / "metrics_summary.csv", index=False)

    # brief console summary
    print("\n=== Metrics Summary ===")
    print(metrics_df.to_string(index=False))
    print(f"\nSaved models and plots → {outdir.resolve()}")

if __name__ == "__main__":
    main()
