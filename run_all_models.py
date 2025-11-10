#!/usr/bin/env python3
"""
Run all thresholded ML models (KNN, Linear SVM, RBF SVM, Decision Tree)
and evaluate using Accuracy, Precision, Recall, F1, and Balanced Accuracy.

Usage:
  & "E:/Program Files/Python/python.exe" "E:/documents/Thesis code/run_all_models.py" `
    --in "E:/documents/Thesis code/features/all_strokes_combined.csv" `
    --outdir "E:/documents/Thesis code/models_0thres" `
    --threshold 0.0
"""

import argparse, pandas as pd, numpy as np, joblib, seaborn as sns, matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, f1_score, accuracy_score, precision_score,
                             recall_score, balanced_accuracy_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# ---------------- Utility ----------------
def evaluate_model(name, model, X_train, X_test, y_train, y_test, threshold, outdir):
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)
    preds = np.array([model.classes_[i] if p.max() >= threshold else "uncertain"
                      for p, i in zip(probs, probs.argmax(axis=1))])

    mask = preds != "uncertain"
    # Filtered evaluation (ignore uncertain predictions)
    if np.any(mask):
        y_test_f = y_test[mask]
        preds_f  = preds[mask]
        acc = accuracy_score(y_test_f, preds_f)
        prec = precision_score(y_test_f, preds_f, average="macro", zero_division=0)
        rec = recall_score(y_test_f, preds_f, average="macro", zero_division=0)
        f1m = f1_score(y_test_f, preds_f, average="macro", zero_division=0)
        bal = balanced_accuracy_score(y_test_f, preds_f)
    else:
        acc = prec = rec = f1m = bal = 0.0

    cm = confusion_matrix(y_test, preds, labels=np.unique(y_test))
    plt.figure(figsize=(7,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.title(f"{name} Confusion (Threshold={threshold})")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(outdir / f"confusion_{name.lower().replace(' ', '_')}.png", dpi=150)
    plt.close()

    joblib.dump(model, outdir / f"best_{name.lower().replace(' ', '_')}.joblib")

    return {
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1m,
        "Balanced_Accuracy": bal,
        "Threshold": threshold,
        "Accepted (%)": 100*np.mean(mask)
    }

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--threshold", type=float, default=0.6)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.inp)
    if "label" not in df.columns:
        raise ValueError("Missing 'label' column in dataset.")
    X = df.drop(columns=["label"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    models = {
        "KNN": KNeighborsClassifier(n_neighbors=5, weights="distance", p=2),
        "Linear SVM": SVC(kernel="linear", probability=True, C=1),
        "RBF SVM": SVC(kernel="rbf", probability=True, C=2, gamma="scale"),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, min_samples_leaf=3, random_state=42)
    }

    results = []
    for name, model in models.items():
        print(f"\n🧠 Training {name}...")
        res = evaluate_model(name, model, X_train, X_test, y_train, y_test, args.threshold, outdir)
        results.append(res)
        print(f"→ {name}: Acc={res['Accuracy']:.3f}, Prec={res['Precision']:.3f}, "
              f"Rec={res['Recall']:.3f}, F1={res['F1']:.3f}, BalAcc={res['Balanced_Accuracy']:.3f}, "
              f"Accepted={res['Accepted (%)']:.1f}%")

    df_res = pd.DataFrame(results).sort_values(by="F1", ascending=False)
    df_res.to_csv(outdir / "metrics_summary.csv", index=False)
    print("\n✅ Saved metrics summary to:", outdir / "metrics_summary.csv")

    # Optional bar chart summary
    plt.figure(figsize=(10,5))
    ax = sns.barplot(x="Model", y="Accuracy", data=df_res, color="skyblue", label="Accuracy")
    sns.barplot(x="Model", y="F1", data=df_res, color="orange", label="F1")
    plt.ylabel("Score"); plt.title(f"Model Comparison (Threshold={args.threshold})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "model_comparison_bar.png", dpi=150)
    plt.close()

    print("\n📊 Final Metrics:\n", df_res.round(3).to_string(index=False))

if __name__ == "__main__":
    main()
