#!/usr/bin/env python3
"""
Evaluate a trained model on a labeled dataset and print accuracy metrics.

Usage example:
  & "E:/Program Files/Python/python.exe" "E:/documents/Thesis code/testing_models.py" `
  --model "E:\documents\Thesis code\models\best_svm_rbf_threshold.joblib" `
  --data "E:\documents\Thesis code\features\all_strokes_combined.csv" `
  --threshold 0.6

  E:\documents\Thesis code\feature_test\combined_testing_sample.csv

"""

import argparse, joblib, pandas as pd, numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, balanced_accuracy_score, confusion_matrix, classification_report
)
import seaborn as sns, matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to trained .joblib model")
    ap.add_argument("--data", required=True, help="CSV containing test features + true labels")
    ap.add_argument("--threshold", type=float, default=None, help="Optional confidence threshold")
    args = ap.parse_args()

    # Load model and data
    model = joblib.load(args.model)
    df = pd.read_csv(args.data)
    if "label" not in df.columns:
        raise ValueError("The test dataset must contain a 'label' column for evaluation.")

    X = df.drop(columns=["label"]).values
    y_true = df["label"].values

    # Predict
    if hasattr(model, "predict_proba") and args.threshold is not None:
        probs = model.predict_proba(X)
        preds = np.array([
            model.classes_[i] if p.max() >= args.threshold else "uncertain"
            for p, i in zip(probs, probs.argmax(axis=1))
        ])
    else:
        preds = model.predict(X)

    # Filter out 'uncertain' cases for fair comparison
    mask = preds != "uncertain"
    y_eval = y_true[mask]
    preds_eval = preds[mask]

    total = len(y_true)
    accepted = np.sum(mask)
    correct = np.sum(y_eval == preds_eval)
    acc = accuracy_score(y_eval, preds_eval) if accepted > 0 else 0
    bal_acc = balanced_accuracy_score(y_eval, preds_eval) if accepted > 0 else 0
    prec = precision_score(y_eval, preds_eval, average="macro", zero_division=0) if accepted > 0 else 0
    rec = recall_score(y_eval, preds_eval, average="macro", zero_division=0) if accepted > 0 else 0
    f1m = f1_score(y_eval, preds_eval, average="macro", zero_division=0) if accepted > 0 else 0

    print("\n===== MODEL EVALUATION METRICS =====")
    print(f"Total windows tested: {total}")
    print(f"Accepted predictions: {accepted} ({100*accepted/total:.1f}%)")
    print(f"Correct predictions:  {correct} ({100*correct/accepted if accepted>0 else 0:.1f}%)")
    print("------------------------------------")
    print(f"Accuracy:           {acc:.3f}")
    print(f"Balanced Accuracy:  {bal_acc:.3f}")
    print(f"Precision (macro):  {prec:.3f}")
    print(f"Recall (macro):     {rec:.3f}")
    print(f"F1 Score (macro):   {f1m:.3f}")

    # Optional: Confusion matrix
    if accepted > 0:
        cm = confusion_matrix(y_eval, preds_eval, labels=np.unique(y_true))
        plt.figure(figsize=(7,6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu",
                    xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
        plt.xlabel("Predicted"); plt.ylabel("True")
        plt.title("Confusion Matrix (Test Data)")
        plt.tight_layout()
        plt.savefig("confusion_test_data.png", dpi=150)
        plt.close()
        print("✅ Saved confusion matrix → confusion_test_data.png")

    print("\nPer-class metrics:")
    print(classification_report(y_eval, preds_eval, zero_division=0))

if __name__ == "__main__":
    main()
