#!/usr/bin/env python3
import argparse, pandas as pd, numpy as np, joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import seaborn as sns, matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", default="best_knn_threshold.joblib")
    ap.add_argument("--threshold", type=float, default=0.6)
    args = ap.parse_args()

    df = pd.read_csv(args.inp)
    X = df.drop(columns=["label"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    model = KNeighborsClassifier(n_neighbors=5, weights="distance", p=2)
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)
    preds = np.array([model.classes_[i] if p.max() >= args.threshold else "uncertain"
                      for p, i in zip(probs, probs.argmax(axis=1))])

    acc = accuracy_score(y_test[preds != "uncertain"], preds[preds != "uncertain"]) if np.any(preds != "uncertain") else 0
    print(f"Accuracy (filtered): {acc:.3f}")
    print(f"F1 (macro): {f1_score(y_test, preds, average='macro', zero_division=0):.3f}")
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, preds, labels=np.unique(y_test)))

    cm = confusion_matrix(y_test, preds, labels=np.unique(y_test))
    plt.figure(figsize=(7,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("KNN (Thresholded) Confusion")
    plt.tight_layout(); plt.savefig("confusion_knn_threshold.png", dpi=150)
    joblib.dump(model, args.out)
    print("✅ Saved model:", args.out)

if __name__ == "__main__":
    main()

"""
& "E:/Program Files/Python/python.exe" "E:/documents/Thesis code/train_knn.py" `
  --in "E:/documents/Thesis code/features/all_strokes_combined.csv" `
  --out "E:/documents/Thesis code/models/best_knn_threshold.joblib" `
  --threshold 0.6
  """