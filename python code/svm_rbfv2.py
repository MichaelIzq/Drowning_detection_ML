#!/usr/bin/env python3
import argparse, joblib, pandas as pd, numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", required=True)
    ap.add_argument("--out", default="best_svm_rbf.joblib")
    ap.add_argument("--threshold", type=float, default=0.6)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    X = df.drop(columns=["label"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    pipe = Pipeline([("scaler", StandardScaler()),
                     ("clf", SVC(kernel="rbf", probability=True, C=1.0, gamma="scale"))])
    pipe.fit(X_train, y_train)

    probs = pipe.predict_proba(X_test)
    preds = np.array([pipe.classes_[i] if p.max() >= args.threshold else "uncertain"
                      for p, i in zip(probs, probs.argmax(axis=1))])

    acc = accuracy_score(y_test, preds[preds != "uncertain"])
    print(f"Accuracy (filtered): {acc:.3f}")
    print(f"F1-score (macro): {f1_score(y_test, preds, average='macro', zero_division=0):.3f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds, labels=np.unique(y_test)))
    joblib.dump(pipe, args.out)
    print(f"Saved model: {args.out}")

if __name__ == "__main__":
    main()
