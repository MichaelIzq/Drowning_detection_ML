#!/usr/bin/env python3
import pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import joblib

df = pd.read_csv(r"E:/documents/Thesis code/features_all_5s.csv")
X = df.drop(columns=["label","t_start_ms","t_end_ms"], errors="ignore")
y = df["label"]

Xtr, Xte, ytr, yte = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
sc = StandardScaler().fit(Xtr)
Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)

grid = {'n_neighbors':[3,5,7,9],'weights':['uniform','distance'],'p':[1,2]}
gs = GridSearchCV(KNeighborsClassifier(), grid, cv=5, scoring='f1_macro', n_jobs=-1)
gs.fit(Xtr_s, ytr)

print("Best params:", gs.best_params_)
print(classification_report(yte, gs.predict(Xte_s)))
cm = confusion_matrix(yte, gs.predict(Xte_s), labels=gs.classes_)
ConfusionMatrixDisplay(cm, display_labels=gs.classes_).plot(cmap="Blues")
plt.title("Confusion Matrix – KNN"); plt.savefig("E:/documents/Thesis code/models/confusion_knn.png", dpi=150)
joblib.dump(gs.best_estimator_, "E:/documents/Thesis code/models/model_knn.joblib")
print("✅ KNN model saved.")
