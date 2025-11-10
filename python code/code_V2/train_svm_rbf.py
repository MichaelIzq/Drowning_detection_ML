#!/usr/bin/env python3
import pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import joblib

df = pd.read_csv(r"E:/documents/Thesis code/features_all_5s.csv")
X = df.drop(columns=["label","t_start_ms","t_end_ms"], errors="ignore")
y = df["label"]

Xtr, Xte, ytr, yte = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
sc = StandardScaler().fit(Xtr)
Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)

param_grid = {'C':[0.5,1,2,5],'gamma':['scale',0.1,0.01,0.001]}
gs = GridSearchCV(SVC(kernel='rbf',class_weight='balanced'), param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
gs.fit(Xtr_s, ytr)

print("Best params:", gs.best_params_)
print(classification_report(yte, gs.predict(Xte_s)))
cm = confusion_matrix(yte, gs.predict(Xte_s), labels=gs.classes_)
ConfusionMatrixDisplay(cm, display_labels=gs.classes_).plot(cmap="Oranges")
plt.title("Confusion Matrix – RBF SVM"); plt.savefig("E:/documents/Thesis code/models/confusion_svm_rbf.png", dpi=150)
joblib.dump(gs.best_estimator_, "E:/documents/Thesis code/models/model_svm_rbf.joblib")
