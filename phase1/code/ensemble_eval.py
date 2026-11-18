"""Average saved TTA probabilities across trained models -> ensemble top-1."""
import os, glob, json
import numpy as np
from sklearn.metrics import (classification_report, confusion_matrix,
                             balanced_accuracy_score, accuracy_score)

CLASS_NAMES = ['Actinic Ker.', 'Basal Cell Ca.', 'Benign Ker.',
               'Dermatofibroma', 'Melanoma', 'Nevi', 'Vascular']

dirs = sorted(glob.glob("artifacts/m*_*"))
probs, y = [], None
for d in dirs:
    p = os.path.join(d, "tta_probs.npy")
    if not os.path.exists(p):
        print(f"skip {d} (no tta_probs yet)"); continue
    probs.append(np.load(p))
    y = np.load(os.path.join(d, "y_test.npy"))
    acc = accuracy_score(y, probs[-1].argmax(1))
    print(f"{d:28s} top-1 = {acc*100:.2f}%")

if not probs:
    raise SystemExit("no model probabilities found")

ens = np.mean(probs, axis=0)
pred = ens.argmax(1)
top1 = accuracy_score(y, pred)
bal  = balanced_accuracy_score(y, pred)
top3 = float(np.mean([y[i] in np.argsort(p)[-3:] for i, p in enumerate(ens)]))
print("\n===== ENSEMBLE (%d models) =====" % len(probs))
print(f"Top-1 accuracy    : {top1*100:.2f}%")
print(f"Top-3 accuracy    : {top3*100:.2f}%")
print(f"Balanced accuracy : {bal*100:.2f}%")
print("\n" + classification_report(y, pred, target_names=CLASS_NAMES, digits=4))
print("Confusion matrix (rows=true):")
print(confusion_matrix(y, pred))
json.dump({"n_models": len(probs), "top1": top1, "top3": top3, "balanced": bal},
          open("artifacts/ensemble_results.json", "w"), indent=2)
