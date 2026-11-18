"""
Combine phase-2 members (validation-tuned) and apply a prior-corrected decision rule.

For each member directory (argument list) we expect val_tta_probs.npy, test_tta_probs.npy,
y_val.npy, y_test.npy written by train_v2.py.

Steps
 1. Ensemble weights: grid over the simplex (step 0.1), chosen by validation top-1.
 2. Prior correction (logit adjustment, Menon et al. 2021):
        score_c(x) = log p_c(x) - tau * log pi_c,   pi = train-split class prior.
    tau = 0 reproduces the plain argmax (max top-1); larger tau trades top-1 for
    balanced accuracy. tau is chosen on validation for two operating points:
      A  "accuracy"  : tau = 0
      B  "screening" : largest tau whose validation top-1 is within DELTA of the best
      C  "balanced"  : tau maximising validation balanced accuracy
 3. Report every operating point on the test split and save figures + JSON.
"""
import os, sys, json, itertools
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             classification_report, roc_auc_score, f1_score)

CLASS_NAMES = ['Actinic Ker.', 'Basal Cell Ca.', 'Benign Ker.', 'Dermatofibroma', 'Melanoma', 'Nevi', 'Vascular']
OUT = os.environ.get("ENS_OUT", "artifacts/v2_ensemble"); os.makedirs(OUT, exist_ok=True)
DELTA = float(os.environ.get("DELTA", 0.005))
dirs = sys.argv[1:] or ["artifacts/single_b0", "artifacts/v2_b0_resume", "artifacts/v2_v2b0"]
dirs = [d for d in dirs if os.path.exists(os.path.join(d, "test_tta_probs.npy"))]
assert dirs, "no member probabilities found"

y_val = np.load(os.path.join(dirs[0], "y_val.npy")); y_te = np.load(os.path.join(dirs[0], "y_test.npy"))
V = [np.load(os.path.join(d, "val_tta_probs.npy")) for d in dirs]
T = [np.load(os.path.join(d, "test_tta_probs.npy")) for d in dirs]
y_tr = np.load("dermamnist_64.npz")["train_labels"].flatten()
prior = np.bincount(y_tr, minlength=7) / len(y_tr)

def top3(p, y): return float(np.mean([y[i] in np.argsort(r)[-3:] for i, r in enumerate(p)]))
def metrics(p, y, pred=None):
    pred = p.argmax(1) if pred is None else pred
    return dict(top1=accuracy_score(y, pred), top3=top3(p, y), balanced=balanced_accuracy_score(y, pred),
                macro_f1=f1_score(y, pred, average="macro"),
                auc=roc_auc_score(y, p / p.sum(1, keepdims=True), multi_class="ovr", average="macro"))

print("Members:")
for d, v, t in zip(dirs, V, T):
    print(f"  {d:30s} val top-1 {accuracy_score(y_val, v.argmax(1))*100:.2f}%  "
          f"test top-1 {accuracy_score(y_te, t.argmax(1))*100:.2f}%  test bal {balanced_accuracy_score(y_te, t.argmax(1))*100:.2f}%")

# 1. ensemble weights on validation
best_w, best_acc = None, -1
grid = np.arange(0, 1.0001, 0.1)
for w in itertools.product(grid, repeat=len(dirs)):
    if abs(sum(w) - 1) > 1e-6: continue
    pv = sum(wi * vi for wi, vi in zip(w, V))
    acc = accuracy_score(y_val, pv.argmax(1))
    if acc > best_acc + 1e-9: best_acc, best_w = acc, w
w_equal = tuple([1 / len(dirs)] * len(dirs))
print(f"\nEnsemble weights (val-tuned): {dict(zip(dirs, [round(x,2) for x in best_w]))}  val top-1 {best_acc*100:.2f}%")
pv_tuned = sum(wi * vi for wi, vi in zip(best_w, V)); pt_tuned = sum(wi * ti for wi, ti in zip(best_w, T))
pv_eq = sum(wi * vi for wi, vi in zip(w_equal, V));  pt_eq = sum(wi * ti for wi, ti in zip(w_equal, T))

# 2. prior correction sweep on validation
def adjust(p, tau): return np.log(np.clip(p, 1e-9, 1)) - tau * np.log(prior)
taus = np.round(np.arange(0, 1.51, 0.05), 2)
curve = []
for tau in taus:
    pred_v = adjust(pv_tuned, tau).argmax(1); pred_t = adjust(pt_tuned, tau).argmax(1)
    curve.append(dict(tau=float(tau),
                      val_top1=accuracy_score(y_val, pred_v), val_bal=balanced_accuracy_score(y_val, pred_v),
                      test_top1=accuracy_score(y_te, pred_t), test_bal=balanced_accuracy_score(y_te, pred_t)))
v_top1 = np.array([c["val_top1"] for c in curve]); v_bal = np.array([c["val_bal"] for c in curve])
tau_B = float(taus[np.where(v_top1 >= v_top1.max() - DELTA)[0].max()])
tau_C = float(taus[v_bal.argmax()])
print(f"tau for screening point (val top-1 within {DELTA*100:.1f} pt of best): {tau_B};  tau maximising val balanced acc: {tau_C}")

# 3. report
points = {}
def report(name, p, tau):
    pred = adjust(p, tau).argmax(1)
    m = metrics(p, y_te, pred); m["tau"] = tau
    points[name] = m
    print(f"\n== {name} (tau={tau}) ==  top-1 {m['top1']*100:.2f}%  top-3 {m['top3']*100:.2f}%  "
          f"balanced {m['balanced']*100:.2f}%  macro-F1 {m['macro_f1']*100:.2f}%  AUC {m['auc']:.4f}")
    print(classification_report(y_te, pred, target_names=CLASS_NAMES, digits=4))
    return pred

for d, t in zip(dirs, T): points["member:" + d] = metrics(t, y_te) | {"tau": 0.0}
report("ensemble_equal", pt_eq, 0.0)
pred_A = report("ensemble_tuned_A_accuracy", pt_tuned, 0.0)
pred_B = report("ensemble_tuned_B_screening", pt_tuned, tau_B)
pred_C = report("ensemble_tuned_C_balanced", pt_tuned, tau_C)

json.dump(dict(members=dirs, weights=list(best_w), prior=prior.tolist(), delta=DELTA,
               tau_B=tau_B, tau_C=tau_C, curve=curve, points=points),
          open(os.path.join(OUT, "ensemble_v2_results.json"), "w"), indent=2)
np.save(os.path.join(OUT, "test_probs_tuned.npy"), pt_tuned)

# figures
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(taus, [c["val_top1"]*100 for c in curve], "o-", ms=3, label="Validation top-1")
ax.plot(taus, [c["val_bal"]*100 for c in curve], "s-", ms=3, label="Validation balanced acc.")
ax.plot(taus, [c["test_top1"]*100 for c in curve], "o--", ms=3, alpha=.6, label="Test top-1")
ax.plot(taus, [c["test_bal"]*100 for c in curve], "s--", ms=3, alpha=.6, label="Test balanced acc.")
for t, lab in [(tau_B, "B"), (tau_C, "C")]:
    ax.axvline(t, color="grey", ls=":", lw=1); ax.text(t, ax.get_ylim()[0]+1, f" {lab}", fontsize=9)
ax.set_xlabel(r"prior-correction exponent $\tau$"); ax.set_ylabel("accuracy (%)"); ax.grid(alpha=.3); ax.legend(fontsize=8)
ax.set_title("Prior correction: top-1 vs balanced accuracy"); plt.tight_layout()
plt.savefig(os.path.join(OUT, "tau_tradeoff.png"), dpi=160); plt.close()

for name, pred in [("A", pred_A), ("B", pred_B), ("C", pred_C)]:
    cm = confusion_matrix(y_te, pred); cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    for a, M, fmt, title in [(ax[0], cm, "d", "counts"), (ax[1], cmn, ".2f", "row-normalised")]:
        a.imshow(M, cmap="Blues"); a.set_xticks(range(7)); a.set_yticks(range(7))
        a.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=8); a.set_yticklabels(CLASS_NAMES, fontsize=8)
        for i in range(7):
            for j in range(7):
                a.text(j, i, format(M[i, j], fmt), ha="center", va="center", fontsize=7,
                       color="white" if M[i, j] > 0.6 * M.max() else "black")
        a.set_title(f"Operating point {name}: {title}"); a.set_xlabel("predicted"); a.set_ylabel("true")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, f"cm_{name}.png"), dpi=150); plt.close()

# per-class recall comparison figure
fig, ax = plt.subplots(figsize=(8, 4.2)); xs = np.arange(7); wdt = 0.26
rec = lambda pred: confusion_matrix(y_te, pred, normalize="true").diagonal() * 100
base_pred = T[0].argmax(1)
ax.bar(xs - wdt, rec(base_pred), wdt, label=f"baseline member ({dirs[0].split('/')[-1]})")
ax.bar(xs, rec(pred_A), wdt, label="ensemble, point A (accuracy)")
ax.bar(xs + wdt, rec(pred_B), wdt, label="ensemble, point B (screening)")
ax.set_xticks(xs); ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right", fontsize=8); ax.set_ylabel("recall (%)")
ax.set_ylim(0, 100); ax.grid(axis="y", alpha=.3); ax.legend(fontsize=8); ax.set_title("Per-class recall on the test split")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "per_class_recall.png"), dpi=160); plt.close()
print("\nSaved to", OUT)
