"""numpy-only version of prior_correction_baseline.py (no sklearn/matplotlib) for a memory-starved machine.
Same JSON output; the figure is produced later by prior_correction_baseline.py or ensemble_v2.py."""
import json, os, numpy as np
OUT = "../DDP_Phase2/results"; os.makedirs(OUT, exist_ok=True)
p = np.load("artifacts/single_b0/tta_probs.npy"); y = np.load("artifacts/single_b0/y_test.npy")
prior = np.bincount(np.load("dermamnist_64.npz")["train_labels"].flatten(), minlength=7) / 7007
K = 7
def recall(pred): return [float(np.mean(pred[y == c] == c)) for c in range(K)]
def precision(pred): return [float(np.mean(y[pred == c] == c)) if np.any(pred == c) else 0.0 for c in range(K)]
def m(pred):
    r, pr = recall(pred), precision(pred)
    f1 = [2 * a * b / (a + b) if a + b > 0 else 0.0 for a, b in zip(pr, r)]
    return dict(top1=float(np.mean(pred == y)), balanced=float(np.mean(r)), macro_f1=float(np.mean(f1)), recall=r, precision=pr, f1=f1)
def auc_ovr(p, y):   # Mann-Whitney one-vs-rest, macro average
    aucs = []
    for c in range(K):
        s = p[:, c]; pos = s[y == c]; neg = s[y != c]
        order = np.argsort(np.concatenate([pos, neg])); ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
        # tie handling: average ranks
        allv = np.concatenate([pos, neg]); uniq, inv = np.unique(allv, return_inverse=True)
        sums = np.bincount(inv, weights=ranks); cnts = np.bincount(inv); ranks = (sums / cnts)[inv]
        aucs.append((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
    return float(np.mean(aucs))
top3 = float(np.mean([y[i] in np.argsort(r)[-3:] for i, r in enumerate(p)]))
adj = lambda tau: (np.log(np.clip(p, 1e-9, 1)) - tau * np.log(prior)).argmax(1)
taus = np.round(np.arange(0, 1.51, 0.05), 2)
sweep = [dict(tau=float(t), **{k: v for k, v in m(adj(t)).items() if k in ("top1", "balanced", "macro_f1")}) for t in taus]
res = dict(prior=prior.tolist(), auc_macro=auc_ovr(p, y), top3=top3, tau0=m(adj(0.0)), tau05=m(adj(0.5)), tau1=m(adj(1.0)), sweep=sweep)
json.dump(res, open(os.path.join(OUT, "prior_correction_baseline.json"), "w"), indent=2)
print("prior", np.round(prior, 3)); print("AUC %.4f top3 %.4f" % (res["auc_macro"], top3))
for k in ("tau0", "tau05", "tau1"):
    r = res[k]; print(f"{k:6s} top-1 {r['top1']*100:.2f} balanced {r['balanced']*100:.2f} macroF1 {r['macro_f1']*100:.2f} recall", [round(x, 3) for x in r["recall"]])
