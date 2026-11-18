"""Phase-3 statistics from saved probabilities (numpy only; runs on the Mac).
Inputs : run/artifacts_phase3/<run>/test_tta_probs.npy, ensemble/test_probs_ensemble.npy, ensemble/ensemble_results.json,
         run/artifacts/v2_ensemble/test_probs_tuned.npy (Phase-2 PRIME), research/data/test_clean_mask.npy + test_lesion_ids.json
Outputs: DDP_Phase2/phase3/results/stats_phase3.json with, per system:
         - official test metrics (top-1, balanced, macro-F1, macro-AUC) with image-level AND lesion-cluster bootstrap 95% CIs
         - the same metrics on the lesion-clean test subset (1,232 images whose lesion never appears in train or val) and the leaky remainder
         - melanoma one-vs-rest AUROC, sensitivity at 80/90/95% specificity (thresholds chosen on the OFFICIAL VALIDATION split)
         - calibration: ECE (15 equal-width bins), NLL, Brier, for plain and tau-adjusted probabilities
         - paired McNemar + paired bootstrap delta vs Phase-2 PRIME-B and vs the best single model
"""
import os, json, glob, numpy as np
from math import comb
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
A3 = os.path.join(ROOT, "run", "artifacts_phase3"); ENS = os.path.join(A3, "ensemble"); R = os.path.join(ROOT, "DDP_Phase2", "research", "data")
OUT = os.path.join(ROOT, "DDP_Phase2", "phase3", "results", "stats_phase3.json"); K = 7
y = np.load(os.path.join(ENS, "y_test.npy")) if os.path.exists(os.path.join(ENS, "y_test.npy")) else np.load(os.path.join(ROOT, "run", "artifacts", "single_b0", "y_test.npy"))
y_val = np.load(os.path.join(ENS, "y_val.npy")) if os.path.exists(os.path.join(ENS, "y_val.npy")) else np.load(os.path.join(ROOT, "run", "dermamnist_64.npz"))["val_labels"].flatten()
clean = np.load(os.path.join(R, "test_clean_mask.npy")); lesions = np.array(json.load(open(os.path.join(R, "test_lesion_ids.json"))))
ens = json.load(open(os.path.join(ENS, "ensemble_results.json"))); prior = np.array(ens["prior"]); tauB, tauC = ens["tau_B"], ens["tau_C"]
rng = np.random.default_rng(0)

def bal(pred, yy): return float(np.mean([np.mean(pred[yy == c] == c) for c in range(K) if np.any(yy == c)]))
def f1m(pred, yy):
    f = []
    for c in range(K):
        tp = np.sum((pred == c) & (yy == c)); fp = np.sum((pred == c) & (yy != c)); fn = np.sum((pred != c) & (yy == c))
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))
def auc_bin(s, pos):
    order = np.argsort(s); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    u, inv = np.unique(s, return_inverse=True); sums = np.bincount(inv, weights=ranks); cnts = np.bincount(inv); ranks = (sums / cnts)[inv]
    npos = pos.sum(); nneg = len(s) - npos
    return float((ranks[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg)) if npos and nneg else float("nan")
def auc_macro(p, yy): return float(np.nanmean([auc_bin(p[:, c], yy == c) for c in range(K)]))
def metrics(p, yy, pred=None):
    pred = p.argmax(1) if pred is None else pred
    return dict(top1=float(np.mean(pred == yy)), balanced=bal(pred, yy), macro_f1=f1m(pred, yy), auc=auc_macro(p, yy), n=int(len(yy)))
def boot_ci(pred, yy, groups=None, n=2000):
    t1, b = [], []; idx_all = np.arange(len(yy))
    if groups is not None:
        ug, inv = np.unique(groups, return_inverse=True); members = [np.where(inv == g)[0] for g in range(len(ug))]
    for _ in range(n):
        if groups is None: i = rng.integers(0, len(yy), len(yy))
        else: i = np.concatenate([members[g] for g in rng.integers(0, len(ug), len(ug))])
        t1.append(np.mean(pred[i] == yy[i])); b.append(bal(pred[i], yy[i]))
    return dict(top1=[float(np.percentile(t1, 2.5)), float(np.percentile(t1, 97.5))], balanced=[float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))])
def ece(p, yy, bins=15):
    conf = p.max(1); pred = p.argmax(1); acc = (pred == yy).astype(float); e = 0.0
    for lo, hi in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any(): e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)
def nll(p, yy): return float(-np.mean(np.log(np.clip(p[np.arange(len(yy)), yy], 1e-12, 1))))
def brier(p, yy): oh = np.eye(K)[yy]; return float(np.mean(np.sum((p - oh) ** 2, 1)))
def adjust(p, tau): return np.log(np.clip(p, 1e-9, 1)) - tau * np.log(prior)
def softmax(z): z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)
def mcnemar(a, b):
    ca, cb = a == y, b == y; n01 = int(np.sum(ca & ~cb)); n10 = int(np.sum(~ca & cb)); n = n01 + n10; k = min(n01, n10)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return dict(only_first=n01, only_second=n10, p=p, net=n10 - n01)
def delta_ci(a, b, n=2000):
    d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y)); d.append(np.mean(b[i] == y[i]) - np.mean(a[i] == y[i]))
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
def mel_points(p_val, p_te):
    """Melanoma-vs-rest: thresholds for 80/90/95% specificity chosen on the validation split, applied to test."""
    s_v = p_val[:, 4]; neg_v = s_v[y_val != 4]; out = {}
    for spec in (0.80, 0.90, 0.95):
        thr = float(np.quantile(neg_v, spec)); s_t = p_te[:, 4]
        sens = float(np.mean(s_t[y == 4] >= thr)); spc = float(np.mean(s_t[y != 4] < thr)); out[f"sens_at_spec{int(spec*100)}"] = dict(threshold=thr, sensitivity=sens, achieved_specificity=spc)
    out["auroc"] = auc_bin(p_te[:, 4], y == 4); return out

systems = {}
for d in sorted(glob.glob(os.path.join(A3, "*"))):
    f = os.path.join(d, "test_tta_probs.npy")
    if os.path.exists(f) and os.path.basename(d) != "ensemble":
        systems[os.path.basename(d)] = (np.load(f), np.load(os.path.join(d, "val_tta_probs.npy")) if os.path.exists(os.path.join(d, "val_tta_probs.npy")) else None, 0.0)
pt = np.load(os.path.join(ENS, "test_probs_ensemble.npy")); pv = np.load(os.path.join(ENS, "val_probs_ensemble.npy")) if os.path.exists(os.path.join(ENS, "val_probs_ensemble.npy")) else None
systems["ensemble_A"] = (pt, pv, 0.0); systems["ensemble_B"] = (pt, pv, tauB); systems["ensemble_C"] = (pt, pv, tauC)
p2 = np.load(os.path.join(ROOT, "run", "artifacts", "v2_ensemble", "test_probs_tuned.npy")); prior2 = np.array(json.load(open(os.path.join(ROOT, "run", "artifacts", "v2_ensemble", "ensemble_v2_results.json")))["prior"])
pred_p2B = (np.log(np.clip(p2, 1e-9, 1)) - 0.15 * np.log(prior2)).argmax(1)
res = {"n_test": int(len(y)), "n_clean": int(clean.sum()), "systems": {}}
for name, (p, pval, tau) in systems.items():
    pred = adjust(p, tau).argmax(1); pa = softmax(adjust(p, tau))
    r = dict(tau=tau, official=metrics(p, y, pred), ci_image=boot_ci(pred, y), ci_lesion=boot_ci(pred, y, lesions),
             clean=metrics(p[clean], y[clean], pred[clean]), leaky=metrics(p[~clean], y[~clean], pred[~clean]),
             per_class_recall=[float(np.mean(pred[y == c] == c)) for c in range(K)],
             calibration=dict(plain=dict(ece=ece(p, y), nll=nll(p, y), brier=brier(p, y)), adjusted=dict(ece=ece(pa, y), nll=nll(pa, y), brier=brier(pa, y))),
             vs_phase2_B=dict(mcnemar=mcnemar(pred_p2B, pred), delta_ci=delta_ci(pred_p2B, pred)))
    if pval is not None: r["melanoma"] = mel_points(pval, p)
    res["systems"][name] = r
best = max((k for k in systems if not k.startswith(("ensemble", "abl_", "probe_"))), key=lambda k: res["systems"][k]["official"]["top1"], default=None)
if best:
    pb = systems[best][0].argmax(1); pe = adjust(pt, 0.0).argmax(1)
    res["ensemble_vs_best_single"] = dict(best=best, mcnemar=mcnemar(pb, pe), delta_ci=delta_ci(pb, pe))
json.dump(res, open(OUT, "w"), indent=1)
for k, r in res["systems"].items():
    o, c, l = r["official"], r["clean"], r["leaky"]
    print(f"{k:28s} off {o['top1']*100:.2f} [{r['ci_image']['top1'][0]*100:.1f},{r['ci_image']['top1'][1]*100:.1f}] lesionCI [{r['ci_lesion']['top1'][0]*100:.1f},{r['ci_lesion']['top1'][1]*100:.1f}] "
          f"bal {o['balanced']*100:.2f} AUC {o['auc']:.4f} | clean {c['top1']*100:.2f} bal {c['balanced']*100:.2f} | leaky {l['top1']*100:.2f} | ECE {r['calibration']['plain']['ece']:.3f} | vs P2-B p={r['vs_phase2_B']['mcnemar']['p']:.1e}")
if best: print("ensemble vs best single:", res["ensemble_vs_best_single"])
