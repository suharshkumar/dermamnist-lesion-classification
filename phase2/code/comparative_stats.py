"""Comparative statistics for the final report (numpy only).
 - bootstrap 95% CIs (2,000 resamples of the 2,005 test images) for top-1 / balanced accuracy of baseline, members, ensemble points
 - paired McNemar exact-ish test (mid-p via binomial) baseline vs ensemble on the same test images
 - per-class recall deltas, and rank among published 28/64-px results
Writes ../DDP_Phase2/results/comparative_stats.json
"""
import json, os, numpy as np
from math import comb
rng = np.random.default_rng(0)
ART = "artifacts"; OUT = "../DDP_Phase2/results/comparative_stats.json"
y = np.load(os.path.join(ART, "single_b0", "y_test.npy"))
base = np.load(os.path.join(ART, "single_b0", "tta_probs.npy"))                 # archived Phase-1 (80.00%)
ens = json.load(open(os.path.join(ART, "v2_ensemble", "ensemble_v2_results.json")))
prior = np.array(ens["prior"]); pt = np.load(os.path.join(ART, "v2_ensemble", "test_probs_tuned.npy"))
members = {os.path.basename(m): np.load(os.path.join(ART, os.path.basename(m), "test_tta_probs.npy")) for m in ens["members"]}
adjust = lambda p, tau: (np.log(np.clip(p, 1e-9, 1)) - tau * np.log(prior)).argmax(1)
preds = {"phase1_baseline": base.argmax(1), "phase1_tau0.5": adjust(base, 0.5),
         "ensemble_equal": np.mean(list(members.values()), axis=0).argmax(1),
         "PRIME_A": adjust(pt, 0.0), "PRIME_B": adjust(pt, ens["tau_B"]), "PRIME_C": adjust(pt, ens["tau_C"])}
for k, v in members.items(): preds["member_" + k] = v.argmax(1)
K = 7
def bal(pred, yy): return float(np.mean([np.mean(pred[yy == c] == c) for c in range(K) if np.any(yy == c)]))
def boot(pred, n=2000):
    t1, b = [], []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y)); t1.append(np.mean(pred[idx] == y[idx])); b.append(bal(pred[idx], y[idx]))
    return dict(top1=float(np.mean(pred == y)), top1_ci=[float(np.percentile(t1, 2.5)), float(np.percentile(t1, 97.5))],
                balanced=bal(pred, y), balanced_ci=[float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))])
def mcnemar(p1, p2):
    c1, c2 = (p1 == y), (p2 == y); b = int(np.sum(c1 & ~c2)); c = int(np.sum(~c1 & c2)); n = b + c
    # exact two-sided binomial test on discordant pairs
    k = min(b, c); p = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n * 2 if n else 1.0
    return dict(only_first_correct=b, only_second_correct=c, p_value=float(min(1.0, p)), net_gain=c - b)
def paired_boot_delta(p1, p2, n=2000):
    d = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y)); d.append(np.mean(p2[idx] == y[idx]) - np.mean(p1[idx] == y[idx]))
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
res = {"n_test": int(len(y)), "systems": {k: boot(v) for k, v in preds.items()},
       "mcnemar": {"baseline_vs_PRIME_A": mcnemar(preds["phase1_baseline"], preds["PRIME_A"]),
                   "baseline_vs_PRIME_B": mcnemar(preds["phase1_baseline"], preds["PRIME_B"]),
                   "baseline_vs_ensemble_equal": mcnemar(preds["phase1_baseline"], preds["ensemble_equal"]),
                   "member_b0_96_vs_PRIME_B": mcnemar(preds["member_b0_96"], preds["PRIME_B"])},
       "paired_delta_ci": {"baseline_to_PRIME_A": paired_boot_delta(preds["phase1_baseline"], preds["PRIME_A"]),
                           "baseline_to_PRIME_B": paired_boot_delta(preds["phase1_baseline"], preds["PRIME_B"])},
       "per_class_recall": {k: [float(np.mean(v[y == c] == c)) for c in range(K)] for k, v in preds.items()},
       "per_class_precision": {k: [float(np.mean(y[v == c] == c)) if np.any(v == c) else 0.0 for c in range(K)] for k, v in preds.items()}}
# error reduction
e0 = 1 - res["systems"]["phase1_baseline"]["top1"]; eB = 1 - res["systems"]["PRIME_B"]["top1"]
res["relative_error_reduction_B"] = float((e0 - eB) / e0)
# malignant / pre-malignant sensitivity (AK=0, BCC=1, MEL=4) pooled
mal = np.isin(y, [0, 1, 4])
res["malignant_pooled_recall"] = {k: float(np.mean(v[mal] == y[mal])) for k, v in preds.items()}
# published 28/64-px results (official split), top-1 %
pub = {"ResNet-18 (28) v2": 73.5, "ResNet-18 (224 from 28) v2": 75.4, "ResNet-50 (28) v2": 73.5, "ResNet-50 (224 from 28) v2": 73.1,
       "auto-sklearn v2": 71.9, "AutoKeras v2": 74.9, "Google AutoML Vision v2": 76.8,
       "ResNet-18 (64) MedMNIST+": 76.36, "DenseNet-121 (64) MedMNIST+": 76.16, "ViT-B/16 (64) MedMNIST+": 77.01,
       "VGG16 (64) MedMNIST+": 79.53, "AlexNet (64) MedMNIST+": 78.69, "DINO ViT-B/16 (64) MedMNIST+": 76.87,
       "EfficientNet-B4 (64) MedMNIST+": 71.45, "EVA-02 ViT-B/16 (64) MedMNIST+": 75.28, "SAM ViT-B/16 (64) MedMNIST+": 74.68,
       "CLIP ViT-B/16 (64) MedMNIST+": 72.44}
ours = {"Phase 1 (this work)": 100 * res["systems"]["phase1_baseline"]["top1"], "PRIME A (this work)": 100 * res["systems"]["PRIME_A"]["top1"],
        "PRIME B (this work)": 100 * res["systems"]["PRIME_B"]["top1"]}
allr = sorted({**pub, **ours}.items(), key=lambda kv: -kv[1])
res["leaderboard_28_64px"] = allr
res["published_best_64px"] = 79.53
json.dump(res, open(OUT, "w"), indent=2)
S = res["systems"]
print("n =", len(y))
for k in ["phase1_baseline", "phase1_tau0.5", "member_b0_96", "member_b1_112", "ensemble_equal", "PRIME_A", "PRIME_B", "PRIME_C"]:
    s = S[k]; print(f"{k:18s} top-1 {s['top1']*100:.2f} [{s['top1_ci'][0]*100:.2f},{s['top1_ci'][1]*100:.2f}]  bal {s['balanced']*100:.2f} [{s['balanced_ci'][0]*100:.2f},{s['balanced_ci'][1]*100:.2f}]  mal-recall {res['malignant_pooled_recall'][k]*100:.1f}")
print("McNemar:", json.dumps(res["mcnemar"]))
print("paired delta CI:", res["paired_delta_ci"], "rel. error reduction B:", round(res["relative_error_reduction_B"], 4))
print("leaderboard:", [(k, round(v, 2)) for k, v in allr[:8]])
