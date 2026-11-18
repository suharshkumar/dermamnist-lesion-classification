"""Aggregate Phase-3 run directories (run/artifacts_phase3 and optional run/artifacts_phase3_128) into the JSON files the
Phase-3 report builder and the paper read: phase3/results/ablations.json, phase3/results/seeds.json, phase3/results/summary.json.
numpy-only (no sklearn), so it runs on the memory-starved Mac.
Usage: run/.venv/bin/python -I DDP_Phase2/phase3/aggregate_phase3.py
"""
import os, json, glob, numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
A3 = os.path.join(ROOT, "run", "artifacts_phase3"); A128 = os.path.join(ROOT, "run", "artifacts_phase3_128")
OUT = os.path.join(ROOT, "DDP_Phase2", "phase3", "results"); os.makedirs(OUT, exist_ok=True)
K = 7
def load_runs(base):
    runs = {}
    for d in sorted(glob.glob(os.path.join(base, "*"))):
        f = os.path.join(d, "results.json")
        if os.path.exists(f):
            r = json.load(open(f)); r.pop("history", None); runs[os.path.basename(d)] = r
    return runs
runs = load_runs(A3); runs128 = load_runs(A128) if os.path.isdir(A128) else {}
y = np.load(os.path.join(A3, "ensemble", "y_test.npy")) if os.path.exists(os.path.join(A3, "ensemble", "y_test.npy")) else None
def auc_ovr(p, y):
    aucs = []
    for c in range(K):
        s = p[:, c]; pos = s[y == c]; neg = s[y != c]; allv = np.concatenate([pos, neg])
        order = np.argsort(allv); ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
        uniq, inv = np.unique(allv, return_inverse=True); sums = np.bincount(inv, weights=ranks); cnts = np.bincount(inv); ranks = (sums / cnts)[inv]
        aucs.append((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
    return float(np.mean(aucs))
# ---- ablations: pair each abl_* run with its reference run ----
REF = {"abl_dinov2_s14_uniformlr": ("dinov2_s14_224_s0", "DINOv2 ViT-S/14: uniform LR vs layer-wise decay"),
       "abl_convnext_t_nomix": ("convnext_t_224_s0", "ConvNeXt-T: no mixup/CutMix vs full recipe"),
       "abl_densenet121_plain": ("densenet121_224_s0", "DenseNet-121: plain protocol vs our recipe")}
abl = []
for k, r in runs.items():
    base = k.rsplit("_s", 1)[0]
    if base in REF:
        ref_key, note = REF[base]; ref = runs.get(ref_key)
        abl.append(dict(name=note.split(":")[1].strip(), model=note.split(":")[0], top1=r["tta_top1"], balanced=r["tta_balanced"], auc=r["tta_auc"],
                        note=(f"reference run: {ref['tta_top1']*100:.2f}% / {ref['tta_balanced']*100:.2f}% / {ref['tta_auc']:.3f}" if ref else "reference pending")))
for k, r in runs128.items():
    ref = runs.get("convnext_t_224_s0")
    abl.append(dict(name="128-px data, same recipe", model="ConvNeXt-T", top1=r["tta_top1"], balanced=r["tta_balanced"], auc=r["tta_auc"],
                    note=(f"224-px reference: {ref['tta_top1']*100:.2f}% / {ref['tta_balanced']*100:.2f}% / {ref['tta_auc']:.3f}" if ref else "reference pending")))
json.dump(abl, open(os.path.join(OUT, "ablations.json"), "w"), indent=1)
# ---- seeds ----
seeds = {}
for k, r in runs.items():
    if k.startswith(("abl_", "probe_")): continue
    seeds.setdefault(k.rsplit("_s", 1)[0], []).append(dict(seed=r.get("seed"), top1=r["tta_top1"], balanced=r["tta_balanced"], auc=r["tta_auc"], f1=r["tta_macro_f1"]))
summ = {k: dict(n=len(v), top1_mean=float(np.mean([x["top1"] for x in v])), top1_std=float(np.std([x["top1"] for x in v], ddof=1)) if len(v) > 1 else None,
                auc_mean=float(np.mean([x["auc"] for x in v])), bal_mean=float(np.mean([x["balanced"] for x in v])), runs=v) for k, v in seeds.items()}
json.dump(summ, open(os.path.join(OUT, "seeds.json"), "w"), indent=1)
# ---- summary ----
ens = json.load(open(os.path.join(A3, "ensemble", "ensemble_results.json"))) if os.path.exists(os.path.join(A3, "ensemble", "ensemble_results.json")) else None
json.dump(dict(runs={k: {kk: v for kk, v in r.items() if kk != "cfg"} for k, r in runs.items()}, ensemble_points=(ens or {}).get("points"), cis=(ens or {}).get("cis"),
               tau_B=(ens or {}).get("tau_B"), tau_C=(ens or {}).get("tau_C")), open(os.path.join(OUT, "summary.json"), "w"), indent=1)
print("runs:", list(runs)); print("ablations:", len(abl)); print("seeds:", {k: v["n"] for k, v in summ.items()})
if ens: print("ensemble points:", {k: round(v["top1"], 4) for k, v in ens["points"].items() if k.startswith("ensemble")})
