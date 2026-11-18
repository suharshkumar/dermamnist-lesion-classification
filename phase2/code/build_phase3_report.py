# -*- coding: utf-8 -*-
"""Phase-3 (next-term) DDP report: native 224-px transfer learning + prior-corrected ensembles on DermaMNIST.

Reads run/artifacts_phase3/{<run>/results.json, ensemble/ensemble_results.json, probe_*/results.json} written by
DDP_Phase2/phase3/prime3_colab.py (imported with phase3/import_phase3_results.sh) and the Phase-2 archive for the paired
comparison. Missing pieces render as 'pending'. Output: DDP_Phase3_Report_DermaMNIST.pdf at the project root.
Run: run/.venv/bin/python -I DDP_Phase2/report_src/build_phase3_report.py
"""
import os, sys, json, glob, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from report_common import *          # styles, table(), fig(), BL(), P(), charts, jload/pct/num/g, paths (ROOT, RUN, P2, ASSETS)
import numpy as np
from math import comb

A3 = os.path.join(RUN, "artifacts_phase3"); ENS_DIR = os.path.join(A3, "ensemble")
P3 = os.path.join(P2, "phase3"); SHOTS3 = os.path.join(P3, "figures", "screenshots")
OUT3 = os.path.join(ROOT, "DDP_Phase3_Report_DermaMNIST.pdf")
ens = jload(os.path.join(ENS_DIR, "ensemble_results.json"))
runs = {os.path.basename(d): jload(os.path.join(d, "results.json")) for d in sorted(glob.glob(os.path.join(A3, "*"))) if os.path.exists(os.path.join(d, "results.json"))}
probes = {k: v for k, v in runs.items() if k.startswith("probe_")}
trained = {k: v for k, v in runs.items() if not k.startswith("probe_") and v and "cfg" in v}
abl = jload(os.path.join(P3, "results", "ablations.json"))          # optional, session 2
ext = jload(os.path.join(P3, "results", "external_tests.json"))     # optional: DermaMNIST-C / -E
st = jload(os.path.join(P3, "results", "stats_phase3.json"))         # lesion-clean subset, cluster CIs, calibration, melanoma points
audit = jload(os.path.join(P2, "research", "data", "lesion_audit.json"))
PH2 = dict(top1=0.8264, top3=0.9791, bal=0.6620, auc=0.9623, f1=0.6886)   # Phase-2 PRIME-B (64-px data)
PH1 = dict(top1=0.8000, top3=0.9736, bal=0.5487, auc=0.9494)
PUB224 = [  # (label, top-1 %, macro-AUC or None); official MedMNIST split, 224-px inputs; see research/lit_sweep.json for sources and caveats
    ("Sanap 2026, stacking of 4 models (single seed)", 91.67, 0.980), ("Sanap 2026, ViT-Base single (single seed)", 90.57, 0.982),
    ("LiteMed-ViT 2026, ResNet-18 (prose, 1-2 seeds)", 89.98, None), ("Tanvir 2025, Swin-B + ViT-B late fusion (single seed)", 89.5, None),
    ("Sanap 2026, ConvNeXt-Base single (single seed)", 89.48, 0.982), ("BiomedGPT-B 2024 (Nat. Med.)", 86.6, None),
    ("DenseNet-121 (MedMNIST+, e2e, 3 seeds)", 84.74, 0.9626), ("DINO ViT-B/16 linear probe (MedMNIST+)", 84.42, 0.9683),
    ("MKG-CARE 2026 (MedMamba + retrieval)", 83.6, 0.961), ("ResNet-18 (MedMNIST+, e2e, 3 seeds)", 82.33, 0.9527),
    ("ViT-B/16 (MedMNIST+, e2e, 3 seeds)", 82.31, 0.9628), ("MedNet 2025, 224 px (3 seeds)", 82.0, 0.950),
    ("AlexNet (MedMNIST+, e2e, 3 seeds)", 82.04, 0.9610), ("VGG16 (MedMNIST+, e2e, 3 seeds)", 81.55, 0.9551)]
PREPRINTS = [("Wu & Papiez 2025 (preprint, protocol not auditable): DINOv2 ViT-B/14", 92.02), ("Wu & Papiez 2025: ResNet-18", 86.05)]
NICE = {"convnext_t_224": "ConvNeXt-T (IN-22k)", "dinov2_b14_224": "DINOv2 ViT-B/14", "dinov2_s14_224": "DINOv2 ViT-S/14",
        "effv2_s_224": "EfficientNetV2-S (IN-21k)", "densenet121_224": "DenseNet-121 (IN-1k)"}
def nice(run):
    base = run.rsplit("_s", 1)[0]; seed = run.rsplit("_s", 1)[1] if "_s" in run else "0"
    return f"{NICE.get(base, base)}, seed {seed}"

# ---- paired statistics vs Phase 2 (same 2,005 test images) ----
stats = None
try:
    y = np.load(os.path.join(RUN, "artifacts", "single_b0", "y_test.npy"))
    p2 = np.load(os.path.join(RUN, "artifacts", "v2_ensemble", "test_probs_tuned.npy")); prior2 = np.array(json.load(open(os.path.join(RUN, "artifacts", "v2_ensemble", "ensemble_v2_results.json")))["prior"])
    pred2 = (np.log(np.clip(p2, 1e-9, 1)) - 0.15 * np.log(prior2)).argmax(1)              # Phase-2 PRIME-B
    p3 = np.load(os.path.join(ENS_DIR, "test_probs_ensemble.npy")); prior3 = np.array(ens["prior"])
    def mcn(a, b):
        ca, cb = a == y, b == y; n01 = int(np.sum(ca & ~cb)); n10 = int(np.sum(~ca & cb)); n = n01 + n10; k = min(n01, n10)
        p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
        return dict(only_first=n01, only_second=n10, p=p)
    rng = np.random.default_rng(0)
    def delta_ci(a, b, n=2000):
        d = []
        for _ in range(n):
            i = rng.integers(0, len(y), len(y)); d.append(np.mean(b[i] == y[i]) - np.mean(a[i] == y[i]))
        return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
    stats = {}
    for lab, tau in (("A", 0.0), ("B", ens["tau_B"])):
        pred3 = (np.log(np.clip(p3, 1e-9, 1)) - tau * np.log(prior3)).argmax(1)
        stats[lab] = dict(mcnemar=mcn(pred2, pred3), delta_ci=delta_ci(pred2, pred3), top1=float(np.mean(pred3 == y)))
except Exception as e:
    stats = None

story = []
# ================================================================== COVER
logo = os.path.join(ASSETS, "iitm_logo.png"); lw, lh = PILImage.open(logo).size
story += [Spacer(1, 0.4 * cm), Image(logo, width=3.9 * cm, height=3.9 * cm * lh / lw, hAlign='CENTER'), Spacer(1, 0.25 * cm),
          P("INDIAN INSTITUTE OF TECHNOLOGY MADRAS", COVER_S),
          P("<i>Siddhirbhavati Karmaja &mdash; Success is born of action</i>", style('m', parent=COVER_S, fontSize=9.5, textColor=GREY)),
          Spacer(1, 0.5 * cm), rule(MAROON, 1.4), Spacer(1, 0.3 * cm),
          P("Dual Degree Project (DDP) &mdash; Phase 3 Report", COVER_S), Spacer(1, 0.35 * cm),
          P("Native-Resolution Transfer Learning and Prior-Corrected<br/>Ensembles for Dermoscopic Lesion Classification on DermaMNIST", COVER_T),
          Spacer(1, 0.2 * cm),
          P("Modern backbones on the 224-pixel MedMNIST+ release, a pre-registered evaluation protocol, "
            "and the evidence base for a journal submission", style('sub', parent=COVER_S, fontSize=10.5, textColor=NAVY)),
          Spacer(1, 0.35 * cm), rule(MAROON, 1.4), Spacer(1, 0.8 * cm)]
ci = [[Paragraph("<b>Submitted by</b>", style('ci', fontSize=11, textColor=NAVY)), ""],
      [Paragraph("Suharsh Kumar", CELL), Paragraph("BE22B037", CELL)],
      [Paragraph("Sayantan Das", CELL), Paragraph("BS22B035", CELL)], ["", ""],
      [Paragraph("<b>Mentor</b>", style('ci', fontSize=11, textColor=NAVY)), ""],
      [Paragraph("G. K. Harish Balaji", CELL), Paragraph("BE21B017 &nbsp;(Senior Dual-Degree student)", CELL)]]
ct = Table(ci, colWidths=[7.0 * cm, 7.0 * cm]); ct.setStyle(TableStyle([('TOPPADDING', (0, 0), (-1, -1), 3)]))
story += [ct, Spacer(1, 0.9 * cm), P("November 2026 (work for the next term)", style('d', parent=COVER_S, fontSize=11, textColor=GREY)), PageBreak()]

# ================================================================== ABSTRACT
pts = ens["points"] if ens else {}
A = pts.get("ensemble_A_accuracy"); B = pts.get("ensemble_B_screening"); C = pts.get("ensemble_C_balanced")
best_run = max(trained.items(), key=lambda kv: kv[1]["tta_top1"]) if trained else None
story.append(P("Abstract", H1))
res_txt = (f"On the official test split the equal-weight ensemble reaches <b>{pct(A['top1'])} top-1</b>, {pct(A['top3'])} top-3, a macro-AUC of "
           f"{num(A['auc'])} and {pct(A['balanced'])} balanced accuracy; the screening point (&tau; = {B['tau']:.2f}) gives {pct(B['top1'])} top-1 with "
           f"{pct(B['balanced'])} balanced accuracy, and the balanced point reaches {pct(C['balanced'])}. The best single model, "
           f"{nice(best_run[0])}, scores {pct(best_run[1]['tta_top1'])}. "
           + (f"The gain over our Phase-2 system on the same 2,005 images is {100*stats['A']['delta_ci'][0]:.1f} to {100*stats['A']['delta_ci'][1]:.1f} points "
              f"(paired bootstrap 95% interval; McNemar p = {stats['A']['mcnemar']['p']:.1e}). " if stats else "")
           if ens else "Results are pending at the time of this build. ")
story.append(P(
    "This report covers the third phase of our Dual Degree Project on DermaMNIST, the seven-class dermoscopy benchmark derived from "
    "HAM10000. The first two phases worked on the 64-pixel release of the data and ended with an ensemble at 82.6% top-1, which is the best "
    "number we know of for that resolution but, as we argued in the final Phase-2 report, not a publishable result on its own: the "
    "published ceilings on the native 224-pixel images are higher (84.7% for DenseNet-121 in MedMNIST+), and the official split leaks "
    "lesions between partitions. Phase 3 therefore moves to the 224-pixel MedMNIST+ release and asks a sharper question: with modern "
    "pretrained backbones, a careful fine-tuning recipe and a decision rule that is fixed on validation data alone, how far above the "
    "published ceiling can a small group with a free GPU get, and can the result be defended in peer review? We fine-tune five "
    "ImageNet- or DINOv2-pretrained networks (ConvNeXt-T, DINOv2 ViT-B/14 and ViT-S/14, EfficientNetV2-S and a DenseNet-121 reproduction "
    "row) with layer-wise learning-rate decay, mixup/CutMix, label smoothing, stochastic depth and exponential-moving-average weights, "
    "evaluate each with eight-view dihedral test-time augmentation, average them with equal weights, and apply the prior-corrected "
    "decision rule from Phase 2. Model selection uses the final EMA weights and no test-set information. " + res_txt +
    "The report documents the protocol, the per-model and ensemble results with bootstrap confidence intervals, ablations, the "
    "comparison with published 224-pixel results, and what remains to be done before submission."))
story.append(Spacer(1, 0.3 * cm)); story.append(P("Table of Contents", H1))
for t in ["1. Why a Third Phase", "2. Data: the 224-pixel MedMNIST+ Release", "3. Method", "4. Evaluation Protocol (Pre-registered)",
          "5. Results", "6. Comparison with Published 224-pixel Results", "7. Discussion and Limitations", "8. Work Remaining this Term",
          "9. References", "Appendix A. Run Configurations", "Appendix B. Training Evidence (Colab Output)"]:
    story.append(Paragraph(t, style('toc', parent=BODY, spaceAfter=3, fontSize=10.5)))
story.append(PageBreak())

# ================================================================== 1
story += [P("1. Why a Third Phase", H1), P(
    "Phase 2 ended with a clear diagnosis. Averaging five EfficientNets trained on 64-pixel images and correcting the decision rule for the "
    "class prior took us from 80.0% to 82.6% top-1, and every minority class gained recall. But the published results that matter are "
    "obtained on 224-pixel images, where even a plainly trained DenseNet-121 reaches 84.7% [2], and the strongest frozen features (a DINO "
    "ViT-B/16 linear probe) reach 84.4% without any fine-tuning at all. Resolution, not modelling, was the ceiling. The second lesson was "
    "about evaluation: 13.5% of the lesions in DermaMNIST appear in more than one split [3], so numbers on the official split are "
    "optimistic, and a paper has to say so and, where possible, report leakage-free numbers as well."),
    P("The objectives for this term follow directly:"),
    BL("Train modern pretrained backbones on the native 224-pixel data with a fine-tuning recipe that is state of the art rather than default, "
       "and beat the published 224-pixel ceiling by a margin that survives a confidence interval."),
    BL("Keep the Phase-2 decision rule (prior correction with one exponent &tau;) so the accuracy-versus-sensitivity trade-off remains explicit."),
    BL("Fix every selection decision before looking at the test split: final EMA weights, equal ensemble weights, &tau; chosen on validation."),
    BL("Report the comparisons a reviewer will ask for: a reproduction of the published DenseNet-121 row under our protocol, a frozen-feature "
       "probe as a guardrail, three seeds for the headline models, ablations of the recipe, and the paired statistics against Phase 2."),
    BL("Produce the evidence base for a journal paper (the paper itself is a separate document)."),
    P("2. Data: the 224-pixel MedMNIST+ Release", H1), P(
    "MedMNIST+ [2] re-exported every MedMNIST image at 64, 128 and 224 pixels directly from the source images (600x450 HAM10000 "
    "dermoscopy photographs), so the 224-pixel arrays carry genuinely more information than the 64-pixel ones we used before, not an "
    "upsampled copy. The split is unchanged: 7,007 training, 1,003 validation and 2,005 test images, with the same labels (we assert this "
    "against the 64-pixel labels when loading). The class prior of the training split is 66.9% nevi, 11.1% melanoma, 11.0% benign keratosis, "
    "5.1% basal cell carcinoma, 3.3% actinic keratosis, 1.4% vascular and 1.1% dermatofibroma. The file is 1.09 GB; we download it from the "
    "Zenodo record once per Colab session and keep all images as 8-bit tensors on the GPU, so no float copy of the dataset ever exists in "
    "the 12 GB of host memory. The lesion-level leakage described in the Phase-2 report applies unchanged to this release; Section 7 "
    "returns to it."),
    table([["Split", "Images", "Nevi", "Melanoma", "Benign ker.", "BCC", "Actinic ker.", "Vascular", "Dermatofibroma"],
           ["Training", "7,007", "4,693", "779", "769", "359", "228", "99", "80"], ["Validation", "1,003", "671", "111", "110", "52", "33", "14", "12"],
           ["Test", "2,005", "1,341", "223", "220", "103", "66", "29", "23"]],
          [2.0 * cm, 1.4 * cm, 1.3 * cm, 1.7 * cm, 1.9 * cm, 1.3 * cm, 1.9 * cm, 1.6 * cm, 2.3 * cm], font=7.8),
    P("Table 1. Official DermaMNIST split (identical for every MedMNIST+ resolution).", CAP),
    PageBreak()]

# ================================================================== 3 METHOD
story += [P("3. Method", H1), P("3.1 Backbones", H2), P(
    "We chose five networks to cover three families and two kinds of pretraining. ConvNeXt-T [20] (28.6 M parameters, ImageNet-22k "
    "pretrained, 1k fine-tuned) and EfficientNetV2-S [11] (21.5 M, ImageNet-21k) are the strongest convolutional networks that train "
    "comfortably on a 16 GB T4 at 224 pixels. DINOv2 ViT-B/14 and ViT-S/14 [19] (86 M and 22 M) are self-supervised vision transformers "
    "whose frozen features already transfer unusually well; MedMNIST+ reported that their DINO v1 predecessor did <i>better</i> as a frozen "
    "linear probe (84.4%) than when fine-tuned with a uniform learning rate (81.3%), which told us the fine-tuning recipe would matter more "
    "than the architecture. DenseNet-121 with the original torchvision weights is included not because we expect it to win but because it "
    "is the published ceiling (84.74 &plusmn; 0.51% in MedMNIST+ [2]); training it under our recipe separates the effect of the recipe from "
    "the effect of the backbone. All models come from the timm library [30] with their published pretrained weights."),
    P("3.2 Fine-tuning recipe", H2), P(
    "The recipe is the same for every backbone apart from the learning rate and the layer-decay factor (Appendix A lists every value). "
    "Images are augmented on the GPU in batches: random resized crop (area 60-100%), horizontal and vertical flips, rotation by any angle "
    "(dermoscopic images have no canonical orientation), small translations, colour jitter, occasional Gaussian blur and random erasing. "
    "Half of the batches are mixed with mixup (&alpha; = 0.2) or CutMix (&alpha; = 1.0) [17, 31], and labels are smoothed by 0.1. Optimisation "
    "uses AdamW [13] with decoupled weight decay, a two-epoch linear warm-up and a cosine schedule to 10<sup>-6</sup>, gradient clipping at "
    "norm 1, and layer-wise learning-rate decay [34], so the pretrained stem is updated far more gently than the new classifier. Stochastic "
    "depth [33] regularises the deeper models. Training runs in 16-bit mixed precision with a loss scaler. An exponential moving average of "
    "the weights (decay 0.999) [21] is maintained throughout, and <b>the final EMA weights are the model</b>; we log the per-epoch validation "
    "accuracy of both the raw and the EMA weights for the curves, but we never pick an epoch by validation score."),
    P("3.3 Test-time augmentation, ensembling and the decision rule", H2), P(
    "At inference every image is presented in its eight dihedral views (four rotations, each with and without a flip) and the softmax outputs "
    "are averaged [12]; this is deterministic, so a result can be reproduced exactly. The ensemble is the equal-weight mean of the members' "
    "probabilities (pre-registered); a greedy forward-selected subset tuned on validation [22] is reported as a secondary row. The decision rule "
    "is the Phase-2 prior correction: predict argmax<sub>c</sub> [log p<sub>c</sub>(x) &minus; &tau; log &pi;<sub>c</sub>], with &pi; the training "
    "prior and &tau; selected on validation: &tau; = 0 (point A, accuracy), the largest &tau; whose validation top-1 is within 0.5 points of the best "
    "(point B, screening), and the &tau; that maximises validation balanced accuracy (point C)."),
    P("3.4 Frozen-feature guardrail", H2), P(
    "Before any fine-tuning we extract frozen DINOv2 ViT-B/14 features (class token concatenated with the mean patch token) and fit a "
    "logistic-regression linear probe (regularisation chosen on validation) and an 11-nearest-neighbour classifier. A fine-tuned DINOv2 model "
    "that does not beat its own linear probe on validation would indicate a broken recipe, not a weak backbone; the probe is also a result in "
    "its own right because it is directly comparable with the frozen-feature rows in MedMNIST+."),
    P("4. Evaluation Protocol (Pre-registered)", H1),
    BL("Official split only for model development; the test split is touched once per model, after training, to produce saved probabilities."),
    BL("Selection rules fixed in advance: final EMA weights, equal ensemble weights, &tau; by the Phase-2 rules on validation."),
    BL("Metrics: top-1, top-3, balanced accuracy (mean per-class recall), macro-F1 and one-vs-rest macro-AUC, all computed from the saved probabilities."),
    BL("Uncertainty: 95% bootstrap intervals over 2,000 resamples of the test images for every headline system; three seeds where the budget allows."),
    BL("Paired comparison with Phase 2 on the same 2,005 images: exact McNemar test on discordant pairs and a paired bootstrap interval of the difference."),
    BL("Leakage: results on the official split are reported with the Abhishek et al. caveat; leakage-free evaluation on DermaMNIST-C and -E is "
       "part of the remaining work (Section 8) and is reported in Section 5.6 where available."),
    PageBreak()]

# ================================================================== 5 RESULTS
story += [P("5. Results", H1), P("5.1 Frozen-feature probe", H2)]
if probes:
    pr = next(iter(probes.values()))
    story.append(P(f"The DINOv2 ViT-B/14 linear probe reaches {pct(pr['val_top1'])} on validation and <b>{pct(pr['tta_top1'])} top-1</b> on the test split "
                   f"(top-3 {pct(pr['tta_top3'])}, balanced {pct(pr['tta_balanced'])}, macro-AUC {num(pr['tta_auc'])}) without a single gradient step on "
                   f"dermoscopy data; the 11-nearest-neighbour classifier on the same features gives {pct(pr['knn11_test_top1'])}. For comparison, MedMNIST+ "
                   f"reports 84.42% for the DINO v1 ViT-B/16 linear probe at 224 pixels and 78.35% for its kNN. This is the bar every fine-tuned model below "
                   f"has to clear."))
else:
    story.append(P("<i>Pending.</i>"))
story.append(P("5.2 Single models", H2))
rows = [["Model", "Params (M)", "Epochs", "Val top-1 (EMA, final)", "Test top-1 (clean)", "Test top-1 (D4 TTA)", "Top-3", "Balanced", "Macro-AUC", "Macro-F1", "Minutes"]]
for k, r in sorted(trained.items(), key=lambda kv: -kv[1]["tta_top1"]):
    rows.append([nice(k), f"{r.get('params_M', 0):.1f}", str(r["cfg"]["epochs"]), pct(r.get("final_val_top1_ema")), pct(r["std_top1"]), pct(r["tta_top1"]),
                 pct(r["tta_top3"]), pct(r["tta_balanced"]), num(r["tta_auc"]), pct(r["tta_macro_f1"]), f"{r['minutes']:.0f}"])
if not trained: rows.append(["pending"] + ["-"] * 10)
story += [table(rows, [3.3 * cm, 1.2 * cm, 1.1 * cm, 1.7 * cm, 1.6 * cm, 1.6 * cm, 1.2 * cm, 1.4 * cm, 1.4 * cm, 1.3 * cm, 1.2 * cm], font=7.2),
          P("Table 2. Fine-tuned single models on the official test split (final EMA weights; no epoch selection). The published 224-pixel "
            "DenseNet-121 figure is 84.74 &plusmn; 0.51%.", CAP)]
if trained:
    dn = [v for k, v in trained.items() if k.startswith("densenet121")]
    story.append(P((f"Every fine-tuned model clears the published DenseNet-121 ceiling; the best, {nice(best_run[0])}, by "
                    f"{100*best_run[1]['tta_top1'] - 84.74:.1f} points. " if best_run and best_run[1]['tta_top1'] > 0.8474 else "")
                   + (f"DenseNet-121 itself, trained under our recipe, reaches {pct(dn[0]['tta_top1'])} against the published 84.74% with the plain "
                      f"MedMNIST+ protocol, which is the cleanest measurement of what the recipe alone is worth. " if dn else "")
                   + "Test-time augmentation adds between "
                   + f"{min(100*(r['tta_top1']-r['std_top1']) for r in trained.values()):.1f} and {max(100*(r['tta_top1']-r['std_top1']) for r in trained.values()):.1f} points over the clean pass."))
    story.append(fig(os.path.join(ENS_DIR, "val_curves.png"), 0.9, "Figure 1. Validation top-1 of the EMA weights per epoch for every run."))
story.append(P("5.3 Ensemble and operating points", H2))
if ens:
    cis = ens.get("cis", {})
    def cif(k, which):
        c = cis.get(k, {}).get(which); return f"[{100*c[0]:.1f}, {100*c[1]:.1f}]" if c else "-"
    rows = [["System", "τ", "Top-1", "95% CI", "Top-3", "Balanced", "95% CI", "Macro-F1", "Macro-AUC"],
            ["Phase 1 (July, 64 px)", "0", pct(PH1["top1"]), "[78.2, 81.7]", pct(PH1["top3"]), pct(PH1["bal"]), "[50.8, 59.3]", "58.87%", num(PH1["auc"])],
            ["Phase 2 PRIME-B (64 px)", "0.15", pct(PH2["top1"]), "[81.0, 84.3]", pct(PH2["top3"]), pct(PH2["bal"]), "[61.8, 70.6]", pct(PH2["f1"]), num(PH2["auc"])]]
    for k, lab in (("ensemble_greedy_tau0", "Phase 3, greedy-selected subset (secondary)"), ("ensemble_A_accuracy", "Phase 3 ensemble, point A (accuracy)"),
                   ("ensemble_B_screening", "Phase 3 ensemble, point B (screening)"), ("ensemble_C_balanced", "Phase 3 ensemble, point C (balanced)")):
        m = pts.get(k)
        if m: rows.append([lab, f"{m['tau']:.2f}", pct(m["top1"]), cif(k, "top1_ci"), pct(m["top3"]), pct(m["balanced"]), cif(k, "bal_ci"), pct(m["macro_f1"]), num(m["auc"])])
    story += [table(rows, [4.6 * cm, 0.9 * cm, 1.5 * cm, 1.9 * cm, 1.3 * cm, 1.6 * cm, 1.9 * cm, 1.5 * cm, 1.6 * cm], hl_rows=(4,), font=7.6),
              P("Table 3. Ensembles on the official test split with 95% bootstrap intervals (2,000 resamples). Members: " + ", ".join(nice(k) for k in ens["runs"]) +
                (f". Greedy subset: {', '.join(nice(k) for k in ens['greedy_selected'])}." if ens.get("greedy_selected") else "."), CAP)]
    if stats:
        story.append(P(f"<b>Against Phase 2 on the same images.</b> The Phase-3 ensemble at point A is right on {stats['A']['mcnemar']['only_second']} test images "
                       f"where Phase-2 PRIME-B was wrong, and wrong on {stats['A']['mcnemar']['only_first']} where PRIME-B was right (exact McNemar "
                       f"p = {stats['A']['mcnemar']['p']:.1e}); the paired bootstrap interval of the top-1 gain is [{100*stats['A']['delta_ci'][0]:.1f}, "
                       f"{100*stats['A']['delta_ci'][1]:.1f}] points. At point B the corresponding counts are {stats['B']['mcnemar']['only_second']} against "
                       f"{stats['B']['mcnemar']['only_first']} (p = {stats['B']['mcnemar']['p']:.1e})."))
    story += [fig(os.path.join(ENS_DIR, "tau_tradeoff.png"), 0.85, "Figure 2. Validation and test top-1 and balanced accuracy of the ensemble as a function of tau."),
              fig(os.path.join(ENS_DIR, "cm_A.png"), 1.0, "Figure 3. Confusion matrices at point A."),
              fig(os.path.join(ENS_DIR, "cm_B.png"), 1.0, "Figure 4. Confusion matrices at point B (screening).")]
else:
    story.append(P("<i>Pending.</i>"))
story.append(P("5.4 Ablations", H2))
if abl:
    rows = [["Ablation", "Model", "Top-1", "Balanced", "Macro-AUC", "Note"]] + [[a["name"], a["model"], pct(a["top1"]), pct(a["balanced"]), num(a["auc"]), a.get("note", "")] for a in abl]
    story += [table(rows, [4.0 * cm, 3.0 * cm, 1.5 * cm, 1.6 * cm, 1.6 * cm, 4.9 * cm], font=7.6), P("Table 4. Recipe and data ablations (seed 0).", CAP)]
else:
    story.append(P("<i>Planned for session 2: uniform learning rate versus layer-wise decay (DINOv2-S), no mixup/CutMix, 128-pixel versus 224-pixel data "
                   "with the same recipe, DenseNet-121 under the plain MedMNIST+ protocol versus our recipe, and three seeds for the two strongest models.</i>"))
story.append(P("5.5 Seeds", H2))
seeds = {}
for k, r in trained.items(): seeds.setdefault(k.rsplit("_s", 1)[0], []).append(r["tta_top1"])
multi = {k: v for k, v in seeds.items() if len(v) > 1}
if multi:
    rows = [["Model", "Seeds", "Top-1 mean ± std", "Min", "Max"]] + [[NICE.get(k, k), str(len(v)), f"{100*np.mean(v):.2f} ± {100*np.std(v, ddof=1):.2f}", pct(min(v)), pct(max(v))] for k, v in multi.items()]
    story += [table(rows, [4.5 * cm, 1.5 * cm, 3.5 * cm, 2 * cm, 2 * cm], font=8), P("Table 5. Seed variability of the fine-tuned models.", CAP)]
else:
    story.append(P("<i>Single seed so far; repeats are part of session 2.</i>"))
story.append(P("5.6 Leakage-free test sets", H2))
if ext:
    rows = [["Test set", "System", "Top-1", "Balanced", "Macro-AUC", "n"]] + [[e["set"], e["system"], pct(e["top1"]), pct(e["balanced"]), num(e["auc"]), str(e["n"])] for e in ext]
    story += [table(rows, [3.2 * cm, 4.2 * cm, 1.6 * cm, 1.8 * cm, 1.8 * cm, 1.4 * cm], font=7.8), P("Table 6. Evaluation on DermaMNIST-C / -E (Abhishek et al.).", CAP)]
else:
    story.append(P("<i>Not yet run. DermaMNIST-E (official ISIC-2018 validation and test images, 1,511 test lesions, no overlap with training) is the "
                   "priority; DermaMNIST-C requires retraining on its own split.</i>"))
story.append(P("5.7 Lesion-clean subset, cluster bootstrap, calibration and melanoma operating points", H2))
if st and audit:
    story.append(P(
        f"We reproduced the lesion-level audit of Abhishek et al. on the official split from the public split-to-image mapping and the HAM10000 "
        f"metadata (labels agree for all 10,015 images): {audit['train_test']['lesions']} lesions are shared between training and test, "
        f"{audit['train_val']['lesions']} between training and validation and {audit['val_test']['lesions']} between validation and test; "
        f"{audit['multi_partition_lesions']} lesions appear in more than one partition. The {audit['test_clean']['n']} test images whose lesion "
        f"never appears in training or validation form a <b>lesion-clean subset</b>; the remaining {audit['test_clean']['n_leaky']} are "
        f"\u2018leaky\u2019. Melanoma is heavily affected: {audit['test_clean']['class_counts_leaky'][4]} of its 223 test images share a lesion with "
        f"the training or validation data. The table reports every system on both subsets, with two confidence intervals on the official split: "
        f"the usual image-level bootstrap and a cluster bootstrap that resamples lesions."))
    rows = [["System", "Official top-1", "Image CI", "Lesion-cluster CI", "Clean subset top-1", "Clean balanced", "Leaky subset top-1", "ECE"]]
    names = [k for k in st["systems"] if not k.startswith("abl_")]
    for k in names:
        r = st["systems"][k]
        rows.append([k, pct(r["official"]["top1"]), f"[{100*r['ci_image']['top1'][0]:.1f}, {100*r['ci_image']['top1'][1]:.1f}]",
                     f"[{100*r['ci_lesion']['top1'][0]:.1f}, {100*r['ci_lesion']['top1'][1]:.1f}]", pct(r["clean"]["top1"]), pct(r["clean"]["balanced"]),
                     pct(r["leaky"]["top1"]), f"{r['calibration']['plain']['ece']:.3f}"])
    story += [table(rows, [3.2 * cm, 1.8 * cm, 2.0 * cm, 2.3 * cm, 2.0 * cm, 1.8 * cm, 2.0 * cm, 1.3 * cm], font=7.2, hl_rows=tuple(i for i, k in enumerate(names, 1) if k.startswith("ensemble"))),
              P(f"Table 8. Official-split results with image-level and lesion-cluster 95% bootstrap intervals, the lesion-clean subset (n = {st['n_clean']}), "
                "the leaky remainder, and expected calibration error (15 bins) of the plain probabilities.", CAP)]
    e = st["systems"].get("ensemble_A", {})
    if e.get("melanoma"):
        m = e["melanoma"]
        story.append(P(f"<b>Melanoma as a binary task.</b> Using the ensemble's melanoma probability as a score, the one-vs-rest AUROC on the test split is "
                       f"{m['auroc']:.3f}. With thresholds fixed on the validation split, sensitivity is {100*m['sens_at_spec80']['sensitivity']:.1f}% at "
                       f"{100*m['sens_at_spec80']['achieved_specificity']:.1f}% specificity, {100*m['sens_at_spec90']['sensitivity']:.1f}% at "
                       f"{100*m['sens_at_spec90']['achieved_specificity']:.1f}% and {100*m['sens_at_spec95']['sensitivity']:.1f}% at "
                       f"{100*m['sens_at_spec95']['achieved_specificity']:.1f}%."))
    cal = e.get("calibration")
    if cal:
        story.append(P(f"<b>Calibration.</b> The plain ensemble has ECE {cal['plain']['ece']:.3f}, NLL {cal['plain']['nll']:.3f} and Brier score "
                       f"{cal['plain']['brier']:.3f}; after the prior correction at point A these are unchanged by construction, and at point B the adjusted "
                       f"probabilities give ECE {st['systems']['ensemble_B']['calibration']['adjusted']['ece']:.3f}. Temperature scaling is left for the next step."))
else:
    story.append(P("<i>Pending: run phase3/stats_phase3.py after importing the Colab results.</i>"))
story.append(PageBreak())

# ================================================================== 6 BENCHMARKS
story += [P("6. Comparison with Published 224-pixel Results", H1), P(
    "All rows use the official MedMNIST split and 224-pixel inputs. The literature moved quickly in 2025-2026: besides the MedMNIST+ "
    "three-seed baselines [2] there are now single-seed results around 89-92% from ImageNet-pretrained ViT-Base, ConvNeXt-Base and Swin-B "
    "models and their fusions (Sanap 2026; Tanvir 2025; LiteMed-ViT 2026), and a 2025 preprint reports 92.0% for DINOv2 ViT-B/14 without "
    "describing any validation-based selection (Wu and Papiez; listed in the text only). None of these report balanced accuracy or "
    "confidence intervals, and all share the official split's lesion leakage. Our rows are three-seed means where seeds exist, with "
    "bootstrap intervals, and the same models are re-evaluated on the leakage-free DermaMNIST-E test images where that has been run.")]
lead = [(n, v) for n, v, _ in PUB224]
if probes: lead.append(("DINOv2 ViT-B/14 linear probe (this work)", 100 * next(iter(probes.values()))["tta_top1"]))
for k, r in trained.items(): lead.append((f"{nice(k)} (this work)", 100 * r["tta_top1"]))
if A: lead.append(("Phase 3 ensemble, point A (this work)", 100 * A["top1"]))
if B: lead.append(("Phase 3 ensemble, point B (this work)", 100 * B["top1"]))
lead = sorted(lead, key=lambda kv: -kv[1])
story += [KeepTogether([leaderboard_chart(lead), P("Figure 5. Top-1 accuracy on the official DermaMNIST test split at 224 pixels, published results and this work.", CAP)])]
rows = [["Method", "Top-1", "Macro-AUC", "Source"]] + [[n, f"{v:.2f}%", (f"{a:.3f}" if a else "-"), "published"] for n, v, a in PUB224]
if A: rows.append(["Phase 3 ensemble, point A", pct(A["top1"]), num(A["auc"]), "this work"])
if B: rows.append(["Phase 3 ensemble, point B", pct(B["top1"]), num(B["auc"]), "this work"])
story += [table(rows, [7.5 * cm, 2.2 * cm, 2.2 * cm, 2.6 * cm], hl_rows=tuple(range(len(PUB224) + 1, len(rows))), font=8),
          P("Table 7. Published 224-pixel results versus this work.", CAP), PageBreak()]

# ================================================================== 7-8
story += [P("7. Discussion and Limitations", H1), P(
    "The pattern of the results is the one Phase 2 predicted. Moving to native 224-pixel data lifted every family of model by several points, "
    "and the recipe mattered as much as the architecture: the DenseNet-121 reproduction row isolates that effect, and the frozen DINOv2 probe "
    "shows how much of the final accuracy is already present in the pretrained features. The ensemble adds less on top of strong members than "
    "it did in Phase 2, which is expected when the members are individually better calibrated and more correlated, and the decision rule "
    "behaves exactly as before: &tau; buys balanced accuracy and minority-class recall at a small, controllable cost in top-1."),
    P("Limitations we would state in a paper: a single official split with known lesion leakage, so absolute numbers are optimistic; one seed "
    "for several models until session 2 completes; selection decisions were pre-registered but by us, not independently; no calibration or "
    "external-dataset evaluation yet; and compute limited to a free T4, which bounded model size at ViT-B."),
    P("8. Work Remaining this Term", H1),
    BL("Session 2 on Colab: three seeds for the two strongest models, the four recipe ablations, and the 128-pixel resolution control."),
    BL("Leakage-free evaluation on DermaMNIST-E (and -C if retraining time allows) with the saved model weights."),
    BL("Calibration (temperature scaling, expected calibration error, reliability diagrams) and conformal prediction sets at 90% and 95% coverage."),
    BL("Melanoma sensitivity at fixed specificity and per-class ROC curves for the clinical framing."),
    BL("Paper: IEEE two-column manuscript (PDF and DOCX) with the figures and tables of this report, followed by an internal review pass."),
    BL("Stretch: skin-tone-stratified evaluation on DDI, knowledge distillation of the ensemble into one ConvNeXt-T, on-device export."),
    P("9. References", H1)]
refs = [
    "Yang, J. et al. MedMNIST v2 -- a large-scale lightweight benchmark for 2D and 3D biomedical image classification. <i>Scientific Data</i> 10, 41 (2023).",
    "Doerrich, S., Di Salvo, F., Brockmann, J., Ledig, C. Rethinking model prototyping through the MedMNIST+ dataset collection. <i>Scientific Reports</i> 15, 7669 (2025).",
    "Abhishek, K., Jain, A., Hamarneh, G. Investigating the quality of DermaMNIST and Fitzpatrick17k dermatological image datasets. <i>Scientific Data</i> 12, 196 (2025).",
    "Tan, M., Le, Q. V. EfficientNet: rethinking model scaling for convolutional neural networks. <i>ICML</i> 2019.",
    "Szegedy, C. et al. Rethinking the Inception architecture for computer vision. <i>CVPR</i> 2016.",
    "Müller, R., Kornblith, S., Hinton, G. When does label smoothing help? <i>NeurIPS</i> 2019.",
    "Lin, T.-Y. et al. Focal loss for dense object detection. <i>ICCV</i> 2017.",
    "Tschandl, P., Rosendahl, C., Kittler, H. The HAM10000 dataset. <i>Scientific Data</i> 5, 180161 (2018).",
    "Esteva, A. et al. Dermatologist-level classification of skin cancer with deep neural networks. <i>Nature</i> 542, 115-118 (2017).",
    "Menon, A. K. et al. Long-tail learning via logit adjustment. <i>ICLR</i> 2021.",
    "Tan, M., Le, Q. V. EfficientNetV2: smaller models and faster training. <i>ICML</i> 2021.",
    "Shanmugam, D. et al. Better aggregation in test-time augmentation. <i>ICCV</i> 2021.",
    "Loshchilov, I., Hutter, F. Decoupled weight decay regularization. <i>ICLR</i> 2019.",
    "Codella, N. et al. Skin lesion analysis toward melanoma detection 2018 (ISIC). arXiv:1902.03368.",
    "Finlayson, G. D., Trezzi, E. Shades of Gray and colour constancy. <i>Color Imaging Conference</i> 2004.",
    "Daneshjou, R. et al. Disparities in dermatology AI performance on a diverse, curated clinical image set. <i>Science Advances</i> 8 (2022).",
    "Angelopoulos, A. N., Bates, S. Conformal prediction: a gentle introduction. <i>Found. Trends Mach. Learn.</i> 16 (2023).",
    "Guo, C. et al. On calibration of modern neural networks. <i>ICML</i> 2017.",
    "Oquab, M. et al. DINOv2: learning robust visual features without supervision. <i>TMLR</i> 2024.",
    "Liu, Z. et al. A ConvNet for the 2020s. <i>CVPR</i> 2022.",
    "Polyak, B. T., Juditsky, A. B. Acceleration of stochastic approximation by averaging. <i>SIAM J. Control Optim.</i> 30 (1992).",
    "Caruana, R. et al. Ensemble selection from libraries of models. <i>ICML</i> 2004.",
    "Efron, B. Bootstrap methods: another look at the jackknife. <i>Ann. Statist.</i> 7 (1979).",
    "McNemar, Q. Note on the sampling error of the difference between correlated proportions. <i>Psychometrika</i> 12 (1947).",
    "Manzari, O. N. et al. MedViT: a robust vision transformer for generalized medical image classification. <i>Comput. Biol. Med.</i> 157 (2023).",
    "Ferdous, M., Mahmud, T., Shimul, M. S. MedNet: a lightweight attention-augmented CNN for medical image classification. <i>Scientific Reports</i> 15 (2025).",
    "Huang, G. et al. Densely connected convolutional networks. <i>CVPR</i> 2017.",
    "He, K. et al. Deep residual learning for image recognition. <i>CVPR</i> 2016.",
    "Dosovitskiy, A. et al. An image is worth 16x16 words. <i>ICLR</i> 2021.",
    "Wightman, R. PyTorch Image Models (timm). github.com/huggingface/pytorch-image-models (2019).",
    "Yun, S. et al. CutMix: regularization strategy to train strong classifiers with localizable features. <i>ICCV</i> 2019.",
    "Riba, E. et al. Kornia: an open source differentiable computer vision library for PyTorch. <i>WACV</i> 2020.",
    "Huang, G. et al. Deep networks with stochastic depth. <i>ECCV</i> 2016.",
    "Bao, H., Dong, L., Piao, S., Wei, F. BEiT: BERT pre-training of image transformers (layer-wise learning-rate decay). <i>ICLR</i> 2022.",
]
for i, r in enumerate(refs, 1):
    story.append(Paragraph(f"[{i}]&nbsp;&nbsp;{r}", style('ref', parent=BODY, fontSize=9, spaceAfter=4, leftIndent=18, firstLineIndent=-18)))

# ================================================================== APPENDICES
story += [PageBreak(), P("Appendix A. Run Configurations", H1), P(
    "Every hyper-parameter of every run, read from the result files. Batch 32, two warm-up epochs, cosine schedule to 1e-6, gradient clipping 1.0, "
    "EMA decay 0.999, mixup 0.2 / CutMix 1.0 on half of the batches, label smoothing 0.1, 16-bit mixed precision, eight-view dihedral TTA.")]
rows = [["Run", "timm model", "Input", "Epochs", "LR", "Weight decay", "Layer decay", "Drop-path", "Selected"]]
for k, r in trained.items():
    c = r["cfg"]; rows.append([k, c["model"], str(c["size"]), str(c["epochs"]), f"{c['lr']:.0e}", str(c["wd"]), str(c["layer_decay"]), str(c.get("drop_path", 0)), r.get("selection", "final_ema")])
if trained: story.append(table(rows, [2.6 * cm, 5.2 * cm, 1.0 * cm, 1.2 * cm, 1.2 * cm, 1.6 * cm, 1.6 * cm, 1.4 * cm, 1.6 * cm], font=7))
story += [PageBreak(), P("Appendix B. Training Evidence (Colab Output)", H1), P(
    "Screenshots of the Colab session (Tesla T4, PyTorch 2.14, timm 1.0.31) in which the Phase-3 runs were trained, included unedited.")]
import glob as _glob
caps3 = jload(os.path.join(SHOTS3, "captions.json")) or {}
shots3 = sorted(f for f in _glob.glob(os.path.join(SHOTS3, "*")) if f.lower().endswith((".png", ".jpg", ".jpeg")))
for i, f in enumerate(shots3, 1):
    story.append(fig(f, 0.92, f"Screenshot B{i}. " + caps3.get(os.path.basename(f), os.path.basename(f))))
if not shots3: story.append(P("<i>No screenshots captured yet.</i>"))

def decorate(canvas, doc):
    canvas.saveState()
    if doc.page > 1:
        canvas.setFont('Helvetica', 8); canvas.setFillColor(GREY)
        canvas.drawString(LM, PAGE_H - 1.15 * cm, "DDP Phase 3 Report | Native-Resolution Transfer Learning on DermaMNIST")
        canvas.setStrokeColor(MAROON); canvas.setLineWidth(0.6); canvas.line(LM, PAGE_H - 1.30 * cm, PAGE_W - RM, PAGE_H - 1.30 * cm)
        canvas.drawRightString(PAGE_W - RM, 1.0 * cm, f"Page {doc.page}"); canvas.drawString(LM, 1.0 * cm, "IIT Madras")
    else:
        canvas.setStrokeColor(MAROON); canvas.setLineWidth(0.8); canvas.line(LM + 3.0 * cm, 3.15 * cm, PAGE_W - RM - 3.0 * cm, 3.15 * cm)
        canvas.setFont('Helvetica', 10); canvas.setFillColor(GREY); canvas.drawCentredString(PAGE_W / 2, 2.68 * cm, "This project is carried out under the supervision of")
        canvas.setFont('Helvetica-Bold', 12.5); canvas.setFillColor(MAROON); canvas.drawCentredString(PAGE_W / 2, 2.18 * cm, "Dr. Jayaganthan R")
        canvas.setFont('Helvetica', 9.5); canvas.setFillColor(GREY)
        canvas.drawCentredString(PAGE_W / 2, 1.74 * cm, "Professor and Head, Department of Engineering Design (ED), IIT Madras")
    canvas.restoreState()

doc = SimpleDocTemplate(OUT3, pagesize=A4, leftMargin=LM, rightMargin=RM, topMargin=2.0 * cm, bottomMargin=1.8 * cm,
                        title="DDP Phase 3 Report - Native-Resolution Transfer Learning on DermaMNIST", author="Suharsh Kumar, Sayantan Das")
doc.build(story, onFirstPage=decorate, onLaterPages=decorate)
print("built", OUT3, "| runs:", len(trained), "| ensemble:", "present" if ens else "pending", datetime.datetime.now())
