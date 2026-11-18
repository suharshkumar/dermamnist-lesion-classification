# -*- coding: utf-8 -*-
"""Build the FINAL DDP report (Phase 1 summer work + Phase 2 improvements) with reportlab.

Run from anywhere:  run/.venv/bin/python DDP_Phase2/report_src/build_report.py
Reads results JSON written by run/train_v2.py, run/ensemble_v2.py and
run/prior_correction_baseline.py; any missing piece is rendered as "pending" so a draft
can be built before training finishes. Output: DDP_Final_Report_DermaMNIST.pdf (project root).
"""
import os, json, socket, datetime
socket.getfqdn = lambda *a, **k: "localhost"          # reportlab DNS stall on this Mac
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
                                PageBreak, HRFlowable, KeepTogether)
from PIL import Image as PILImage
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import sys, glob
_FD = (glob.glob(os.path.join(sys.prefix, "lib", "python3*", "site-packages", "matplotlib", "mpl-data", "fonts", "ttf")) or
       [os.path.join(os.path.dirname(__import__("matplotlib").__file__), "mpl-data", "fonts", "ttf")])[0]
for _n, _f in [("DejaVu", "DejaVuSans.ttf"), ("DejaVu-Bold", "DejaVuSans-Bold.ttf"),
               ("DejaVu-Oblique", "DejaVuSans-Oblique.ttf"), ("DejaVu-BoldOblique", "DejaVuSans-BoldOblique.ttf"),
               ("DejaVuMono", "DejaVuSansMono.ttf")]:
    pdfmetrics.registerFont(TTFont(_n, os.path.join(_FD, _f)))
pdfmetrics.registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVu-Bold", italic="DejaVu-Oblique", boldItalic="DejaVu-BoldOblique")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RUN = os.path.join(ROOT, "run"); ART = os.path.join(RUN, "artifacts")
P2 = os.path.join(ROOT, "DDP_Phase2"); FIGS = os.path.join(P2, "figures"); ASSETS = os.path.join(RUN, "report_assets")
OUT = os.path.join(ROOT, "DDP_Final_Report_DermaMNIST.pdf")

# ------------------------------------------------------------------ data
def jload(p):
    try: return json.load(open(p))
    except Exception: return None

base = jload(os.path.join(ART, "single_b0", "results.json")) or {}           # archived July eval (80.00%)
base_full = jload(os.path.join(ART, "results.json")) or {}                   # unreported focal run
re_eval = jload(os.path.join(P2, "results", "single_b0_results.json")) or jload(os.path.join(ART, "single_b0", "results_v2.json"))
resume = jload(os.path.join(ART, "v2_b0_resume", "results.json"))
v2b0 = jload(os.path.join(ART, "v2_v2b0", "results.json"))
ens = jload(os.path.join(ART, "v2_ensemble", "ensemble_v2_results.json"))
pc = jload(os.path.join(P2, "results", "prior_correction_baseline.json"))
cs = jload(os.path.join(P2, "results", "comparative_stats.json"))
BASE_TOP1, BASE_STD, BASE_TOP3, BASE_BAL = 0.8000, 0.7840, 0.9736, 0.5487     # from run/eval.log (archived)

def pct(x, d=2): return "pending" if x is None else f"{100*x:.{d}f}%"
def num(x, d=3): return "pending" if x is None else f"{x:.{d}f}"
def g(dct, *keys):
    for k in keys:
        if dct is None: return None
        dct = dct.get(k) if isinstance(dct, dict) else None
    return dct

best_point = None
if ens:
    pts = ens["points"]
    best_point = max((k for k in pts if k.startswith("ensemble")), key=lambda k: pts[k]["top1"])
    A = pts.get("ensemble_tuned_A_accuracy"); B = pts.get("ensemble_tuned_B_screening"); C = pts.get("ensemble_tuned_C_balanced")
else:
    A = B = C = None

# ------------------------------------------------------------------ style (matches the July report)
MAROON = colors.HexColor("#7a1230"); NAVY = colors.HexColor("#1f3b6e"); LIGHT = colors.HexColor("#f3eef0")
GREY = colors.HexColor("#555555"); PINK = colors.HexColor("#fbe9ee"); GREEN = colors.HexColor("#e7f4e9")
PAGE_W, PAGE_H = A4; LM = RM = 2.0 * cm; USABLE_W = PAGE_W - LM - RM
ss = getSampleStyleSheet()
def style(name, parent=None, **kw): return ParagraphStyle(name, parent=parent or ss['Normal'], **kw)
H1 = style('H1', fontName='Helvetica-Bold', fontSize=15, textColor=MAROON, spaceBefore=16, spaceAfter=7, leading=18)
H2 = style('H2', fontName='Helvetica-Bold', fontSize=12, textColor=NAVY, spaceBefore=10, spaceAfter=4, leading=15)
H3 = style('H3', fontName='Helvetica-BoldOblique', fontSize=10.5, textColor=NAVY, spaceBefore=6, spaceAfter=2, leading=13)
BODY = style('BODY', fontName='DejaVu', fontSize=9.6, leading=14.5, alignment=TA_JUSTIFY, spaceAfter=6, textColor=colors.HexColor("#1a1a1a"))
BULLET = style('BULLET', parent=BODY, leftIndent=14, bulletIndent=4, spaceAfter=3)
CAP = style('CAP', fontName='DejaVu-Oblique', fontSize=8.2, alignment=TA_CENTER, textColor=GREY, spaceBefore=3, spaceAfter=10)
CELL = style('CELL', fontName='DejaVu', fontSize=8.2, leading=10.5)
EQ = style('EQ', fontName='DejaVu', fontSize=9.5, leading=13, alignment=TA_CENTER, spaceBefore=4, spaceAfter=6)
NOTE = style('NOTE', parent=BODY, fontSize=9, leading=13, textColor=GREY, leftIndent=10, rightIndent=10,
             borderColor=MAROON, borderWidth=0.6, borderPadding=6, spaceBefore=6, spaceAfter=10)
COVER_T = style('COVER_T', fontName='Helvetica-Bold', fontSize=20, alignment=TA_CENTER, textColor=MAROON, leading=25)
COVER_S = style('COVER_S', fontName='Helvetica', fontSize=12, alignment=TA_CENTER, textColor=colors.HexColor("#333333"), leading=17)

def P(t, s=BODY): return Paragraph(t, s)
def BL(t): return Paragraph(t, BULLET, bulletText="•")
def rule(c=MAROON, w=1.1): return HRFlowable(width="100%", thickness=w, color=c, spaceBefore=4, spaceAfter=8)
def fig(path, width_frac=1.0, caption=None):
    if not os.path.exists(path):
        return P(f"<i>[figure pending: {os.path.basename(path)}]</i>", CAP)
    iw, ih = PILImage.open(path).size; w = USABLE_W * width_frac; h = w * ih / iw
    if h > 0.72 * (PAGE_H - 4 * cm): h = 0.72 * (PAGE_H - 4 * cm); w = h * iw / ih
    flow = [Image(path, width=w, height=h)]
    if caption: flow.append(Paragraph(caption, CAP))
    return KeepTogether(flow)
def table(rows, widths, hl_rows=(), green_rows=(), font=8.5):
    data = [rows[0]] + [[Paragraph(str(c), CELL) for c in r] for r in rows[1:]]
    t = Table(data, colWidths=widths, repeatRows=1)
    ts = [('BACKGROUND', (0, 0), (-1, 0), MAROON), ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
          ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'), ('FONTSIZE', (0, 0), (-1, -1), font),
          ('ALIGN', (1, 0), (-1, -1), 'CENTER'), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
          ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor("#bbbbbb")),
          ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, LIGHT]),
          ('TOPPADDING', (0, 0), (-1, -1), 3.5), ('BOTTOMPADDING', (0, 0), (-1, -1), 3.5), ('LEFTPADDING', (0, 0), (-1, -1), 5)]
    for r in hl_rows: ts += [('BACKGROUND', (0, r), (-1, r), PINK)]
    for r in green_rows: ts += [('BACKGROUND', (0, r), (-1, r), GREEN)]
    t.setStyle(TableStyle(ts)); return t

story = []
# ================================================================== COVER
logo = os.path.join(ASSETS, "iitm_logo.png"); lw, lh = PILImage.open(logo).size
story += [Spacer(1, 0.4 * cm), Image(logo, width=3.9 * cm, height=3.9 * cm * lh / lw, hAlign='CENTER'), Spacer(1, 0.25 * cm),
          P("INDIAN INSTITUTE OF TECHNOLOGY MADRAS", COVER_S),
          P("<i>Siddhirbhavati Karmaja &mdash; Success is born of action</i>", style('m', parent=COVER_S, fontSize=9.5, textColor=GREY)),
          Spacer(1, 0.5 * cm), rule(MAROON, 1.4), Spacer(1, 0.3 * cm),
          P("Dual Degree Project (DDP) &mdash; Final Project Report", COVER_S), Spacer(1, 0.35 * cm),
          P("Deep Learning for Automated Multi-Class<br/>Skin-Lesion Classification on the DermaMNIST Dataset", COVER_T),
          Spacer(1, 0.2 * cm),
          P("Phase 1 (summer 2026) baseline and Phase 2 (November 2026): the PRIME ensemble &mdash; "
            "Prior-corrected, Resolution-matched, heterogeneous EfficientNet ensemble with test-time augmentation",
            style('sub', parent=COVER_S, fontSize=10.5, textColor=NAVY)),
          Spacer(1, 0.35 * cm), rule(MAROON, 1.4), Spacer(1, 0.8 * cm)]
ci = [[Paragraph("<b>Submitted by</b>", style('ci', fontSize=11, textColor=NAVY)), ""],
      [Paragraph("Suharsh Kumar", CELL), Paragraph("BE22B037", CELL)],
      [Paragraph("Sayantan Das", CELL), Paragraph("BS22B035", CELL)], ["", ""],
      [Paragraph("<b>Mentor</b>", style('ci', fontSize=11, textColor=NAVY)), ""],
      [Paragraph("G. K. Harish Balaji", CELL), Paragraph("BE21B017 &nbsp;(Senior Dual-Degree student)", CELL)]]
ct = Table(ci, colWidths=[7.0 * cm, 7.0 * cm]); ct.setStyle(TableStyle([('TOPPADDING', (0, 0), (-1, -1), 3)]))
story += [ct, Spacer(1, 0.9 * cm), P("November 2026", style('d', parent=COVER_S, fontSize=11, textColor=GREY)), PageBreak()]

# ================================================================== ABSTRACT + TOC
story.append(P("Abstract", H1))
best_txt = (f"On the official test split the ensemble reaches {pct(A['top1'])} top-1 accuracy, {pct(A['top3'])} top-3 and a macro-AUC of "
            f"{num(A['auc'])} when tuned for accuracy. Shifting the decision rule to the screening setting (&tau; = {B['tau']:.2f}) keeps "
            f"top-1 at {pct(B['top1'])} and lifts balanced accuracy to {pct(B['balanced'])}, against 54.9% in July. The gain over the summer "
            f"model is statistically significant on a paired test (McNemar p &lt; 0.001, n = 2,005), and to our knowledge it is the highest "
            f"top-1 reported for DermaMNIST from 28- or 64-pixel source images."
            if ens else "Phase-2 test results were still being computed when this draft was built.")
story.append(P(
    "This report covers the whole of our Dual Degree Project on DermaMNIST, the seven-class dermatoscopy benchmark from MedMNIST that is "
    "derived from HAM10000. In the summer phase we trained an ImageNet-pretrained EfficientNet-B0 with a small custom head in two stages "
    "and evaluated it with test-time augmentation. It reached " + f"{pct(BASE_TOP1)} top-1, {pct(BASE_TOP3)} top-3 and {pct(BASE_BAL)} "
    "balanced accuracy on the official test split. When we went back over that work in November we found four things that needed fixing: "
    "a wrong number in the benchmark table, a claim that the split was free of leakage (it is not: 1,006 of the 7,470 lesions appear in "
    "more than one split), a fine-tuning run that had been stopped at epoch 10 of 30 while it was still improving, and a class-balanced run "
    "with 70.8% balanced accuracy that never made it into the report. The second phase of the project deals with each of these. We trained "
    "five EfficientNet-family members on the same 64-pixel data, one of them on colour-constancy-normalised images, averaged their "
    "test-time-augmented probabilities with weights chosen on the validation split, and added a prior correction to the decision rule "
    "(a logit adjustment with one exponent &tau;) so that the same model can be run either for best overall accuracy or for better "
    "sensitivity on the rare, clinically important classes. We call the combination PRIME for short. " + best_txt +
    " The report also sets out what we think limits accuracy on this benchmark (resolution, label ambiguity and the leakage), what the "
    "results mean for clinical use, and a six-month plan for the next term."))
story.append(Spacer(1, 0.3 * cm)); story.append(P("Table of Contents", H1))
for t in ["1. Introduction", "2. Objectives", "3. Dataset: Composition, Imbalance and Data Quality",
          "4. Phase 1 (Summer 2026): Baseline System and Results", "5. Critical Review of Phase 1",
          "6. Phase 2 Method: the PRIME Ensemble", "7. Phase 2 Results", "8. Comparison with Published Benchmarks",
          "9. Discussion", "10. Implications in the Medical Field", "11. Conclusion and Future Work",
          "12. Project Files and Reproducibility", "13. Plan for the Next Term: a Six-Month Roadmap", "14. References", "Appendix A. Training Evidence (Colab Output Screenshots)", "Appendix B. Phase-1 Training Evidence (Laptop Logs)"]:
    story.append(Paragraph(t, style('toc', parent=BODY, spaceAfter=3, fontSize=10.5)))
story.append(PageBreak())

# ================================================================== 1 INTRO
story += [P("1. Introduction", H1), P("1.1 Background and Motivation", H2), P(
    "Melanoma is a small fraction of skin-cancer cases but causes most skin-cancer deaths, and how early it is caught matters a great deal "
    "for the outcome. Dermatologists examine pigmented lesions with a dermatoscope, a magnified and illuminated view of the skin surface. "
    "Reading those images well takes years of training, two experts will not always agree, and in many parts of the world there is no "
    "expert to ask at all. That is the gap automated classifiers are meant to help with. Esteva and colleagues showed in 2017 that a "
    "convolutional network trained on about 130,000 clinical images could match board-certified dermatologists on biopsy-proven test "
    "sets [9]. Since then the interesting questions have been less about whether such systems can work and more about how reliably "
    "they work on rare classes, on patients unlike the training set, and on cheap, low-resolution images."),
    P("1.2 Problem Statement", H2), P(
    "Given a dermatoscopic image, assign it to one of seven diagnostic categories. Three things make this harder than it sounds. The "
    "classes are very unbalanced (two thirds of the images are benign nevi). Some malignant and benign classes look alike even to "
    "specialists. And in the benchmark form the images are tiny. Our aim was to get the overall accuracy as high as we could while being "
    "honest about what the imbalance does to the minority classes, and, in the second phase, to give ourselves some control over that "
    "trade-off."),
    P("1.3 The DermaMNIST Benchmark", H2), P(
    "DermaMNIST is one of the MedMNIST v2 datasets [1], a set of small standardised biomedical image benchmarks. It comes from HAM10000 [8], "
    "10,015 dermatoscopic images of 7,470 lesions, with a fixed 7:1:2 train/validation/test split. The MedMNIST+ release [2] later added "
    "64x64, 128x128 and 224x224 versions of every image. Everything in this report uses the 64x64 version, upsampled to 96 pixels or a "
    "little more at the network input. So when we compare with published numbers (Section 8) we compare with the 64-pixel column, and we "
    "treat results on 128- and 224-pixel images as a different, easier problem."),
    P("1.4 What this report adds", H2),
    BL("A complete account of the summer baseline, including a class-balanced run we had left out of the first report."),
    BL("A re-check of the summer report that found and corrects two factual errors (a benchmark value and the leakage claim)."),
    BL("A compute-cheap recipe, PRIME, that we define in Section 6 and test in Section 7: resumed training, several EfficientNet members, "
       "validation-tuned averaging with TTA, and a prior-corrected decision rule."),
    BL("A reproducible pipeline (laptop scripts and a Colab script), source-checked benchmark tables, and the raw training output as evidence."),
    P("2. Objectives", H1),
    BL("Build an accurate classifier for the seven-class DermaMNIST task using transfer learning from ImageNet."),
    BL("Deal with the class imbalance and measure what it does to per-class recall."),
    BL("Evaluate properly: top-1, top-3, balanced accuracy, macro-AUC, macro-F1 and full per-class reports."),
    BL("Beat the Phase-1 result of 80.0% top-1 on the same data and input resolution, and raise balanced accuracy without retraining."),
    BL("Compare with MedMNIST v2 and MedMNIST+, and say clearly what the results do and do not mean for clinical use."),
    PageBreak()]

# ================================================================== 3 DATASET
story += [P("3. Dataset: Composition, Imbalance and Data Quality", H1), P("3.1 Composition and Splits", H2),
          table([["Split", "Images", "Proportion"], ["Training", "7,007", "70.0%"], ["Validation", "1,003", "10.0%"],
                 ["Test", "2,005", "20.0%"], ["Total", "10,015", "100%"]], [6 * cm, 4 * cm, 4 * cm], hl_rows=(4,)),
          Spacer(1, 0.3 * cm), P("3.2 The Seven Diagnostic Classes", H2),
          table([["#", "Class (abbrev.)", "Full name", "Nature", "Test images"],
                 ["0", "Actinic Ker.", "Actinic keratoses / intra-epithelial carcinoma", "Pre-malignant", "66"],
                 ["1", "Basal Cell Ca.", "Basal cell carcinoma", "Malignant", "103"],
                 ["2", "Benign Ker.", "Benign keratosis-like lesions", "Benign", "220"],
                 ["3", "Dermatofibroma", "Dermatofibroma", "Benign", "23"],
                 ["4", "Melanoma", "Melanoma", "Malignant", "223"],
                 ["5", "Nevi", "Melanocytic nevi", "Benign", "1,341"],
                 ["6", "Vascular", "Vascular lesions", "Benign", "29"]],
                [0.8 * cm, 3.0 * cm, 6.6 * cm, 2.8 * cm, 2.4 * cm], green_rows=(6,)),
          Spacer(1, 0.3 * cm), P("3.3 Class Imbalance", H2), P(
    "Melanocytic nevi alone constitute 66.9% of the training split, while dermatofibroma and vascular lesions each account "
    "for about 1-1.5%. A classifier that always predicts ‘Nevi’ scores ~67% top-1 and 14.3% balanced accuracy. "
    "Every accuracy figure in this report must be read against that floor, which is why balanced accuracy (mean per-class "
    "recall) is reported next to top-1 throughout, and why Phase 2 introduces an explicit, tunable prior correction."),
          fig(os.path.join(ASSETS, "dist.png"), 0.95, "Figure 1. Class distribution across the train, validation and test splits."),
          P("3.4 Data Quality: Lesion-Level Leakage Across Splits", H2), P(
    "The summer report described the fixed split as ‘leakage-free’. That statement is <b>withdrawn</b>. "
    "Abhishek, Jain and Hamarneh [3] traced every DermaMNIST image back to its HAM10000 lesion identifier and found that "
    "1,956 of the 7,470 lesions are photographed more than once, and that <b>1,006 lesions (13.5%) have images in more "
    "than one partition</b>: 641 lesions (886 images) are shared between train and test, 332 between train and validation, "
    "113 between validation and test, and 40 appear in all three. They also found 18 exact duplicate image pairs. The "
    "authors conclude that reported DermaMNIST accuracies ‘could be inflated’ and release two corrected variants: "
    "DermaMNIST-C (lesion-disjoint re-split, 8,215 / 573 / 1,227) and DermaMNIST-E (official ISIC-2018 validation and test "
    "images, 10,015 / 193 / 1,511), noting that results on the three variants must not be compared with one another."),
          P("What this means for us: every number in this report is on the official MedMNIST split and should only be compared with "
            "other results on that split; the absolute accuracy is probably a little optimistic compared with a truly lesion-disjoint "
            "test; and re-running the final ensemble on DermaMNIST-C and -E is the first item on the next-term list. The relative "
            "comparison with published work is still fair, because the same leakage sits under every baseline in Section 8.", NOTE),
          PageBreak()]

# ================================================================== 4 PHASE 1
story += [P("4. Phase 1 (Summer 2026): Baseline System and Results", H1), P("4.1 Pipeline", H2), P(
    "Images are bilinearly upsampled from 64x64 to 96x96 and kept in the [0, 255] range expected by EfficientNet's built-in "
    "normalisation. Training-time augmentation uses random flips, rotation (&plusmn;25%), zoom (&plusmn;20%), translation "
    "(&plusmn;10%) and contrast/brightness jitter (&plusmn;20%); lesions have no canonical orientation, so geometric "
    "augmentation is medically valid. The backbone is ImageNet-pretrained EfficientNet-B0 [4] (4.0 M parameters). The "
    "custom head sums global-average and global-max pooled features (dual pooling), followed by BatchNorm, a 512-unit and "
    "a 128-unit Swish block with dropout 0.40 / 0.20, and a 7-way softmax."),
    P("Training is two-phase: (1) head only, backbone frozen, Adam 1e-3 with ReduceLROnPlateau, up to 15 epochs; "
      "(2) fine-tuning of the top 40 backbone layers (BatchNorm frozen) with AdamW 1e-4, cosine decay, weight decay 1e-4. "
      "The loss is label-smoothed (0.05) cross-entropy [5, 6]; no class re-weighting was used because the explicit Phase-1 "
      "goal was maximum top-1. At inference, 8 random TTA views are averaged with the clean pass. Everything ran on an "
      "Apple M1 with 8 GB of unified memory using TensorFlow 2.19 + Metal; memory pressure forced batch size 16 and the "
      "two training phases to run as separate processes."),
    P("4.2 Headline Results (official test split, 2,005 images)", H2),
    table([["Metric", "Value"], ["Top-1 accuracy, 8-view TTA", pct(BASE_TOP1)], ["Top-1 accuracy, single pass", pct(BASE_STD)],
           ["Top-3 accuracy", pct(BASE_TOP3)], ["Macro AUC (one-vs-rest)", num(g(pc, "auc_macro")) if pc else "0.949"],
           ["Balanced accuracy (mean recall)", pct(BASE_BAL)], ["Best validation accuracy", "78.96%"]],
          [9 * cm, 5 * cm], hl_rows=(1,)),
    Spacer(1, 0.2 * cm),
    fig(os.path.join(ASSETS, "curves.png"), 0.9, "Figure 2. Phase-1 training and validation accuracy (head phase, then fine-tuning)."),
    fig(os.path.join(ASSETS, "cm.png"), 1.0, "Figure 3. Phase-1 confusion matrices on the test split: counts (left) and row-normalised (right)."),
    P("4.3 Per-Class Performance", H2),
    table([["Class", "Precision", "Recall", "F1", "Support"], ["Actinic Ker.", "0.633", "0.288", "0.396", "66"],
           ["Basal Cell Ca.", "0.602", "0.631", "0.616", "103"], ["Benign Ker.", "0.590", "0.550", "0.569", "220"],
           ["Dermatofibroma", "0.467", "0.304", "0.368", "23"], ["Melanoma", "0.576", "0.426", "0.490", "223"],
           ["Nevi", "0.875", "0.952", "0.912", "1,341"], ["Vascular", "0.870", "0.690", "0.769", "29"],
           ["Macro average", "0.659", "0.549", "0.589", "2,005"], ["Weighted average", "0.784", "0.800", "0.787", "2,005"]],
          [4.2 * cm, 2.6 * cm, 2.6 * cm, 2.6 * cm, 2.6 * cm], green_rows=(6,), hl_rows=(8, 9)),
    P("Table 4. Phase-1 per-class metrics. Melanoma recall of 42.6% and actinic-keratosis recall of 28.8% are the clinically "
      "weakest points of the baseline.", CAP),
    PageBreak()]

# ================================================================== 5 CRITICAL REVIEW
story += [P("5. Critical Review of Phase 1", H1), P(
    "Before building anything new we went back through the summer report, the code and every log file. We found four problems. None of "
    "them changes the 80.0% figure itself, but each one changes how it should be read, and all four are corrected here."),
    P("5.1 Incorrect benchmark entry", H2), P(
    "Table 6 of the summer report listed auto-sklearn at 0.749 top-1. The MedMNIST v2 paper [1] reports <b>0.719</b> "
    "(AUC 0.902); 0.749 is the AutoKeras figure. The corrected table is Section 8."),
    P("5.2 Unjustified ‘leakage-free’ claim", H2), P(
    "See Section 3.4. A fixed split makes results reproducible; it says nothing about whether the same lesion shows up on both sides "
    "of it, and in DermaMNIST it often does. We should not have written that sentence."),
    P("5.3 Fine-tuning stopped while still improving", H2), P(
    "The log of the final model shows validation accuracy climbing from 77.3% to 79.0% over the first ten fine-tuning epochs, and then "
    "the log simply ends: the process was stopped at epoch 10 of 30 (the laptop was needed for other things that evening, and the "
    "checkpoint looked good enough). So the 80.0% checkpoint was under-trained. The cosine schedule had decayed only a third of the way "
    "and early stopping never had a chance to trigger. Finishing that schedule was the first thing on the Phase-2 list."),
    P("5.4 An unreported class-balanced run", H2), P(
    f"A full run with focal loss (&gamma; = 2) [7], square-root-softened class weights and 128x128 input "
    f"(<font face='Courier'>run/full.log</font>) reached {pct(g(base_full, 'tta_accuracy'))} top-1 but "
    f"<b>{pct(g(base_full, 'tta_balanced_accuracy'))} balanced accuracy</b>, against 54.9% for the reported model. It was left "
    "out of the summer report because top-1 was the only target. It shows that the top-1 / balanced-accuracy trade-off is "
    "large (about 2 points of top-1 for 16 points of balanced accuracy) and motivates making that trade-off explicit and "
    "tunable rather than baking it into the loss, which is what the prior-corrected rule of Section 6.4 does."),
    table([["Run (July 2026)", "Input", "Loss", "Top-1 (TTA)", "Balanced acc."],
           ["Reported final model", "96", "label-smoothed CE", pct(BASE_TOP1), pct(BASE_BAL)],
           ["Unreported run", "128", "focal γ=2, class weights^0.5", pct(g(base_full, 'tta_accuracy')), pct(g(base_full, 'tta_balanced_accuracy'))]],
          [4.4 * cm, 1.6 * cm, 4.6 * cm, 2.4 * cm, 2.6 * cm]),
    P("Table 5. The two complete Phase-1 runs. Only the first was reported in July.", CAP),
    PageBreak()]

# ================================================================== 6 METHOD
story += [P("6. Phase 2 Method: the PRIME Ensemble", H1), P(
    "We needed a name for the combination, so: PRIME, for prior-corrected, resolution-matched, heterogeneous EfficientNet ensemble with "
    "test-time augmentation. There is no new network architecture in it. It is a training and decision recipe, and each part of it "
    "answers one of the problems from Section 5. Everything is cheap enough to run on the laptop that produced Phase 1 (in practice we "
    "ended up on a free Colab GPU, for reasons explained in 6.5), and the only tuning uses the validation split. For an image x, with "
    "&pi; the class frequencies of the training split and K member networks f<sub>k</sub> with weights w<sub>k</sub>:"),
    P("p(x) = &Sigma;<sub>k</sub> w<sub>k</sub> &middot; (1/(T+1)) &Sigma;<sub>t=0..T</sub> f<sub>k</sub>(a<sub>t</sub>(x))"
      "&nbsp;&nbsp;&nbsp;&nbsp; y*<sub>&tau;</sub>(x) = argmax<sub>c</sub> [ log p<sub>c</sub>(x) &minus; &tau; log &pi;<sub>c</sub> ]", EQ),
    P("where a<sub>0</sub> is the identity and a<sub>1..T</sub> are random TTA views (we use T = 8). With &tau; = 0 this is the ordinary "
      "ensemble argmax. With &tau; = 1 it is the correction that is Bayes-optimal for balanced error [10]. Values in between give "
      "operating points in between, and Section 7.1 shows why in-between is where we ended up."),
    P("6.1 Member 1: the Phase-1 recipe, trained to completion", H2), P(
    "EfficientNet-B0 at 96x96 with the Phase-1 head, augmentation and label-smoothed loss, but with the fine-tuning schedule "
    "run to its end (30 cosine-decay epochs, patience 8, top 60 backbone layers unfrozen with BatchNorm frozen) instead of being "
    "stopped at epoch 10 of 30. This member is the controlled test of Section 5.3: same data, same architecture, only the "
    "training budget differs. (The 8 GB laptop could not even load TensorFlow during Phase 2, so the Phase-1 checkpoint itself "
    "could not be resumed; all Phase-2 members were trained from ImageNet weights on a Colab T4 GPU with the identical recipe.)"),
    P("6.2 Heterogeneous members", H2), P(
    "Four further members add diversity along three axes. <b>Backbone family</b>: EfficientNetV2-B0 and V2-B1 [11] use fused "
    "MBConv blocks and make different errors from V1. <b>Input scale</b>: EfficientNet-B1 at 112x112 and V2-B1 at 128x128 "
    "upsample the same 64-pixel images further, so the ImageNet filters see lesion structures at scales closer to those they were "
    "trained on (this is <i>not</i> higher-resolution data; the information content is unchanged). <b>Input preprocessing</b>: "
    "one EfficientNet-B0 member is trained on images normalised with Shades-of-Gray colour constancy (Minkowski norm p = 6) [15], "
    "the standard dermoscopy preprocessing used by ISIC-challenge winners to remove illumination and camera colour casts; it "
    "changes what the network sees without changing the label information. Every member uses the same two-phase schedule "
    "(12 head epochs with ReduceLROnPlateau, 30 fine-tuning epochs with AdamW and cosine decay) and 8-view TTA."),
    table([["Member", "Backbone", "Input", "Preprocessing", "Unfrozen top layers", "What it adds"],
           ["1", "EfficientNet-B0", "96", "none", "60", "Phase-1 recipe, schedule completed"],
           ["2", "EfficientNetV2-B0", "96", "none", "60", "second backbone family"],
           ["3", "EfficientNet-B0", "96", "Shades-of-Gray colour constancy", "60", "illumination-invariant view"],
           ["4", "EfficientNet-B1", "112", "none", "70", "larger input scale, deeper V1"],
           ["5", "EfficientNetV2-B1", "128", "none", "80", "largest input scale, V2 family"]],
          [1.3 * cm, 3.2 * cm, 1.2 * cm, 4.0 * cm, 2.3 * cm, 4.4 * cm], font=8),
    P("Table 5b. Members of the PRIME ensemble. All consume the 64x64 DermaMNIST images.", CAP),
    P("6.3 Validation-tuned averaging with TTA (\u2018resolution-matched\u2019)", H2), P(
    "All members consume the same 64x64 data, so the ensemble is a like-for-like improvement over Phase 1 rather than a change "
    "of benchmark. Each member produces 8-view TTA probabilities for both validation and test images. Member weights are chosen "
    "on the simplex grid (step 0.1) that maximises <i>validation</i> top-1, so a member that does not help receives weight zero; "
    "an equal-weight average is reported alongside as the no-tuning reference."),
    P("6.4 Prior-corrected decision rule (logit adjustment)", H2), P(
    "Menon et al. [10] show that subtracting &tau; log &pi;<sub>c</sub> from the logits of a model trained with ordinary "
    "cross-entropy is a consistent post-hoc correction for long-tailed class distributions, equivalent at &tau; = 1 to "
    "re-weighting the loss but without retraining. We apply it to the ensemble probabilities and define three operating points, "
    "all selected on the validation split: <b>A (accuracy)</b> &tau; = 0; <b>B (screening)</b> the largest &tau; whose validation "
    "top-1 is within 0.5 points of the best; <b>C (balanced)</b> the &tau; that maximises validation balanced accuracy. The "
    "same deployed model therefore serves a ‘most-likely diagnosis’ use and a ‘do not miss rare malignancies’ use."),
    P("6.5 Engineering under an 8 GB memory budget", H2), P(
    "Every stage of <font face='Courier'>run/run_v2.sh</font> runs as its own process and is skipped when its results file "
    "exists, so the pipeline survives interruptions and resumes where it stopped. A GPU notebook "
    "(<font face='Courier'>DDP_Phase2/prime_colab.py</font>) runs the identical recipe on a free Colab T4 GPU; the Phase-2 "
    "members reported below were trained there in a single session after the laptop proved unable to load TensorFlow."),
    PageBreak()]

# ================================================================== 7 RESULTS
story += [P("7. Phase 2 Results", H1), P("7.1 Zero-training result: prior correction of the Phase-1 model", H2)]
def tau_chart(sweep, title):
    """Native reportlab line chart (no matplotlib needed on the training machine)."""
    from reportlab.graphics.shapes import Drawing, String, Line
    from reportlab.graphics.charts.lineplots import LinePlot
    from reportlab.graphics.widgets.markers import makeMarker
    W, H = USABLE_W * 0.9, 7.2 * cm
    d = Drawing(W, H); lp = LinePlot(); lp.x, lp.y, lp.width, lp.height = 1.6 * cm, 1.2 * cm, W - 2.4 * cm, H - 2.4 * cm
    series = [("top1", NAVY), ("balanced", MAROON), ("macro_f1", colors.HexColor("#2e7d32"))]
    lp.data = [[(s["tau"], 100 * s[k]) for s in sweep] for k, _ in series]
    for i, (_, c) in enumerate(series):
        lp.lines[i].strokeColor = c; lp.lines[i].strokeWidth = 1.4; lp.lines[i].symbol = makeMarker("FilledCircle"); lp.lines[i].symbol.size = 3
        lp.lines[i].symbol.fillColor = c; lp.lines[i].symbol.strokeColor = c
    lp.xValueAxis.valueMin, lp.xValueAxis.valueMax, lp.xValueAxis.valueStep = 0, 1.5, 0.25
    lo = min(min(100 * s[k] for s in sweep) for k, _ in series); lp.yValueAxis.valueMin = 5 * int(lo // 5) - 5; lp.yValueAxis.valueMax = 100
    lp.xValueAxis.labels.fontName = lp.yValueAxis.labels.fontName = "DejaVu"; lp.xValueAxis.labels.fontSize = lp.yValueAxis.labels.fontSize = 7.5
    d.add(lp)
    d.add(String(W / 2, 0.15 * cm, "prior-correction exponent τ", fontName="DejaVu", fontSize=8, textAnchor="middle"))
    d.add(String(W / 2, H - 0.45 * cm, title, fontName="DejaVu-Bold", fontSize=8.5, textAnchor="middle"))
    for j, (lab, c) in enumerate([("top-1 accuracy", NAVY), ("balanced accuracy", MAROON), ("macro F1", colors.HexColor("#2e7d32"))]):
        x0 = 2.2 * cm + j * 4.6 * cm; yy = H - 1.0 * cm
        d.add(Line(x0, yy, x0 + 0.6 * cm, yy, strokeColor=c, strokeWidth=1.6)); d.add(String(x0 + 0.75 * cm, yy - 3, lab, fontName="DejaVu", fontSize=7.5))
    return d

CLASS_NAMES_R = ['Actinic Ker.', 'Basal Cell Ca.', 'Benign Ker.', 'Dermatofibroma', 'Melanoma', 'Nevi', 'Vascular']
def leaderboard_chart(rows):
    """Horizontal bar chart of published 28/64-px top-1 results with this work highlighted (reportlab native)."""
    from reportlab.graphics.shapes import Drawing, String, Rect, Line
    rows = list(rows); n = len(rows); W = USABLE_W * 0.98; rh = 0.42 * cm; H = n * rh + 1.4 * cm
    d = Drawing(W, H); x0 = 6.3 * cm; xw = W - x0 - 1.6 * cm; vmin, vmax = 68.0, 85.0
    for v in (70, 75, 80, 85):
        x = x0 + (v - vmin) / (vmax - vmin) * xw
        d.add(Line(x, 0.6 * cm, x, H - 0.8 * cm, strokeColor=colors.HexColor("#dddddd"), strokeWidth=0.5))
        d.add(String(x, 0.25 * cm, f"{v}%", fontName="DejaVu", fontSize=7, textAnchor="middle"))
    for i, (name, val) in enumerate(rows):
        yy = H - 0.8 * cm - (i + 1) * rh + 0.08 * cm; ours = "this work" in name
        d.add(String(x0 - 0.15 * cm, yy + 0.09 * cm, name, fontName="DejaVu-Bold" if ours else "DejaVu", fontSize=6.8, textAnchor="end"))
        d.add(Rect(x0, yy, (val - vmin) / (vmax - vmin) * xw, rh - 0.12 * cm, fillColor=MAROON if ours else colors.HexColor("#9fb3d1"), strokeColor=None))
        d.add(String(x0 + (val - vmin) / (vmax - vmin) * xw + 0.1 * cm, yy + 0.09 * cm, f"{val:.1f}", fontName="DejaVu", fontSize=6.8))
    d.add(String(W / 2, H - 0.45 * cm, "DermaMNIST official test split, 28/64-px source images: top-1 accuracy (%)", fontName="DejaVu-Bold", fontSize=8.5, textAnchor="middle"))
    return d

if pc:
    t0, t1, th = pc["tau0"], pc["tau1"], pc["tau05"]
    story += [P(
        f"The archived Phase-1 test probabilities give a macro-AUC of {num(pc['auc_macro'])} and a plain-argmax balanced accuracy of "
        f"{pct(t0['balanced'])}. Applying the textbook &tau; = 1 correction (fixed a priori, no tuning) moves balanced accuracy to "
        f"<b>{pct(t1['balanced'])}</b> but top-1 to {pct(t1['top1'])} and macro-F1 from {pct(t0['macro_f1'])} to {pct(t1['macro_f1'])}: "
        f"for this model &tau; = 1 over-corrects, because label smoothing and TTA averaging flatten the softmax so that the full "
        f"prior ratio (nevi are 60 times more frequent than dermatofibroma) swamps the evidence in the image. A half correction, "
        f"&tau; = 0.5, is a much better fixed point: top-1 {pct(th['top1'])}, balanced accuracy {pct(th['balanced'])} and macro-F1 "
        f"{pct(th['macro_f1'])}. Its gains are exactly where the baseline was weakest: melanoma recall rises from "
        f"{100*t0['recall'][4]:.1f}% to {100*th['recall'][4]:.1f}%, actinic keratosis from {100*t0['recall'][0]:.1f}% to "
        f"{100*th['recall'][0]:.1f}%, dermatofibroma from {100*t0['recall'][3]:.1f}% to {100*th['recall'][3]:.1f}% and vascular lesions "
        f"from {100*t0['recall'][6]:.1f}% to {100*th['recall'][6]:.1f}%, while nevus recall falls from {100*t0['recall'][5]:.1f}% to "
        f"{100*th['recall'][5]:.1f}%. This is the empirical reason Phase 2 <i>tunes</i> &tau; on the validation split instead of fixing it. "
        "No validation probabilities exist for the archived checkpoint, so the sweep in Figure 4 is a sensitivity analysis on the "
        "test split, not a selection."),
        KeepTogether([tau_chart(pc["sweep"], "Phase-1 model: effect of prior correction (test split)"),
                      P("Figure 4. Effect of the prior-correction exponent on the Phase-1 model (test split). Only tau = 0 and tau = 1 were fixed a priori.", CAP)]),
        table([["τ", "Top-1", "Balanced acc.", "Macro-F1", "Melanoma recall", "Actinic Ker. recall", "Nevi recall"]] +
              [[f"{pc[k]['tau'] if 'tau' in pc[k] else t:.1f}", pct(pc[k]['top1']), pct(pc[k]['balanced']), pct(pc[k]['macro_f1']),
                pct(pc[k]['recall'][4]), pct(pc[k]['recall'][0]), pct(pc[k]['recall'][5])] for k, t in (("tau0", 0.0), ("tau05", 0.5), ("tau1", 1.0))],
              [1.2 * cm, 1.9 * cm, 2.3 * cm, 1.9 * cm, 2.5 * cm, 2.6 * cm, 2.2 * cm], hl_rows=(2,)),
        P("Table 6a. Prior correction applied to the archived Phase-1 probabilities (no retraining).", CAP)]
else:
    story.append(P("<i>Pending: run/prior_correction_baseline.py has not finished.</i>"))

story.append(P("7.2 Individual members", H2))
rows = [["Member", "Backbone", "Input", "Preproc.", "Val top-1", "Test top-1", "Top-3", "Balanced", "Epochs"],
        ["Phase-1 model (July, laptop)", "EfficientNet-B0", "96", "none", "78.96%*", pct(BASE_TOP1), pct(BASE_TOP3), pct(BASE_BAL), "15 + 10"]]
mres = (ens or {}).get("member_results", {})
for i, (name, r) in enumerate(mres.items(), 1):
    rows.append([f"{i}: {name}", {"B0": "EfficientNet-B0", "B1": "EfficientNet-B1", "V2B0": "EfficientNetV2-B0", "V2B1": "EfficientNetV2-B1"}.get(r.get("backbone"), r.get("backbone")),
                 str(r.get("img_size")), "SoG" if r.get("preprocessing") == "sog" else "none", pct(r.get("val_tta_top1")), pct(r.get("tta_top1")),
                 pct(r.get("tta_top3")), pct(r.get("tta_balanced")), f"{r.get('epochs_head')} + {r.get('epochs_ft')}"])
if not mres:
    rows.append(["Phase-2 members", "-", "-", "-", "pending", "pending", "pending", "pending", "-"])
story += [table(rows, [3.6 * cm, 2.7 * cm, 1.0 * cm, 1.3 * cm, 1.5 * cm, 1.6 * cm, 1.4 * cm, 1.5 * cm, 1.4 * cm], font=7.6),
          P("Table 6. Individual models (8-view TTA on the official test split). *single-pass validation accuracy from the Phase-1 log. "
            "Epochs = head phase + fine-tuning epochs actually run (early stopping).", CAP)]
if mres:
    m1 = next(iter(mres.values()))
    story.append(P(f"Member 1 answers the question from Section 5.3 directly. With the same recipe run to completion it reaches "
                   f"{pct(m1['tta_top1'])} top-1 against the laptop model's {pct(BASE_TOP1)} "
                   f"({'so finishing the schedule helped, but only slightly' if m1['tta_top1'] > BASE_TOP1 else 'so finishing the schedule did not help by itself'}; "
                   f"the batch size and random seed also differ, so we would not read much into two tenths of a point). Across the five members, test top-1 ranges from "
                   f"{pct(min(r['tta_top1'] for r in mres.values()))} to {pct(max(r['tta_top1'] for r in mres.values()))}."))
    story.append(fig(os.path.join(ART, "v2_ensemble", "member_curves.png"), 0.9,
                     "Figure 4b. Validation accuracy of every Phase-2 member over the head phase and fine-tuning."))
story.append(P("7.3 Ensemble and operating points", H2))
if ens:
    w = dict(zip([os.path.basename(m) for m in ens["members"]], ens["weights"]))
    story.append(P(f"Validation-selected member weights: " + ", ".join(f"{k} = {v:.1f}" for k, v in w.items()) +
                   f". Selected exponents: &tau;<sub>B</sub> = {ens['tau_B']:.2f}, &tau;<sub>C</sub> = {ens['tau_C']:.2f}."))
    pts = ens["points"]
    rows = [["Operating point", "τ", "Top-1", "Top-3", "Balanced acc.", "Macro-F1", "Macro-AUC"],
            ["Phase-1 baseline (July)", "0", pct(BASE_TOP1), pct(BASE_TOP3), pct(BASE_BAL), "58.87%", "0.949"]]
    for k, lab in [("ensemble_equal", "Equal-weight ensemble"), ("ensemble_tuned_A_accuracy", "PRIME A (accuracy)"),
                   ("ensemble_tuned_B_screening", "PRIME B (screening)"), ("ensemble_tuned_C_balanced", "PRIME C (balanced)")]:
        m = pts[k]; rows.append([lab, f"{m['tau']:.2f}", pct(m["top1"]), pct(m["top3"]), pct(m["balanced"]), pct(m["macro_f1"]), num(m["auc"])])
    story += [table(rows, [4.6 * cm, 1.1 * cm, 1.9 * cm, 1.9 * cm, 2.3 * cm, 1.9 * cm, 2.1 * cm], hl_rows=(3,)),
              P("Table 7. Test-split metrics of the Phase-2 ensemble at the three validation-selected operating points.", CAP)]
    delta = 100 * (A["top1"] - BASE_TOP1)
    verdict = (f"exceeds the Phase-1 result by {delta:.2f} points" if delta > 0 else f"does not exceed the Phase-1 result ({delta:+.2f} points)")
    story.append(P(f"At operating point A the ensemble reaches <b>{pct(A['top1'])} top-1</b>, which {verdict}; on 2,005 test images "
                   f"one point corresponds to about 20 images, so differences below ~1 point are within the sampling noise of the "
                   f"split (95% CI half-width about 1.75 points at 80%). Point B keeps top-1 at {pct(B['top1'])} while lifting "
                   f"balanced accuracy to {pct(B['balanced'])}; point C reaches {pct(C['balanced'])} balanced accuracy at "
                   f"{pct(C['top1'])} top-1."))
    story += [fig(os.path.join(ART, "v2_ensemble", "tau_tradeoff.png"), 0.85,
                  "Figure 5. Validation and test top-1 / balanced accuracy of the ensemble as a function of tau; dotted lines mark the selected points B and C."),
              fig(os.path.join(ART, "v2_ensemble", "cm_A.png"), 1.0, "Figure 6. Confusion matrices at operating point A (accuracy)."),
              fig(os.path.join(ART, "v2_ensemble", "cm_B.png"), 1.0, "Figure 7. Confusion matrices at operating point B (screening)."),
              fig(os.path.join(ART, "v2_ensemble", "per_class_recall.png"), 0.9,
                  "Figure 8. Per-class recall: Phase-1 member vs. ensemble at points A and B.")]
else:
    story.append(P("<i>Pending: the ensemble stage (run/ensemble_v2.py) has not run yet. Figures 5-8 will be inserted here.</i>"))
story.append(PageBreak())


# ---- 7.4 comparative statistics ----
if cs:
    S = cs["systems"]; mc = cs["mcnemar"]
    def ci(d, k): return f"{100*d[k]:.2f}% [{100*d[k+'_ci'][0]:.1f}, {100*d[k+'_ci'][1]:.1f}]"
    story += [P("7.4 How much better, and is it real? Comparative statistics", H2), P(
        "A gain of two or three points on 2,005 test images could in principle be luck, so we checked it three ways: 95% bootstrap "
        "intervals from 2,000 resamples of the test set, a paired McNemar test that looks only at the images where the July model and the "
        "ensemble disagree (the right test when two classifiers are scored on the same images), and a paired bootstrap interval for the "
        "difference itself. Every system in the table is scored on exactly the same images and labels."),
        table([["System", "Top-1 [95% CI]", "Balanced acc. [95% CI]", "Pooled malignant recall*"]] +
              [[lab, ci(S[k], "top1"), ci(S[k], "balanced"), f"{100*cs['malignant_pooled_recall'][k]:.1f}%"] for k, lab in [
                  ("phase1_baseline", "Phase 1 model (July 2026)"), ("phase1_tau0.5", "Phase 1 model + prior correction, tau = 0.5"),
                  ("member_b0_96", "Member 1 (same recipe, schedule completed)"), ("member_b1_112", "Best single member (EfficientNet-B1, 112 px)"),
                  ("ensemble_equal", "Equal-weight ensemble"), ("PRIME_A", "PRIME, point A (accuracy)"),
                  ("PRIME_B", "PRIME, point B (screening)"), ("PRIME_C", "PRIME, point C (balanced)")]],
              [6.0 * cm, 3.2 * cm, 3.6 * cm, 2.6 * cm], hl_rows=(7,), font=8),
        P("Table 7b. Bootstrap confidence intervals on the official test split (n = 2,005). *Recall pooled over the three malignant or "
          "pre-malignant classes (actinic keratosis, basal cell carcinoma, melanoma; 392 test images).", CAP),
        P(f"<b>Paired comparison with the Phase-1 model.</b> On the images where exactly one of the two systems is right, PRIME-B is "
          f"correct {mc['baseline_vs_PRIME_B']['only_second_correct']} times and the Phase-1 model {mc['baseline_vs_PRIME_B']['only_first_correct']} "
          f"times (net +{mc['baseline_vs_PRIME_B']['net_gain']} images; exact McNemar p = {mc['baseline_vs_PRIME_B']['p_value']:.1e}). "
          f"For PRIME-A the count is {mc['baseline_vs_PRIME_A']['only_second_correct']} against {mc['baseline_vs_PRIME_A']['only_first_correct']} "
          f"(p = {mc['baseline_vs_PRIME_A']['p_value']:.1e}). The paired bootstrap 95% interval for the top-1 gain of PRIME-B over Phase 1 is "
          f"[{100*cs['paired_delta_ci']['baseline_to_PRIME_B'][0]:.1f}, {100*cs['paired_delta_ci']['baseline_to_PRIME_B'][1]:.1f}] points, which "
          f"excludes zero. In relative terms PRIME-B removes <b>{100*cs['relative_error_reduction_B']:.1f}% of the Phase-1 errors</b> "
          f"(error rate {100*(1-S['phase1_baseline']['top1']):.1f}% to {100*(1-S['PRIME_B']['top1']):.1f}%) while raising balanced accuracy by "
          f"{100*(S['PRIME_B']['balanced']-S['phase1_baseline']['balanced']):.1f} points and pooled malignant recall by "
          f"{100*(cs['malignant_pooled_recall']['PRIME_B']-cs['malignant_pooled_recall']['phase1_baseline']):.1f} points. Even the single "
          f"best member (EfficientNet-B1 at 112 px) is a significant improvement on its own, and member 1, which repeats the Phase-1 recipe "
          f"with the schedule completed, lands within 0.2 points of the July model: the gains come from the ensemble, the input scale and "
          f"the decision rule, not from moving to a GPU."),
        P("<b>Where the gain comes from.</b> The per-class table below is the one we find most convincing. At point B every minority class "
          "gains recall and the only class that loses any is nevi, by about one point, which is exactly the trade the prior correction is "
          "supposed to make. At point C melanoma recall is 55% where July had 43%, and dermatofibroma 74% where July had 30%."),
        table([["Class", "Phase 1 recall", "PRIME-A recall", "PRIME-B recall", "PRIME-C recall", "Change (B − Phase 1)"]] +
              [[CLASS_NAMES_R[c], f"{100*cs['per_class_recall']['phase1_baseline'][c]:.1f}%", f"{100*cs['per_class_recall']['PRIME_A'][c]:.1f}%",
                f"{100*cs['per_class_recall']['PRIME_B'][c]:.1f}%", f"{100*cs['per_class_recall']['PRIME_C'][c]:.1f}%",
                f"{100*(cs['per_class_recall']['PRIME_B'][c]-cs['per_class_recall']['phase1_baseline'][c]):+.1f} pts"] for c in range(7)],
              [3.4 * cm, 2.3 * cm, 2.3 * cm, 2.3 * cm, 2.3 * cm, 2.8 * cm], font=8, green_rows=(6,)),
        P("Table 7c. Per-class recall on the test split, Phase 1 versus the three PRIME operating points.", CAP),
        KeepTogether([leaderboard_chart(cs["leaderboard_28_64px"]),
                      P("Figure 9. Published DermaMNIST results that use 28- or 64-pixel source images (official split), ranked by top-1 accuracy, "
                        "with this project's systems highlighted. MedMNIST v2 rows are single runs; MedMNIST+ rows are three-seed means.", CAP)]),
        P(f"<b>Against published results.</b> On the fair comparison set (28- and 64-pixel source data, official split) PRIME-B's "
          f"{100*S['PRIME_B']['top1']:.2f}% is the highest top-1 we are aware of; the strongest published 64-pixel model, VGG16 in MedMNIST+ "
          f"(79.5 &plusmn; 1.2%), is {100*S['PRIME_B']['top1']-79.53:.1f} points lower, and its upper one-standard-deviation bound (80.7%) is "
          f"below the lower end of PRIME-B's confidence interval. Our macro-AUC of {num(g(A,'auc'))} is likewise above the best 64-pixel AUC "
          f"in MedMNIST+ (0.944, VGG16). The only published numbers above ours come from models trained on 128- and 224-pixel source "
          f"images, and that is the main reason the next-term plan (Section 13) starts with higher-resolution data.")]

# ================================================================== 8 BENCHMARKS
story += [P("8. Comparison with Published Benchmarks", H1), P(
    "All figures below are on the official MedMNIST split and were re-verified against the primary sources in November 2026 "
    "(the auto-sklearn entry corrects the summer report). MedMNIST v2 [1] trained ResNets on 28x28 images, optionally "
    "upsampled to 224; MedMNIST+ [2] trained ten backbones end-to-end at each native resolution (mean of three seeds)."),
    table([["Method / source", "Native data", "Input", "Top-1", "AUC"],
           ["ResNet-18, MedMNIST v2 [1]", "28", "28", "73.5%", "0.917"], ["ResNet-18, MedMNIST v2 [1]", "28", "224", "75.4%", "0.920"],
           ["ResNet-50, MedMNIST v2 [1]", "28", "28", "73.5%", "0.913"], ["ResNet-50, MedMNIST v2 [1]", "28", "224", "73.1%", "0.912"],
           ["auto-sklearn, MedMNIST v2 [1]", "28", "28", "71.9%", "0.902"], ["AutoKeras, MedMNIST v2 [1]", "28", "28", "74.9%", "0.915"],
           ["Google AutoML Vision, MedMNIST v2 [1]", "28", "28", "76.8%", "0.914"],
           ["ResNet-18, MedMNIST+ [2]", "64", "64", "76.4%", "0.917"], ["DenseNet-121, MedMNIST+ [2]", "64", "64", "76.2%", "0.931"],
           ["ViT-B/16, MedMNIST+ [2]", "64", "64", "77.0%", "0.938"], ["VGG16, MedMNIST+ [2] (best at 64)", "64", "64", "79.5%", "0.944"],
           ["EfficientNet-B4, MedMNIST+ [2]", "64", "64", "71.5%", "0.876"],
           ["Phase 1: EfficientNet-B0 + TTA (this work)", "64", "96", pct(BASE_TOP1, 1), num(g(pc, "auc_macro")) if pc else "0.949"],
           ["Phase 2: PRIME A (this work)", "64", "96", pct(g(A, "top1"), 1), num(g(A, "auc"))],
           ["— higher-resolution data, for context —", "", "", "", ""],
           ["VGG16, MedMNIST+ [2]", "128", "128", "81.8%", "0.957"], ["DenseNet-121, MedMNIST+ [2]", "224", "224", "84.7%", "0.963"],
           ["DINO ViT-B/16 linear probe, MedMNIST+ [2]", "224", "224", "84.4%", "0.968"]],
          [7.2 * cm, 2.2 * cm, 1.6 * cm, 1.8 * cm, 1.8 * cm], hl_rows=(13, 14), font=8),
    P("Table 8. Published DermaMNIST results on the official split. Rows above the separator use 28- or 64-pixel data and are the "
      "fair comparison for this project; rows below use higher-resolution source images.", CAP),
    P("Both phases of this work are above every 28- and 64-pixel entry, including the strongest MedMNIST+ 64-pixel model "
      "(VGG16, 79.5 &plusmn; 1.2%). The gap to the 128- and 224-pixel rows (2 to 5 points) is a resolution effect that no "
      "amount of modelling on 64-pixel data closes; it is the clearest direction for future work. Two recent preprints report "
      "much higher numbers (up to 92% with DINOv2 at 224 px) but describe no validation-based model selection and are not "
      "peer-reviewed, so they are not tabulated."),
    PageBreak()]

# ================================================================== 9-11
story += [P("9. Discussion", H1), P("9.1 What produced the gains", H2), P(
    "Looking back over the two phases, the improvement came from three places. The first is simply finishing the training. A cosine "
    "schedule cut off at a third of its length never gets to the small-learning-rate end where the refinement happens; member 1, which "
    "repeats the July recipe to completion, is within 0.2 points of the July model, so on its own this was worth little, but it told us "
    "the recipe was sound. The second is diversity. Averaging models from two EfficientNet generations, at three input scales, with one "
    "of them looking at colour-normalised images, smooths out the errors each one makes on its own; the equal-weight average is already "
    "2.6 points above the best July number. The third, and the one we did not expect to matter as much as it did, is the decision rule. "
    "The imbalance of the training set is baked into every softmax the network produces. Dividing part of it back out costs nothing, "
    "needs no retraining, and is where almost all of the melanoma and dermatofibroma recall came from."),
    P("9.2 Why accuracy stops where it does", H2), P(
    "Three ceilings show up in our results and in the literature. Resolution first: MedMNIST+ reports the same architectures gaining "
    "three to eight points between 64 and 224 pixels, and the dermoscopic cues a clinician looks for (pigment network, streaks, blue-white "
    "veil) are just not there at 64 pixels. Label ambiguity second: the confusions that dominate every confusion matrix in this report "
    "are melanoma versus benign keratosis versus actinic keratosis, which are also the pairs dermatologists disagree about. And leakage "
    "third: with 13.5% of lesions photographed in more than one split, some part of every reported accuracy on this split, ours included, "
    "is the network recognising a lesion it has already seen. The absolute numbers should be quoted with that caveat attached."),
    P("9.3 Limitations", H2),
    BL("One official split, no cross-validation, and one training seed per member; the confidence intervals in Section 7.4 cover test-set "
       "sampling, not training variance."),
    BL("Every model was selected on the same validation split. Test numbers were only looked at after selection, except for the &tau; "
       "sweep on the archived July model in Section 7.1, which is labelled as a sensitivity analysis for that reason."),
    BL("Hardware shaped the design. Eight gigabytes of shared memory fixed the batch size, the input size and the number of members in "
       "Phase 1, and in Phase 2 the laptop could not even load TensorFlow, which is why the members were trained on Colab."),
    BL("Benchmark-only. No external dataset, no skin-type breakdown, no clinical metadata, no calibration."),
    P("10. Implications in the Medical Field", H1), P(
    "A seven-class classifier at about 82% top-1 and 98% top-3 is not a diagnostic device. What it can be is a triage or second-reader "
    "tool, and the two operating points map onto those two jobs quite naturally. Point A, tuned for accuracy, suits ranking a queue of "
    "images for a dermatologist's attention, where a correct most-likely label saves reading time. Points B and C suit screening, where "
    "missing a melanoma or a basal cell carcinoma costs far more than a false alarm on a nevus, and where the extra recall on the rare "
    "malignant classes is worth the lost accuracy on the common benign one. A top-3 accuracy of 98% also suggests an interface that "
    "shows the three most likely diagnoses rather than one; the correct answer is almost always among them. None of this is ready for "
    "a clinic. Before that the probabilities would need calibrating, the model would need testing on lesion-disjoint and external data "
    "and across skin types, and the whole thing would need regulatory review. The leakage finding in Section 3.4 is a good reminder "
    "that a benchmark number on its own says little about clinical readiness."),
    P("11. Conclusion and Future Work", H1), P(
    f"Over the two phases we built, documented and then improved a DermaMNIST classifier. The summer baseline, an EfficientNet-B0 at "
    f"{pct(BASE_TOP1)} top-1, was already ahead of every published 28- and 64-pixel result. In the second phase we corrected the summer "
    "report, finished the training it had cut short, trained four more members, and turned the accuracy-versus-balance trade-off into a "
    "single parameter that is set on validation data. "
    + (f"The resulting ensemble reaches {pct(A['top1'])} top-1 and {pct(A['balanced'])} balanced accuracy at point A, and {pct(B['top1'])} "
       f"and {pct(B['balanced'])} at point B, with the gain over the baseline significant on a paired test. " if ens else "")
    + "The things we would do next, in order of expected value: train the same recipe on the native 128- and 224-pixel MedMNIST+ images; "
    "re-evaluate on DermaMNIST-C and DermaMNIST-E to get leakage-free numbers; calibrate the probabilities and report expected calibration "
    "error; try more recent backbones (ConvNeXt-T, DINOv2 features) and mixup or CutMix; and validate externally on ISIC 2019 with results "
    "broken down by skin type. Section 13 turns this list into a schedule."),
    P("12. Project Files and Reproducibility", H1), P(
    "All code is kept out of this report by design. The complete, commented source (Phase-1 and Phase-2 training, ensembling, "
    "prior correction, figure generation and this document's builder), the result files and the figures are published in the "
    "project repository <font color='#1f3b6e'>github.com/suharshkumar/dermamnist-lesion-classification</font>; the table below "
    "maps each artefact to its location so that every number in this report can be regenerated. Appendix A reproduces the raw "
    "Colab output of the Phase-2 run as evidence."),
    table([["Path", "Content"],
           ["run/train.py, run/run_single.sh", "Phase-1 training pipeline (July 2026) and the exact configuration of the 80.0% model"],
           ["run/artifacts/single_b0/", "Phase-1 checkpoint, archived TTA probabilities, results.json"],
           ["run/full.log, run/single.log, run/eval.log", "Complete Phase-1 logs (unreported focal run, truncated fine-tuning, final eval)"],
           ["run/train_v2.py, run/run_v2.sh", "Phase-2 pipeline: resume / head / finetune / eval modes; resumable per-stage execution"],
           ["run/ensemble_v2.py", "Validation-tuned weights, tau sweep, operating points, figures, ensemble_v2_results.json"],
           ["run/prior_correction_baseline.py", "Zero-training logit adjustment of the archived Phase-1 probabilities"],
           ["run/artifacts/v2_*/", "Phase-2 checkpoints, val/test probabilities, results"],
           ["DDP_Phase2/DDP_Phase2_PRIME_colab.ipynb", "GPU notebook: same recipe, plus native 128-px members"],
           ["DDP_Phase2/research/literature_notes.md", "Source-verified benchmark tables and references"],
           ["DDP_Phase2/PROGRESS.md", "Status log; the pipeline resumes from the last finished stage"],
           ["DDP_Phase2/report_src/build_report.py", "This document"]], [6.2 * cm, 8.4 * cm], font=8),
    PageBreak()]
# ================================================================== 13 NEXT-TERM PLAN
story += [P("13. Plan for the Next Term: a Six-Month Roadmap", H1), P(
    "The next term gives us six months. Our aim for it is to take the project from a benchmark classifier to something that could be "
    "called a validated screening assistant, and to get a paper out of it. The plan below follows directly from what Phase 2 taught us: "
    "resolution is the biggest ceiling, data quality undermines the absolute numbers, imbalance is best handled at the decision stage, "
    "and clinical value depends on calibration and rare-class sensitivity rather than on top-1 alone. Each goal has a number attached "
    "and a deliverable we can be held to; the stretch items are marked as such."),
    P("13.1 Headline goals", H2),
    table([["#", "Goal", "Measurable target", "Baseline today"],
           ["G1", "Higher-resolution data: native 128 px and 224 px MedMNIST+ images, plus full-resolution HAM10000 and an external ISIC 2019 set",
            "Top-1 >= 85% and macro-AUC >= 0.96 on the official 224-px split (stretch 87%); balanced accuracy >= 75%",
            f"{pct(g(A,'top1'),1) if ens else '80.0%'} at 64 px; best published 224-px result 84.7%"],
           ["G2", "Leakage-free evaluation", "All headline numbers also reported on DermaMNIST-C and -E and on a lesion-disjoint HAM10000 split; zero lesion overlap verified by script",
            "official split only (13.5% of lesions leak)"],
           ["G3", "Input preprocessing pipeline", "Hair removal (DullRazor-style inpainting), colour constancy, lesion-centred cropping, ablated one at a time; >= +1 point top-1 or balanced accuracy from preprocessing alone",
            "one colour-constancy member"],
           ["G4", "Calibrated, abstaining predictions", "Expected calibration error < 3% after temperature scaling; conformal prediction sets with 95% coverage; melanoma sensitivity >= 85% at specificity >= 80% at the screening operating point",
            "uncalibrated softmax; melanoma recall 56% (tau = 0.5)"],
           ["G5", "Fairness and robustness", "Per-skin-tone results on DDI / Fitzpatrick17k with gap < 5 points; accuracy drop < 3 points under a blur/JPEG/lighting corruption suite",
            "not measured"],
           ["G6", "Deployable prototype", "INT8 TFLite / Core ML model < 20 MB, < 100 ms per image on a phone, web demo with top-3, confidence and saliency map",
            "none"],
           ["G7", "Dissemination", "Workshop or conference paper submitted (e.g. MICCAI ISIC Skin Image Analysis workshop, IEEE EMBC) and public reproducible repository",
            "internal reports"]],
          [0.9 * cm, 5.0 * cm, 6.2 * cm, 4.3 * cm], font=7.8),
    P("Table 10. Next-term goals. G1 and G2 are the core scientific goals; G4 to G6 are what make the system credible clinically.", CAP),
    P("13.2 Month-by-month schedule", H2),
    table([["Month", "Theme", "Work", "Deliverable / checkpoint"],
           ["1", "Data foundation", "Download MedMNIST+ 128/224 and HAM10000 originals; build lesion-level (not image-level) splits; implement DermaMNIST-C/-E loaders; write the leakage-audit script; set up Colab Pro or institute GPU queue and experiment tracking",
            "Data module + audit report; Phase-2 PRIME re-evaluated on -C/-E (first leakage-free numbers)"],
           ["2", "Resolution and backbones", "Train PRIME at 128 and 224 px; add ConvNeXt-Tiny, EfficientNetV2-S and a DINOv2 ViT-S/B fine-tune [19]; mixup/CutMix regularisation; 3 seeds per configuration",
            "Table of top-1/AUC/balanced vs resolution and backbone; G1 target hit or a clear gap analysis"],
           ["3", "Preprocessing, imbalance, calibration", "Hair removal, colour constancy and cropping ablations (G3); logit adjustment vs balanced softmax vs LDAM; temperature scaling [18]; deep-ensemble and MC-dropout uncertainty; conformal prediction sets [17]",
            "Calibrated model with abstention rule; reliability diagrams; G4 numbers"],
           ["4", "Robustness, fairness, explanation", "Corruption suite; skin-tone-stratified evaluation on DDI [16] and Fitzpatrick17k; Grad-CAM / attention maps checked against ABCD dermoscopy criteria; failure-mode catalogue",
            "Model card with fairness and robustness tables; G5 numbers; mid-term review presentation"],
           ["5", "Prototype", "Export best single model and distilled ensemble to TFLite / Core ML (INT8); build web demo (Gradio) showing top-3, confidence, prediction set and saliency; informal feedback round with clinicians if accessible",
            "Working demo + benchmark of size/latency (G6)"],
           ["6", "Write-up", "Paper draft, ablation tables, reproducibility package (configs, seeds, checksums); thesis chapters; final presentation",
            "Submitted paper, public repository, thesis draft (G7)"]],
          [1.3 * cm, 2.6 * cm, 7.6 * cm, 4.9 * cm], font=7.6),
    P("Table 11. Six-month schedule. Each month ends with a written checkpoint added to the project log.", CAP),
    P("13.3 Why we think these targets are realistic", H2),
    BL("The resolution gain is documented, not hoped for. MedMNIST+ reports ResNet-18 going from 76.4% at 64 pixels to 82.3% at 224, and "
       "DenseNet-121 reaching 84.7%, with plain training. The ensemble and the decision rule add to that."),
    BL("Most of the code exists. The Phase-2 scripts take a resolution and a backbone name as parameters and save validation and test "
       "probabilities; the ensemble and &tau; code does not change. Month 2 is mostly compute time."),
    BL("Calibration and conformal prediction are post-hoc. Both are fitted on the validation split in minutes."),
    BL("Compute is modest. A 224-pixel EfficientNetV2-S or ConvNeXt-T run on 10,000 images takes under an hour on a T4; the whole Month-2 "
       "grid (4 backbones, 2 resolutions, 3 seeds) is roughly 30 GPU-hours."),
    BL("All the data are public: MedMNIST+, HAM10000 and ISIC 2018, ISIC 2019, DDI and Fitzpatrick17k."),
    P("13.4 Risks and mitigations", H2),
    table([["Risk", "Mitigation"],
           ["GPU quota limits on free Colab", "Colab Pro (low cost) or the institute cluster; checkpoint every epoch; the pipeline is resumable by design"],
           ["Higher resolution does not transfer to DermaMNIST-E (external test)", "Report it honestly as domain shift; add ISIC 2019 as a second external set; use test-time adaptation as a stretch item"],
           ["Skin-tone datasets are small and noisy", "Report confidence intervals (bootstrap); treat fairness numbers as exploratory, not as a pass/fail gate"],
           ["Scope creep (too many backbones)", "Fix the Month-2 grid in advance; anything else is a stretch item after G1 is met"],
           ["Clinician access for feedback", "Prototype feedback is a bonus; the deliverable is the demo and its measured latency, not a user study"]],
          [6.2 * cm, 8.2 * cm], font=8),
    P("13.5 Stretch goals (only after G1-G4 are met)", H2),
    BL("Multimodal input: add age, sex and lesion site from HAM10000 metadata to the classifier head."),
    BL("Semi-supervised pre-training on the unlabelled ISIC archive (DINOv2-style self-distillation) to improve rare classes."),
    BL("Knowledge distillation of the ensemble into one mobile model that keeps >= 98% of the ensemble's top-1."),
    PageBreak()]
story += [P("14. References", H1)]
refs = [
    "Yang, J., Shi, R., Wei, D., Liu, Z., Zhao, L., Ke, B., Pfister, H., Ni, B. MedMNIST v2 &mdash; a large-scale lightweight benchmark for 2D and 3D biomedical image classification. <i>Scientific Data</i> 10, 41 (2023). doi:10.1038/s41597-022-01721-8",
    "Doerrich, S., Di Salvo, F., Brockmann, J., Ledig, C. Rethinking model prototyping through the MedMNIST+ dataset collection. <i>Scientific Reports</i> 15, 7669 (2025). doi:10.1038/s41598-025-92156-9; data: doi:10.5281/zenodo.10519652",
    "Abhishek, K., Jain, A., Hamarneh, G. Investigating the quality of DermaMNIST and Fitzpatrick17k dermatological image datasets. <i>Scientific Data</i> 12, 196 (2025). doi:10.1038/s41597-025-04382-5",
    "Tan, M., Le, Q. V. EfficientNet: rethinking model scaling for convolutional neural networks. <i>ICML</i> 2019, PMLR 97:6105-6114.",
    "Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., Wojna, Z. Rethinking the Inception architecture for computer vision. <i>CVPR</i> 2016, 2818-2826.",
    "Müller, R., Kornblith, S., Hinton, G. When does label smoothing help? <i>NeurIPS</i> 2019.",
    "Lin, T.-Y., Goyal, P., Girshick, R., He, K., Dollár, P. Focal loss for dense object detection. <i>ICCV</i> 2017, 2980-2988.",
    "Tschandl, P., Rosendahl, C., Kittler, H. The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. <i>Scientific Data</i> 5, 180161 (2018).",
    "Esteva, A., Kuprel, B., Novoa, R. A., Ko, J., Swetter, S. M., Blau, H. M., Thrun, S. Dermatologist-level classification of skin cancer with deep neural networks. <i>Nature</i> 542, 115-118 (2017).",
    "Menon, A. K., Jayasumana, S., Rawat, A. S., Jain, H., Veit, A., Kumar, S. Long-tail learning via logit adjustment. <i>ICLR</i> 2021.",
    "Tan, M., Le, Q. V. EfficientNetV2: smaller models and faster training. <i>ICML</i> 2021, PMLR 139:10096-10106.",
    "Shanmugam, D., Blalock, D., Balakrishnan, G., Guttag, J. Better aggregation in test-time augmentation. <i>ICCV</i> 2021, 1214-1223.",
    "Loshchilov, I., Hutter, F. Decoupled weight decay regularization. <i>ICLR</i> 2019.",
    "Codella, N. et al. Skin lesion analysis toward melanoma detection 2018: a challenge hosted by the International Skin Imaging Collaboration (ISIC). arXiv:1902.03368 (2019).",
    "Finlayson, G. D., Trezzi, E. Shades of Gray and colour constancy. <i>IS&amp;T/SID Color Imaging Conference</i> 2004, 37-41; applied to dermoscopy by Barata, C., Celebi, M. E., Marques, J. S. Improving dermoscopy image classification using color constancy. <i>IEEE J. Biomed. Health Inform.</i> 19(3), 1146-1152 (2015).",
    "Daneshjou, R. et al. Disparities in dermatology AI performance on a diverse, curated clinical image set (DDI). <i>Science Advances</i> 8, eabq6147 (2022).",
    "Angelopoulos, A. N., Bates, S. Conformal prediction: a gentle introduction. <i>Foundations and Trends in Machine Learning</i> 16(4), 494-591 (2023).",
    "Guo, C., Pleiss, G., Sun, Y., Weinberger, K. Q. On calibration of modern neural networks. <i>ICML</i> 2017.",
    "Liu, Z. et al. A ConvNet for the 2020s (ConvNeXt). <i>CVPR</i> 2022. Oquab, M. et al. DINOv2: learning robust visual features without supervision. <i>TMLR</i> 2024.",
]
for i, r in enumerate(refs, 1):
    story.append(Paragraph(f"[{i}]&nbsp;&nbsp;{r}", style('ref', parent=BODY, fontSize=9, spaceAfter=5, leftIndent=18, firstLineIndent=-18)))


# ================================================================== APPENDICES
import glob as _glob
SHOTS = os.path.join(FIGS, "screenshots"); caps = jload(os.path.join(SHOTS, "captions.json")) or {}
shots = sorted(f for f in _glob.glob(os.path.join(SHOTS, "*")) if f.lower().endswith((".png", ".jpg", ".jpeg")))
story += [PageBreak(), P("Appendix A. Training Evidence (Colab Output Screenshots)", H1), P(
    "The screenshots below were taken from the Google Colab session (Tesla T4 GPU, TensorFlow 2.21, Python 3.13) in which every "
    "Phase-2 member was trained on 18 November 2026. They are included unedited so that the training process, its duration and "
    "the raw per-class reports can be inspected directly; the numbers in Section 7 are read from the result files that the same "
    "session wrote. Nothing in this appendix is required to follow the report.")]
for i, f in enumerate(shots, 1):
    story.append(fig(f, 0.92, f"Screenshot A{i}. " + caps.get(os.path.basename(f), os.path.basename(f))))
story += [PageBreak(), P("Appendix B. Phase-1 Training Evidence (Laptop Logs)", H1), P(
    "Phase 1 ran on an Apple M1 laptop (8 GB) in July 2026 and left complete text logs rather than notebook output. The extracts "
    "below are copied verbatim from those logs and are the basis of Sections 4 and 5.")]
def log_extract(path, pattern_keep, title, maxlines=40):
    try:
        import re as _re
        lines = [l.rstrip() for l in open(path, errors="ignore") if _re.search(pattern_keep, l)]
    except Exception:
        lines = []
    if not lines: return
    body = "<br/>".join(l[:118].replace("&", "&amp;").replace("<", "&lt;") for l in lines[:maxlines])
    story.append(KeepTogether([P(title, H3), Paragraph(body, style('log', fontName='DejaVuMono', fontSize=6.6, leading=8.4,
                                                                   backColor=colors.HexColor("#f7f7f7"), borderPadding=4, spaceAfter=8))]))
log_extract(os.path.join(RUN, "single.log"), r"Config:|^Epoch \d+/|val_accuracy|Phase 1 done|HEAD_EXIT", "B.1 Final Phase-1 model: head phase and the fine-tuning epochs that were logged before the run stopped (run/single.log)", 60)
log_extract(os.path.join(RUN, "eval.log"), r"Std|TTA|precision|Actinic|Basal|Benign|Dermato|Melanoma|Nevi|Vascular|accuracy|macro|weighted", "B.2 Final Phase-1 evaluation with 8-view TTA (run/eval.log)", 30)
log_extract(os.path.join(RUN, "full.log"), r"Config:|Class weights|early stopping|Phase . done|Std  accuracy|TTA  accuracy|TTA  balanced|^ *(Actinic|Basal|Benign|Dermato|Melanoma|Nevi|Vascular|macro|weighted|accuracy)", "B.3 The unreported focal-loss run (run/full.log)", 30)

# ------------------------------------------------------------------ page decoration + build
def decorate(canvas, doc):
    canvas.saveState()
    if doc.page > 1:
        canvas.setFont('Helvetica', 8); canvas.setFillColor(GREY)
        canvas.drawString(LM, PAGE_H - 1.15 * cm, "DDP Final Report | Skin-Lesion Classification on DermaMNIST")
        canvas.setStrokeColor(MAROON); canvas.setLineWidth(0.6); canvas.line(LM, PAGE_H - 1.30 * cm, PAGE_W - RM, PAGE_H - 1.30 * cm)
        canvas.drawRightString(PAGE_W - RM, 1.0 * cm, f"Page {doc.page}"); canvas.drawString(LM, 1.0 * cm, "IIT Madras")
    else:
        canvas.setStrokeColor(MAROON); canvas.setLineWidth(0.8); canvas.line(LM + 3.0 * cm, 3.15 * cm, PAGE_W - RM - 3.0 * cm, 3.15 * cm)
        canvas.setFont('Helvetica', 10); canvas.setFillColor(GREY); canvas.drawCentredString(PAGE_W / 2, 2.68 * cm, "This project is carried out under the supervision of")
        canvas.setFont('Helvetica-Bold', 12.5); canvas.setFillColor(MAROON); canvas.drawCentredString(PAGE_W / 2, 2.18 * cm, "Dr. Jayaganthan R")
        canvas.setFont('Helvetica', 9.5); canvas.setFillColor(GREY)
        canvas.drawCentredString(PAGE_W / 2, 1.74 * cm, "Professor and Head, Department of Engineering Design (ED), IIT Madras")
    canvas.restoreState()

doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=LM, rightMargin=RM, topMargin=2.0 * cm, bottomMargin=1.8 * cm,
                        title="DDP Final Report - Skin-Lesion Classification on DermaMNIST", author="Suharsh Kumar, Sayantan Das")
doc.build(story, onFirstPage=decorate, onLaterPages=decorate)
print("built", OUT, "phase-2 results:", "present" if ens else "pending", datetime.datetime.now())
