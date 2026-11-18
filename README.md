# Skin-Lesion Classification on DermaMNIST

Dual Degree Project (DDP) — Indian Institute of Technology Madras.
Seven-class dermatoscopic lesion classification on the **DermaMNIST** benchmark
(MedMNIST v2 / MedMNIST+, derived from HAM10000), official split, **64x64 source images**.

**Final report (Phase 1 + Phase 2, November 2026):** [`report/DDP_Final_Report_DermaMNIST.pdf`](report/DDP_Final_Report_DermaMNIST.pdf)

## Headline results (official test split, 2,005 images)

| System | Top-1 | Top-3 | Balanced acc. | Macro-AUC |
|---|---|---|---|---|
| Phase 1 (July 2026): EfficientNet-B0 @96 px + TTA | 80.00% | 97.36% | 54.87% | 0.949 |
| **Phase 2: PRIME ensemble, point A (accuracy)** | 82.29% | 97.91% | 60.17% | 0.962 |
| **Phase 2: PRIME ensemble, point B (screening, tau = 0.15)** | **82.64%** | 97.91% | **66.20%** | 0.962 |
| Phase 2: PRIME ensemble, point C (balanced, tau = 0.45) | 79.95% | 97.91% | **71.63%** | 0.962 |
| Best published 28/64-px result (VGG16 @64, MedMNIST+) | 79.53 ± 1.19% | — | — | 0.944 |

PRIME-B vs Phase 1: 95% CI of the top-1 gain [+1.3, +4.0] points, exact McNemar p = 9e-5 (116 vs 63 discordant test images),
13.2% relative error reduction, balanced accuracy +11.3 points, pooled malignant-class recall 45.7% → 53.1%.

**PRIME** = **P**rior-corrected, **R**esolution-matched, heterogeneous Efficient-Net ensemble with TTA:
five EfficientNet / EfficientNetV2 members (96–128 px inputs, one with Shades-of-Gray colour constancy),
validation-tuned probability averaging, 8-view test-time augmentation, and a post-hoc logit adjustment
`argmax_c [log p_c(x) − tau · log prior_c]` whose single exponent tau selects the operating point.
All members were trained on a Colab T4 in about 30 minutes (`phase2/code/prime_colab.py`).

## Repository

```
report/DDP_Final_Report_DermaMNIST.pdf   final report (36 pp): both phases, critical review, PRIME, statistics, 6-month plan, evidence appendices
report/DDP_Report_DermaMNIST.pdf         Phase-1 summer report (July 2026; contains the errors corrected in the final report)
phase2/code/prime_colab.py               Phase-2 training + ensemble + figures, single script for Colab GPU (commented)
phase2/code/train_v2.py, ensemble_v2.py  the same pipeline split for the 8 GB laptop (resume-safe, per-stage processes)
phase2/code/comparative_stats.py         bootstrap CIs, McNemar tests, per-class deltas, leaderboard
phase2/code/prior_correction_np.py       zero-training logit adjustment of the archived Phase-1 probabilities
phase2/code/build_report.py              builds the final report PDF from the result files (reportlab)
phase2/results/                          every results JSON (members, ensemble, statistics) + Colab run summary
phase2/figures/                          confusion matrices, tau trade-off, per-class recall, member curves, Colab screenshots
phase2/docs/                             literature/benchmark notes with sources, project progress log
phase1/code, phase1/logs                 Phase-1 training scripts and the complete July logs
notebooks/                               original Phase-1 Colab notebooks
```

## Reproduce Phase 2

Open a Colab GPU runtime and run `phase2/code/prime_colab.py` (it downloads `dermamnist_64.npz` from the
MedMNIST+ Zenodo record and verifies its MD5). The script writes per-member probabilities, `ensemble_v2_results.json`
and all figures to `/content/DDP_Phase2`, then zips them. `phase2/code/import_colab_results.sh <zip>` unpacks the archive
and rebuilds the report; `comparative_stats.py` recomputes the statistics from the saved probabilities.

## Next term (6-month plan, Section 13 of the report)

Native 128/224-px MedMNIST+ data and full-resolution HAM10000 (target ≥ 85% top-1 at 224 px), leakage-free evaluation on
DermaMNIST-C/-E, a preprocessing pipeline (hair removal, colour constancy, lesion cropping), calibration + conformal
prediction sets, skin-tone-stratified fairness evaluation, an on-device prototype, and a workshop paper.

---

## Phase 1 (summer 2026) summary

Dual Degree Project (DDP), Summer 2026 — Indian Institute of Technology Madras.
Seven-class dermatoscopic lesion classification on the **DermaMNIST** benchmark
(derived from HAM10000), built to beat the published MedMNIST v2 baseline.

## Result

| Metric | This work | MedMNIST v2 benchmark |
|---|---|---|
| Top-1 accuracy | **80.0%** | 76.8% |
| Top-3 accuracy | **97.4%** | — |
| Macro-averaged AUC | **0.949** | 0.92 |
| Top-1 *without* TTA | 78.4% | — |
| **Balanced accuracy** | **54.9%** | — |

Test-time augmentation is worth **+1.6 pp** of top-1 on its own (78.40 → 80.00).

## The number that matters most is the last one

Top-1 of 80.0% and balanced accuracy of **54.9%** describe the same model. The gap
is the whole story of this dataset: DermaMNIST is severely imbalanced, so a model
can score well on overall accuracy while performing poorly on the rare classes —
which, in dermatology, are often the ones that matter clinically.

Both figures are reported here deliberately. Quoting only the 80.0% would overstate
what this model does.

## Method

1. **Backbone** — ImageNet-pretrained **EfficientNet-B0**.
2. **Head** — custom **multi-scale** classification head over the backbone features.
3. **Schedule** — **two-phase transfer learning**: frozen-backbone warm-up, then
   discriminative fine-tuning of the full network.
4. **Imbalance** — class-imbalance-aware training rather than uniform sampling.
5. **Inference** — **test-time augmentation**, ablated against the plain forward pass.

## Data

**DermaMNIST** (MedMNIST v2) — 28×28 dermatoscopic images across seven diagnostic
categories, derived from the HAM10000 collection. Standard train/val/test split;
no images were moved between splits.

## Repository

```
notebooks/01_model_training.ipynb    training pipeline end to end
notebooks/02_results_analysis.ipynb  evaluation, TTA ablation, figures
report/DDP_Report_DermaMNIST.pdf     full report: methods, benchmark comparison,
                                     clinical implications and limitations
results/results.json                 headline metrics
```

## Limitations

- **Balanced accuracy is 54.9%.** Rare classes are not reliably identified.
- **28×28 inputs.** DermaMNIST is heavily downsampled; fine dermatoscopic structure
  (pigment networks, vascular patterns) that a dermatologist uses is simply absent
  at this resolution. Results here do not transfer directly to full-resolution images.
- **Benchmark dataset, not clinical validation.** Nothing here has been evaluated
  prospectively, on external cohorts, or against clinician performance.

## Authors

Suharsh Kumar (BE22B037), Sayantan Das (BS22B035).
Mentor: G. K. Harish Balaji (BE21B017).
Supervisor: Dr. Jayaganthan R, Professor and Head, Department of Engineering Design, IIT Madras.
