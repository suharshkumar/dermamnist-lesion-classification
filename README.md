# Skin-Lesion Classification on DermaMNIST

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
