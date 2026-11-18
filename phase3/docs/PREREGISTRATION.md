# Phase 3 pre-registration (DermaMNIST, native 224 px)

Written 2026-11-18 14:25 IST, during Colab session 1, by the project team. Committed to the project repository with this timestamp.

## What had already been observed when this was written
- Session 1 trains runs in sequence and prints each run's test metrics when it finishes. At the time of writing, ONE test result had been
  printed and seen: ConvNeXt-T seed 0 (92.27% top-1 with 8-view TTA). No other test metric of any Phase-3 model, ensemble or operating point
  had been computed. Nothing in the rules below was changed after seeing that number; the rules are the ones encoded in `prime3_colab.py`
  as launched at 13:46 (recipe v1), which is itself byte-recoverable from `prime3_oneliner_blob.txt`.
- Phase-2 (64-px) results and the Phase-1 result were fully known.

## Primary endpoint
Top-1 accuracy on the official DermaMNIST test split (2,005 images) read from the MedMNIST+ file `dermamnist_224.npz`
(Zenodo 10519652), computed from saved per-image softmax probabilities, for the PRIMARY ensemble defined below at operating point A.

## Secondary endpoints
Macro one-vs-rest AUC (MedMNIST definition), balanced accuracy (mean per-class recall), macro-F1, top-3; the same metrics at operating
points B and C; melanoma one-vs-rest AUROC and sensitivity at 80/90/95% specificity with thresholds fixed on the validation split; expected
calibration error (15 bins), NLL and Brier score; all metrics on the lesion-clean test subset (1,232 images whose lesion never appears in
the training or validation split) and on the leaky remainder (773); all metrics on the DermaMNIST-E test split (1,511 ISIC-2018 test images).

## Models (recipe v1, frozen)
Five members, seed 0, trained once each with identical recipe apart from learning rate, layer-decay factor, batch size and epochs
(exact values in `prime3_colab.py` CONFIGS): ConvNeXt-T (IN-22k), DINOv2 ViT-B/14, DINOv2 ViT-S/14, EfficientNetV2-S (IN-21k),
DenseNet-121 (torchvision IN-1k). AdamW, 2-epoch warm-up + cosine to 1e-6, weight decay as configured, gradient clipping 1.0,
layer-wise LR decay, stochastic depth, GPU augmentation (random resized crop 60-100%, flips, any-angle rotation, small translation,
colour jitter, blur p=0.1, random erasing p=0.2), mixup 0.2 / CutMix 1.0 on 50% of batches with label smoothing 0.1, fp16 autocast,
EMA of weights (decay 0.999 with warm-up). If a run goes out of GPU memory it is retried with half the batch size and the learning rate
divided by sqrt(2); this happened to DINOv2 ViT-B/14 (batch 32 -> 16 before the run was repeated on a fresh kernel) and is disclosed.

## Selection rules (fixed before any test metric of the ensemble was computed)
1. Per run, the model is the FINAL EMA weights after the last epoch. No epoch is picked by validation accuracy; the per-epoch validation
   curve of raw and EMA weights is logged only for the figure and the "best-epoch gap" diagnostic.
2. Inference uses the 8 dihedral views (4 rotations x flip), softmax averaged. The clean single view is also saved.
3. PRIMARY ensemble: equal-weight arithmetic mean of the five seed-0 members' TTA probabilities. Secondary rows (reported, never
   headline): greedy forward-selected subset on validation top-1; equal-weight mean over all runs including extra seeds.
4. Decision rule: argmax_c [log p_c(x) - tau * log pi_c] with pi the training-split class prior. tau_A = 0. tau_B = the largest tau in
   {0, 0.05, ..., 1.5} whose validation top-1 is within 0.5 points of the best validation top-1. tau_C = the tau maximising validation
   balanced accuracy. Both chosen on the validation split only, then applied once to the test split.
5. Frozen-feature guardrail: a DINOv2 ViT-B/14 linear probe (logistic regression, C chosen on validation) and 11-NN; fine-tuned DINOv2
   members are expected to exceed the probe on validation; if not, that is reported, not fixed by re-tuning on test.

## Statistics
95% percentile bootstrap intervals over test images (2,000 resamples) and over lesions (cluster bootstrap) for top-1 and balanced accuracy;
exact McNemar tests on discordant pairs and paired bootstrap intervals of the accuracy difference for every comparison against the Phase-2
ensemble (PRIME-B) and against the best single Phase-3 member; mean +- sample standard deviation over seeds where seeds exist (session 2).

## Session 2 (planned, same recipe v1 for every reference arm)
Seeds 1 and 2 for ConvNeXt-T and DINOv2 ViT-B/14; ablations on seed 0: DINOv2 ViT-S/14 with a uniform learning rate (no layer decay),
ConvNeXt-T without mixup/CutMix, DenseNet-121 with a plain recipe (no augmentation beyond resize, no mixup, no label smoothing, no EMA,
no layer decay, no weight decay), ConvNeXt-T on the 128-px MedMNIST+ file; zero-shot evaluation of every run on DermaMNIST-E with the
saved final weights and the tau values selected on the official validation split.

## What will NOT be claimed
State of the art relative to unaudited preprints; clinical usefulness; robustness beyond what the ablations measure.
