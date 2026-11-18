# Literature / benchmark notes for the DermaMNIST final report (verified 2026-11-18)

All numbers copied as printed in the sources. Flags at the end.

## 1. MedMNIST v2 benchmark (Yang et al., Sci Data 10, 41, 2023; arXiv 2110.14795)
DermaMNIST, official split 7,007 / 1,003 / 2,005; source 3x600x450 -> 28x28; "224" rows are 28px upsampled.

| Method | AUC | ACC |
|---|---|---|
| ResNet-18 (28) | 0.917 | 0.735 |
| ResNet-18 (224) | 0.920 | 0.754 |
| ResNet-50 (28) | 0.913 | 0.735 |
| ResNet-50 (224) | 0.912 | 0.731 |
| auto-sklearn | 0.902 | 0.719 |
| AutoKeras | 0.915 | 0.749 |
| Google AutoML Vision | 0.914 | 0.768 |

(The July report printed auto-sklearn as 0.749 — wrong; correct is 0.719.)
Training protocol: CE, batch 128, Adam 1e-3, 100 epochs, LR x0.1 at 50/75, early stopping on val.

## 2. MedMNIST+ (Doerrich, Di Salvo, Brockmann, Ledig; Sci Rep 15, 7669, 2025; arXiv 2404.15786)
10 backbones x {end-to-end, linear probe, kNN} x {28, 64, 128, 224}; mean±std over 3 seeds; ACC/AUC only.
Zenodo 10.5281/zenodo.10519652: dermamnist.npz 19.7 MB; dermamnist_64.npz 100.1 MB (MD5 b70a2f5635c6199aeaa28c31d7202e1f);
dermamnist_128.npz 372.6 MB; dermamnist_224.npz 1.1 GB. DermaMNIST licence CC BY-NC 4.0.

End-to-end ACC % at 28 / 64 / 128 / 224:
- VGG16 76.58±0.26 / 79.53±1.19 / 81.80±1.29 / 81.55±1.89
- AlexNet 76.13±0.35 / 78.69±0.18 / 80.42±0.75 / 82.04±1.19
- ResNet-18 73.38±0.61 / 76.36±0.22 / 79.19±0.51 / 82.33±0.36
- DenseNet-121 74.16±0.65 / 76.16±0.58 / 81.76±0.25 / 84.74±0.51
- EfficientNet-B4 68.74±0.75 / 71.45±0.10 / 73.83±0.30 / 76.38±1.29
- ViT-B/16 74.40±1.51 / 77.01±1.45 / 80.81±0.89 / 82.31±1.36
- CLIP ViT-B/16 72.97±0.15 / 72.44±0.19 / 74.73±0.42 / 75.31±0.28
- EVA-02 ViT-B/16 73.48±0.65 / 75.28±0.98 / 76.41±0.46 / 77.94±0.29
- DINO ViT-B/16 74.40±0.80 / 76.87±0.12 / 79.22±1.56 / 81.31±1.05
- SAM ViT-B/16 73.08±0.65 / 74.68±0.10 / 76.71±1.06 / 77.42±0.17

End-to-end AUC % at 28 / 64 / 128 / 224:
- VGG16 92.69 / 94.36 / 95.71 / 95.51; AlexNet 92.23 / 94.34 / 95.13 / 96.10; ResNet-18 88.35 / 91.70 / 93.46 / 95.27;
  DenseNet-121 91.11 / 93.05 / 95.71 / 96.26; EfficientNet-B4 83.87 / 87.58 / 89.32 / 91.88; ViT-B/16 90.62 / 93.77 / 95.89 / 96.28;
  DINO ViT-B/16 91.60 / 93.57 / 95.34 / 96.50.

Linear probe ACC % (28/64/128/224): ViT-B/16 72.15/77.64/80.90/82.01; DINO ViT-B/16 75.78/79.88/81.65/84.42; CLIP 74.15/77.22/80.28/81.93;
EfficientNet-B4 69.99/72.77/72.97/73.23; ResNet-18 67.83/68.94/70.57/71.19; SAM collapses to majority class 66.88.
kNN (k=11) best: DINO 73.97/75.91/76.51/78.35.

**Key for our report (64x64 column)**: best MedMNIST+ end-to-end result at 64px is VGG16 79.53±1.19; EfficientNet-B4 at 64px is only 71.45.
Our July EfficientNet-B0 (64px data upsampled to 96) at 80.00% is above every 64px row. Best overall: DenseNet-121 @224 84.74.

## 3. Abhishek, Jain, Hamarneh, "Investigating the Quality of DermaMNIST and Fitzpatrick17k…", Sci Data 12, 196 (2025); arXiv 2401.14497
- HAM10000: 10,015 images of 7,470 unique lesions; 1,956 lesions (~26.18%) have 2+ images.
- Leakage across original DermaMNIST partitions (images / lesions): train–test 886 / 641; train–valid 440 / 332; valid–test 128 / 113; all three 51 / 40.
- Headline: **1,006 of 7,470 unique lesions (~13.47%) appear in more than one partition.** Plus 18 unreported duplicate pairs (7 cross-partition).
- DermaMNIST-C: lesion-disjoint, split 8,215 / 573 / 1,227; resized from originals. DermaMNIST-E: train = all 10,015; val/test = ISIC 2018 val/test (193 / 1,511).
- Their Table 1: original 28 R-18 0.917/0.735, R-50 0.913/0.735; original 224 R-18 0.920/0.754; -C 28 R-18 0.946±0.003 / 0.825±0.006; -C 224 R-18 0.953±0.001 / 0.847±0.010;
  -E 28 R-18 0.892±0.006 / 0.676±0.009; -E 224 R-18 0.896±0.010 / 0.685±0.000.
- Quotes: "we strongly suspect that the models' performance could be inflated"; "performance benchmarks for DermaMNIST and DermaMNIST-C should not be compared";
  -E "is guaranteed to be void of any data leakage". Do NOT claim "accuracy falls by X after de-leaking" — -C goes up (easier, smaller test), -E goes down.
- Data: https://zenodo.org/records/12739457 ; project page https://derm.cs.sfu.ca/critique/

## 4. Other DermaMNIST results > 0.80 (cite cautiously)
- MedNet (Ferdous et al., Sci Rep 15, 41936, 2025): 224px ACC 0.810±0.0066 (CBAM) / 0.820 (no CBAM); 384px 0.840±0.0087. Split/upsampling not stated.
- MedViT (Manzari et al., CBM 2023): MedViT-S @224 AUC 0.937 ACC 0.780 (below 0.80).
- Amangeldi et al. arXiv 2505.08259 (low-quality preprint): ResNet18@224 80.26, ViT-S/16 81.56.
- Wu & Papież arXiv 2501.14685 (under review; unusually high, unverified): DINOv2 ViT-B/14 @224 92.02±0.32; ResNet-18 @64->224 84.80±1.09.
- Kadric et al. arXiv 2507.12961: DermaMNIST-C 224 best 0.8490; original 28px EfficientNetV2L 0.7017.
- No source reports balanced accuracy for DermaMNIST.

## 5. Standard references
- Tan & Le, EfficientNet, ICML 2019, arXiv 1905.11946. — Tan & Le, EfficientNetV2, ICML 2021, arXiv 2104.00298.
- Lin et al., Focal loss, ICCV 2017, arXiv 1708.02002.
- Szegedy et al., Rethinking Inception (label smoothing), CVPR 2016, arXiv 1512.00567. — Müller, Kornblith, Hinton, When does label smoothing help?, NeurIPS 2019, arXiv 1906.02629.
- Menon et al., Long-tail learning via logit adjustment, ICLR 2021, arXiv 2007.07314.
- Shanmugam et al., Better aggregation in test-time augmentation, ICCV 2021, arXiv 2011.11156. — Krizhevsky et al., NeurIPS 2012 (multi-crop TTA).
- Zhang et al., mixup, ICLR 2018, arXiv 1710.09412.
- Tschandl, Rosendahl, Kittler, HAM10000, Sci Data 5, 180161 (2018). — Codella et al., arXiv 1902.03368.
- Esteva et al., Dermatologist-level classification of skin cancer, Nature 542, 115–118 (2017).
- Yang et al., MedMNIST v2, Sci Data 10, 41 (2023). — Doerrich et al., MedMNIST+, Sci Rep 15, 7669 (2025). — Abhishek et al., Sci Data 12, 196 (2025).

## Flags
- Nature pages were cookie-gated; numbers taken from arXiv PDFs / medmnist.com / Europe PMC and cross-checked.
- Abhishek et al. per-lesion-count sentence has a typo in the source.
- Wu & Papież and Amangeldi preprints: unverified / inconsistent; cite only with caveats or omit.
