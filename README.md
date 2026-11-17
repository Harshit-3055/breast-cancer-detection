# Breast Cancer Detection — Deep Learning & Classical ML Benchmark

End-to-end Google Colab notebooks for breast cancer detection across **three imaging
modalities (ultrasound, histopathology, mammography) plus tabular cytology data**,
using six different model families. Each notebook is a complete, self-contained
pipeline: dataset download → preprocessing → leakage-safe splitting → training →
evaluation → saving the model, metrics, predictions and plots to Google Drive.
Together they form the experimental section of a comparative study.

> The skin cancer notebooks in the sibling `skin_cancer_detection/` folder belong
> to a **separate** project and are not part of this study.

---

## Notebooks

| # | Notebook | Model / Technology | Dataset (modality) | Key ideas |
|---|----------|--------------------|--------------------|-----------|
| 1 | `1_resnet50_baseline_busi.ipynb` | **ResNet-50** (transfer learning) | BUSI (ultrasound) | Baseline CNN, BCE loss, threshold 0.5 |
| 2 | `2_efficientnet_focal_loss_busi.ipynb` | **EfficientNet-B0 + Focal Loss** | BUSI (ultrasound) | Class-imbalance handling (α, γ, pos_weight), decision-threshold analysis |
| 3 | `3_densenet121_idc_histopathology.ipynb` | **DenseNet-121** + AMP | IDC patches (histopathology) | **Patient-wise split**, balanced 3,000/class subsample, 96×96 input, 4 epochs (fast setting) |
| 4 | `4_vision_transformer_breakhis.ipynb` | **ViT-Base/16** (`timm`) | BreakHis 100× (histopathology) | Transformer fine-tuning (AdamW, low LR, cosine decay), patient-wise split |
| 5 | `5_swin_transformer_cbis_ddsm.ipynb` | **Swin-Tiny** (`timm`) + AMP | CBIS-DDSM (mammography) | Hierarchical vision transformer, mammograms cached at 224×224, official train/test split, 6 epochs (fast setting) |
| 6 | `6_classical_ml_wisconsin.ipynb` | **LogReg, SVM, Random Forest, Gradient Boosting, XGBoost** | WDBC (tabular cytology) | 5-fold CV benchmark, ROC comparison, feature importance, GridSearchCV tuning |

All image notebooks solve the same binary task — **malignant = 1 vs
benign/normal = 0** — so results are directly comparable across modalities and
architectures.

## Datasets

| Dataset | Modality | Size | Source |
|---------|----------|------|--------|
| **BUSI** — Breast Ultrasound Images | Ultrasound | ~780 images (benign / malignant / normal) | Kaggle: `aryashah2k/breast-ultrasound-images-dataset` |
| **IDC** — Invasive Ductal Carcinoma patches | Histopathology (H&E) | ~277,000 patches, 279 patients | Kaggle: `paultimothymooney/breast-histopathology-images` |
| **BreakHis** — Breast tumour histology | Histopathology | ~7,900 images, 4 magnifications (100× used) | Kaggle: `ambarish/breakhis` |
| **CBIS-DDSM** — Curated DDSM | Mammography (JPEG release) | ~3,000 full mammograms (mass + calcification cases) | Kaggle: `awsaf49/cbis-ddsm-breast-cancer-image-dataset` (~6 GB) |
| **WDBC** — Wisconsin Diagnostic Breast Cancer | Tabular (FNA cytology, 30 features) | 569 samples | Built into scikit-learn (`load_breast_cancer`) |

Notes on labels:
- **BUSI**: `malignant` → 1; `benign` and `normal` → 0. Ground-truth `*_mask.png`
  segmentation files are excluded.
- **IDC / BreakHis**: label parsed from the folder structure / filename.
- **CBIS-DDSM**: an image is 1 if any of its annotated abnormalities has
  `pathology == MALIGNANT`, otherwise 0 (`BENIGN`, `BENIGN_WITHOUT_CALLBACK`).
  Full-mammogram JPEGs are resolved through `csv/dicom_info.csv`
  (series-UID → file path).
- **WDBC**: scikit-learn encodes 0 = malignant, 1 = benign; the notebook **flips**
  it so malignant = 1, consistent with everything else.

## Methodology (common to all notebooks)

- **Reproducibility** — fixed seed (42) for `random`, NumPy and PyTorch; all
  hyperparameters exported to a `*_config.json` per experiment.
- **Splits** — stratified 70/15/15 train/val/test. For histopathology (IDC,
  BreakHis) the split is **patient-wise**: all patches/images of a patient stay in
  one split, preventing data leakage that would inflate test scores. CBIS-DDSM
  keeps the dataset's **official** train/test split (15% of train held out for
  validation).
- **Transfer learning** — all image models start from ImageNet weights; inputs
  resized to 224×224 (96×96 for the natively 50×50 IDC patches) and normalized
  with ImageNet statistics.
- **Compute budget** — every notebook fits in a single free-tier Colab T4
  session. The two largest datasets (IDC, CBIS-DDSM) use a *fast setting*: only
  the needed images are read from the downloaded zip, IDC is subsampled, CBIS-DDSM
  mammograms are downscaled once and cached, and training uses mixed precision
  for a few epochs. Their scores are therefore conservative relative to
  full-scale training.
- **Augmentation** — flips, small rotations, color jitter (plus vertical flips for
  histology, which has no canonical orientation).
- **Class imbalance** — `pos_weight` computed from the training distribution
  (ViT, Swin), Focal Loss with positive weighting (EfficientNet), balanced
  subsampling (IDC).
- **Model selection** — best checkpoint by validation accuracy; evaluation runs on
  the untouched test split only.
- **Metrics** — accuracy, per-class precision/recall/F1, confusion matrix and
  **ROC-AUC** (the primary metric under imbalance). All headline numbers use the
  default threshold of 0.5. Notebook 2 additionally sweeps the decision threshold
  to show the precision/recall trade-off (see *Results*).
- **Figures** — training curves, confusion matrix, ROC curve (and EDA/feature
  importance in notebook 6), saved at 150 dpi, colorblind-safe palette.

## How to run (Google Colab)

1. Open a notebook in Colab and select a GPU runtime
   (**Runtime → Change runtime type → T4 GPU**). Notebook 6 runs fine on CPU.
2. Get a Kaggle API token: kaggle.com → *Settings* → *Create New Token* — this
   downloads `kaggle.json`.
3. Run the cells top to bottom. When prompted, upload `kaggle.json` and authorize
   the Google Drive mount. (Notebooks 3 and 5 pick up `kaggle.json` automatically
   if you place it in the root of your Google Drive, so after the Drive popup they
   run unattended.)
4. Everything is saved under `MyDrive/breast_cancer_project/`:

```
breast_cancer_project/
├── data/        (scratch; datasets are downloaded to the Colab VM)
├── models/      best checkpoints (.pth) / fitted models + scaler (.joblib)
├── results/     *_results.txt, *_predictions.csv, *_config.json,
│                threshold analysis, model-comparison table
└── plots/       sample images, training curves, confusion matrix + ROC,
                 EDA, feature importance (.png, 150 dpi)
```

Approximate single-run times on a Colab T4: notebooks 1–2 ≈ 15–25 min,
notebook 3 ≈ 10–15 min (~3 GB download), notebook 4 ≈ 30–40 min,
notebook 5 ≈ 15–20 min (~5 GB download), notebook 6 ≈ 2 min.

## Requirements

Everything comes preinstalled on Colab except `timm` (installed in notebooks 4–5):
`torch`, `torchvision`, `timm`, `scikit-learn`, `xgboost`, `pandas`, `numpy`,
`matplotlib`, `seaborn`, `Pillow`, `joblib`, `kaggle`.

## Results

Held-out **test-set** results, Colab T4, seed 42, threshold 0.5. "Mal." is the
malignant (positive) class. Each notebook also writes a `*_predictions.csv` with
per-sample probabilities, so ROC/PR curves and significance tests can be
regenerated without retraining.

| Model | Modality / Dataset | Test n | Accuracy | Precision (mal.) | Recall (mal.) | F1 (mal.) | ROC-AUC |
|-------|--------------------|-------:|---------:|-----------------:|--------------:|----------:|--------:|
| ResNet-50 | Ultrasound / BUSI | 117 | 0.880 | 0.870 | 0.645 | 0.741 | 0.918 |
| EfficientNet-B0 + Focal Loss | Ultrasound / BUSI | 117 | 0.889 | 0.800 | 0.774 | 0.787 | **0.951** |
| DenseNet-121 | Histopathology / IDC | — | *pending* | | | | |
| ViT-Base/16 | Histopathology / BreakHis 100× | 353 | 0.790 | 0.952 | 0.758 | 0.844 | 0.938 |
| Swin-Tiny | Mammography / CBIS-DDSM | — | *pending* | | | | |
| Logistic Regression (best classical) | Tabular / WDBC | 114 | 0.965 | 0.975 | 0.929 | 0.951 | **0.996** |

### Classical ML on WDBC (notebook 6)

| Model | 5-fold CV ROC-AUC | Test accuracy | Test precision | Test recall | Test F1 | Test ROC-AUC |
|-------|------------------:|--------------:|---------------:|------------:|--------:|-------------:|
| Logistic Regression | 0.995 ± 0.006 | 0.965 | 0.975 | 0.929 | 0.951 | 0.996 |
| SVM (RBF) | 0.995 ± 0.005 | 0.974 | 0.976 | 0.952 | 0.964 | 0.995 |
| Gradient Boosting | 0.988 ± 0.013 | 0.965 | 1.000 | 0.905 | 0.950 | 0.995 |
| Random Forest | 0.990 ± 0.012 | 0.974 | 1.000 | 0.929 | 0.963 | 0.994 |
| XGBoost | 0.990 ± 0.012 | 0.965 | 1.000 | 0.905 | 0.950 | 0.993 |
| XGBoost (GridSearchCV-tuned) | 0.992 | 0.974 | — | — | — | 0.991 |

Tuned XGBoost parameters: `learning_rate=0.1, max_depth=3, n_estimators=400`.

### Decision-threshold trade-off (EfficientNet-B0 + Focal Loss, BUSI test set)

| Threshold | Precision (mal.) | Recall (mal.) | F1 (mal.) |
|----------:|-----------------:|--------------:|----------:|
| 0.2 | 0.592 | 0.935 | 0.725 |
| 0.3 | 0.744 | 0.935 | 0.829 |
| 0.4 | 0.774 | 0.774 | 0.774 |
| 0.5 | 0.800 | 0.774 | 0.787 |
| 0.6 | 0.815 | 0.710 | 0.759 |
| 0.7 | 0.909 | 0.645 | 0.755 |

Lowering the threshold to 0.3 raises malignant recall from 0.774 to 0.935 (29 of
31 cancers detected, accuracy 0.897) at a modest precision cost. This sweep is
computed on the test set, so it illustrates the trade-off; it is not an
independently tuned operating point.

### Observations

- **Focal loss helps on imbalanced ultrasound.** On identical BUSI splits,
  EfficientNet-B0 + Focal Loss beats the ResNet-50 baseline in ROC-AUC
  (0.951 vs 0.918) and malignant recall (0.774 vs 0.645), with the same accuracy.
- **Patient-wise evaluation is harder.** ViT reached 0.91 validation accuracy but
  0.79 on unseen BreakHis patients, and its training loss fell to ~0.001 while
  validation loss rose: the model overfits a small cohort (81 patients). ROC-AUC
  stays high (0.938), so the ranking is good; most errors are malignant images
  scored below 0.5 (64 false negatives vs 10 false positives).
- **Tabular cytology is close to solved.** All five classical models reach
  ROC-AUC ≥ 0.993 on WDBC, and simple, interpretable logistic regression is as
  good as gradient-boosted trees.

## Dataset citations

- Al-Dhabyani W., Gomaa M., Khaled H., Fahmy A. *Dataset of breast ultrasound
  images.* Data in Brief, 28:104863, 2020. (BUSI)
- Janowczyk A., Madabhushi A. *Deep learning for digital pathology image analysis.*
  J Pathol Inform, 2016; Cruz-Roa A. et al., SPIE Medical Imaging, 2014. (IDC)
- Spanhol F., Oliveira L.S., Petitjean C., Heutte L. *A Dataset for Breast Cancer
  Histopathological Image Classification.* IEEE TBME, 63(7):1455–1462, 2016. (BreakHis)
- Lee R.S. et al. *A curated mammography data set for use in computer-aided
  detection and diagnosis research.* Scientific Data, 4:170177, 2017. (CBIS-DDSM)
- Wolberg W., Street W.N., Mangasarian O. *Breast Cancer Wisconsin (Diagnostic).*
  UCI Machine Learning Repository, 1995. (WDBC)

## Disclaimer

These notebooks are for research and educational purposes only and are **not** a
medical device; nothing here should be used for clinical diagnosis.
