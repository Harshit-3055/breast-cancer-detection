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
| 3 | `3_densenet121_idc_histopathology.ipynb` | **DenseNet-121** | IDC patches (histopathology) | ~277k 50×50 patches, **patient-wise split**, balanced subsampling |
| 4 | `4_vision_transformer_breakhis.ipynb` | **ViT-Base/16** (`timm`) | BreakHis 100× (histopathology) | Transformer fine-tuning (AdamW, low LR, cosine decay), patient-wise split |
| 5 | `5_swin_transformer_cbis_ddsm.ipynb` | **Swin-Tiny** (`timm`) + AMP | CBIS-DDSM (mammography) | Hierarchical vision transformer, mixed-precision training, official train/test split |
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
- **CBIS-DDSM**: `pathology == MALIGNANT` → 1; `BENIGN` and
  `BENIGN_WITHOUT_CALLBACK` → 0. Full-mammogram JPEGs are resolved through
  `csv/dicom_info.csv` (series-UID → file path).
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
  resized to 224×224 and normalized with ImageNet statistics.
- **Augmentation** — flips, small rotations, color jitter (plus vertical flips for
  histology, which has no canonical orientation).
- **Class imbalance** — `pos_weight` computed from the training distribution
  (ViT, Swin), Focal Loss with positive weighting (EfficientNet), balanced
  subsampling (IDC).
- **Model selection** — best checkpoint by validation accuracy; evaluation runs on
  the untouched test split only.
- **Metrics** — accuracy, per-class precision/recall/F1, confusion matrix and
  **ROC-AUC** (the primary metric under imbalance). Notebook 2 additionally sweeps
  the decision threshold and selects a recall-first operating point (0.4), since
  missing a malignancy is costlier than a false alarm in screening.
- **Figures** — training curves, confusion matrix, ROC curve (and EDA/feature
  importance in notebook 6), saved at 150 dpi, colorblind-safe palette.

## How to run (Google Colab)

1. Open a notebook in Colab and select a GPU runtime
   (**Runtime → Change runtime type → T4 GPU**). Notebook 6 runs fine on CPU.
2. Get a Kaggle API token: kaggle.com → *Settings* → *Create New Token* — this
   downloads `kaggle.json`.
3. Run the cells top to bottom. When prompted, upload `kaggle.json` and authorize
   the Google Drive mount.
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
notebook 3 ≈ 1–1.5 h (download is ~4 GB), notebook 4 ≈ 30–40 min,
notebook 5 ≈ 1–2 h (download is ~6 GB), notebook 6 ≈ 2 min.

## Requirements

Everything comes preinstalled on Colab except `timm` (installed in notebooks 4–5):
`torch`, `torchvision`, `timm`, `scikit-learn`, `xgboost`, `pandas`, `numpy`,
`matplotlib`, `seaborn`, `Pillow`, `joblib`, `kaggle`.

## Reporting results in the paper

Each notebook writes a `results/*_results.txt` with the final metrics and a
`*_predictions.csv` with per-sample probabilities (so ROC/PR curves and
significance tests can be regenerated without retraining). A suggested main table:

| Model | Modality / Dataset | Accuracy | Precision (mal.) | Recall (mal.) | F1 (mal.) | ROC-AUC |
|-------|--------------------|----------|------------------|---------------|-----------|---------|
| ResNet-50 | Ultrasound / BUSI | | | | | |
| EfficientNet-B0 + Focal | Ultrasound / BUSI | | | | | |
| DenseNet-121 | Histopathology / IDC | | | | | |
| ViT-Base/16 | Histopathology / BreakHis | | | | | |
| Swin-Tiny | Mammography / CBIS-DDSM | | | | | |
| Best classical (XGBoost) | Tabular / WDBC | | | | | |

Fill it from the saved `*_results.txt` files after running each notebook.

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
