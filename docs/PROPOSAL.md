# Phase 2 — Project Proposal

**Submission title:** Phase 2 - Proposal and Code Implementation - [Your Name]

---

## 1. Project Title

**Urban Green Cover Image Assessment: CNN-Based Greenery Classification from Aerial Scene Images**

**Idea code:** Idea 15 of 75 (CNN-Based Image Dataset Project Idea Book)

---

## 2. Background and Motivation

Urban green cover is a key indicator of environmental quality, livability, and climate resilience in cities. Planners and environmental agencies need scalable methods to screen large volumes of scene images and identify areas with high, moderate, sparse, or minimal visible greenery.

Manual inspection does not scale to city-wide monitoring. Convolutional Neural Networks (CNNs) can automate image screening, but real-world deployment requires careful benchmarking, explainability, and awareness of limitations. This project applies supervised deep learning to a public aerial image dataset and frames the system as **decision-support for human experts**, not autonomous policy control.

**Applied use cases:**
1. Urban planning dashboards
2. Neighborhood greenery assessment for smart cities

---

## 3. Problem Statement

Given an aerial scene image, classify the level of urban green cover into one of four greenery categories. The system must:

- Train and compare a custom CNN with transfer learning models
- Handle class imbalance in the dataset
- Report standard classification metrics and inference time
- Provide explainability and error analysis for research-style reporting

---

## 4. Selected Dataset and Link

| Item | Detail |
|------|--------|
| **Dataset** | UC Merced Land Use Dataset (Kaggle) |
| **Link** | https://www.kaggle.com/datasets/abdulhasibuddin/uc-merced-land-use-dataset |
| **Original source** | http://weegee.vision.ucmerced.edu/datasets/landuse.html |
| **License** | Public domain imagery (USGS); follow Kaggle/dataset terms |

---

## 5. Dataset Description

| Property | Value |
|----------|-------|
| **Image type** | Aerial RGB remote sensing patches |
| **Original resolution** | 256 × 256 pixels |
| **Original classes** | 21 land-use categories |
| **Total images used** | 2,099 (after cleaning; 1 duplicate removed) |
| **Project classes** | 4 greenery levels (mapped from 21 UC Merced classes) |
| **Train / Val / Test** | 1,471 / 314 / 314 (70% / 15% / 15%) |

### Greenery taxonomy (project classes)

| Class | Description | Example UC Merced source classes |
|-------|-------------|--------------------------------|
| `dense_green` | High vegetation cover | forest, agricultural, golfcourse |
| `moderate_green` | Moderate vegetation | chaparral, river |
| `sparse_green` | Low urban greenery | mediumresidential, sparseresidential, denseresidential |
| `minimal_green` | Built-up / low green cover | buildings, freeway, parkinglot, runway, etc. |

### Class distribution

| Class | Count | Share |
|-------|------:|------:|
| minimal_green | 1,299 | 61.9% |
| dense_green | 300 | 14.3% |
| sparse_green | 300 | 14.3% |
| moderate_green | 200 | 9.5% |

The dataset is **imbalanced**, so class-weighted training is applied.

---

## 6. Research Questions and Answers

### RQ1: Classification accuracy on public datasets
**Answer:** MobileNetV2 achieved **93.3%** test accuracy (F1 macro 0.91, ROC-AUC 0.996) on UC Merced mapped to 4 greenery classes. ResNet50: 93.0%. Custom CNN: 85.0%. Public aerial imagery supports reliable green-cover screening.

### RQ2: Performance vs. efficiency
**Answer:** Transfer learning outperforms custom CNN. **MobileNetV2** is the best trade-off (93.3% accuracy, ~12 ms/image). ResNet50 matches accuracy but is ~4× slower (~47 ms/image).

### RQ3: Preprocessing, augmentation, class imbalance
**Answer:** Ablation on MobileNetV2 (see `outputs/reports/rq3_ablation.json`):

| Configuration | Aug | Weights | Test Acc. | F1 macro |
|---------------|-----|---------|-----------|----------|
| Full pipeline | Yes | Yes | 93.3% | 0.91 |
| No augmentation | No | Yes | 96.2% | 0.94 |
| No class weights | Yes | No | 95.2% | 0.93 |
| Bare (neither) | No | No | 97.5% | 0.96 |

Preprocessing is essential. Augmentation and class weights act as regularizers; on this fixed split, removing them raised test scores, but they remain recommended for real-world robustness (rotation, lighting) and minority-class recall.

### RQ4: Grad-CAM meaningful regions
**Answer:** Yes. Grad-CAM/saliency maps show activation on vegetated regions for green classes and built-up areas for minimal_green. Errors occur when sparse trees/lawns near buildings trigger false sparse_green predictions. See `outputs/reports/rq4_gradcam_analysis.json` and `outputs/figures/grad_cam/`.

### RQ5: Failure patterns and deployment risks
**Answer:** Main confusion: minimal_green ↔ sparse_green (6.7% error rate on MobileNetV2). Risks: adjacent class similarity, geographic bias, high-confidence errors. Deploy as **decision-support for human experts**, not autonomous policy. See `outputs/reports/mobilenetv2_error_analysis.json`.

---

## 7. Expected Results

Based on methodology and preliminary experiments:

| Model | Expected test accuracy | Expected role |
|-------|------------------------|---------------|
| Custom CNN | 80–88% | Teaching baseline; shows CNN fundamentals |
| MobileNetV2 | 90–94% | Best accuracy–speed trade-off |
| ResNet50 | 90–94% | High accuracy; slower inference |
| EfficientNetB0 | Variable | Additional transfer-learning comparison |

**Observed results (test set):**

| Model | Accuracy | F1 (macro) | ROC-AUC | Inference (ms) |
|-------|----------|------------|---------|----------------|
| MobileNetV2 | 93.3% | 0.91 | 0.996 | 11.9 |
| ResNet50 | 93.0% | 0.90 | 0.996 | 47.1 |
| Custom CNN | 85.0% | 0.80 | 0.976 | 11.0 |
| EfficientNetB0 | 65.0% | 0.20 | 0.644 | 19.2 |

---

## 8. Proposed Methodology

```
Raw UC Merced images
    → Cleaning (corrupt/duplicate removal)
    → Label mapping (21 → 4 greenery classes)
    → Train/val/test split (70/15/15)
    → Resize to 224×224, normalize, augment (train only)
    → Train: Custom CNN + MobileNetV2 + ResNet50 + EfficientNetB0
    → Evaluate: accuracy, precision, recall, F1, ROC-AUC, confusion matrix
    → Explain: Grad-CAM / saliency maps
    → Error analysis for deployment discussion
```

**Framework:** TensorFlow / Keras  
**Environment:** Python 3.x (local or Kaggle Notebook)

---

## 9. CNN Model Design

### Custom CNN (baseline)

- 4 convolutional blocks: 32 → 64 → 128 → 256 filters
- Batch normalization + max pooling after each block
- Global average pooling + dropout (0.4)
- Softmax output (4 classes)
- ~391K parameters

### Transfer learning models

| Model | Backbone | Pretrained weights | Trainable head |
|-------|----------|-------------------|----------------|
| MobileNetV2 | ImageNet | Frozen | Dense + dropout |
| ResNet50 | ImageNet | Frozen | Dense + dropout |
| EfficientNetB0 | ImageNet | Frozen | Dense + dropout |

---

## 10. Evaluation Metrics

- Accuracy
- Precision, recall, F1-score (macro and weighted)
- Confusion matrix
- ROC-AUC (one-vs-rest, macro average)
- Training/validation loss and accuracy curves
- Inference time per image
- Model parameter count and training time

---

## 11. Expected Figures and Tables

| # | Figure / Table | File location |
|---|----------------|---------------|
| 1 | Class distribution bar chart | `outputs/figures/class_distribution.png` |
| 2 | Training loss/accuracy curves (per model) | `outputs/figures/*_training_curves.png` |
| 3 | Confusion matrices (4 models) | `outputs/figures/*_confusion_matrix.png` |
| 4 | ROC curves (4 models) | `outputs/figures/*_roc_curves.png` |
| 5 | Model comparison table | `outputs/reports/model_comparison.json` |
| 6 | Grad-CAM / saliency panels | `outputs/figures/grad_cam/` |
| 7 | Classification reports | `outputs/reports/*_evaluation.json` |
| 8 | Error analysis summary | `outputs/reports/*_error_analysis.json` |

---

## 12. Implementation Summary

All required Phase 2 code components are implemented:

| Requirement | File |
|-------------|------|
| Dataset loading | `src/data_preparation.py`, `src/data_loader.py` |
| Preprocessing & split | `src/data_preparation.py` |
| Augmentation | `src/data_loader.py` |
| Custom CNN | `src/models.py` |
| Transfer learning | `src/models.py` |
| Training | `src/train.py` |
| Evaluation | `src/evaluate.py` |
| Grad-CAM / saliency | `src/grad_cam.py` |
| Error analysis | `src/error_analysis.py` |
| End-to-end runner | `run_pipeline.py` |
| Report figures | `scripts/generate_report_figures.py` |

---

## 13. Reproducibility

```bash
pip install -r requirements.txt
# Place dataset in data/raw/converted_uc_merced_data/
python run_pipeline.py --step prepare
python run_pipeline.py --step train
python run_pipeline.py --step evaluate
python run_pipeline.py --step explain
python scripts/generate_report_figures.py
```

---

## 14. References

1. Yang, Y., & Newsam, S. (2010). Bag-of-visual-words and spatial extensions for land-use classification. *ACM GIS*.
2. Helber, P., et al. (2019). EuroSAT: A novel dataset and deep learning benchmark for land use and land cover classification.
3. CSU-RSISC10 / UC Merced Land Use dataset documentation.
4. Selvaraju, R. R., et al. (2017). Grad-CAM: Visual explanations from deep networks.

---

*Replace `[Your Name]` with your full name before submitting on Teams.*
