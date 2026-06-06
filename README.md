# Urban Green Cover Image Assessment

CNN-based image analysis for urban greenery classification from scene images.

A teaching-oriented Machine Learning project that compares a custom CNN with transfer learning models, evaluates them rigorously, explains predictions with Grad-CAM, and frames deployment as **decision-support for human experts**.

## Research Questions

| ID | Question |
|----|----------|
| **RQ1** | How accurately can CNN-based models perform urban greenery classification from scene images using publicly available image datasets? |
| **RQ2** | Which model family — custom CNN or transfer learning — provides the best balance between predictive performance and computational efficiency? |
| **RQ3** | How do preprocessing, augmentation, and class-imbalance handling influence model robustness and class-wise performance? |
| **RQ4** | Do Grad-CAM explanations show that trained models focus on meaningful visual regions related to urban greenery? |
| **RQ5** | What are the main failure patterns, deployment limitations, and practical risks in real environmental monitoring settings? |

## Use Cases

1. Urban planning dashboards
2. Neighborhood greenery assessment for smart cities

## Dataset

**Primary dataset:** [UC Merced Land Use Dataset (Kaggle)](https://www.kaggle.com/datasets/abdulhasibuddin/uc-merced-land-use-dataset)

| Property | Value |
|----------|-------|
| Images | 2,100 aerial RGB patches (2,099 used after cleaning) |
| Original classes | 21 land-use categories |
| Project classes | 4 greenery levels (mapped from UC Merced) |
| Image size | 256×256 (resized to 224×224 for training) |

### Download and setup

```bash
# Option A: Kaggle CLI
pip install kaggle
kaggle datasets download -d abdulhasibuddin/uc-merced-land-use-dataset
# Extract to data/raw/converted_uc_merced_data/

# Option B: Download manually from Kaggle and extract to:
# data/raw/converted_uc_merced_data/{class_name}/*.jpg
```

### Greenery taxonomy (mapped from 21 UC Merced classes)

| Class | UC Merced source examples |
|-------|---------------------------|
| `dense_green` | forest, agricultural, golfcourse |
| `moderate_green` | chaparral, river |
| `sparse_green` | mediumresidential, sparseresidential, denseresidential |
| `minimal_green` | buildings, freeway, parkinglot, runway, etc. |

### Class distribution

- minimal_green: 1,299 | dense_green: 300 | sparse_green: 300 | moderate_green: 200

## Project Structure

```
urban_green_cover/
├── config.yaml              # Hyperparameters and paths
├── run_pipeline.py          # End-to-end runner
├── requirements.txt
├── data/
│   ├── raw/                 # Place CSU-RSISC10 patches here
│   └── processed/           # Auto-generated train/val/test splits
├── src/
│   ├── data_preparation.py  # Cleaning, dedup, label mapping
│   ├── data_loader.py       # Augmentation and TF datasets
│   ├── models.py            # Custom CNN + transfer learning
│   ├── train.py
│   ├── evaluate.py          # Metrics, confusion matrix, ROC-AUC
│   ├── grad_cam.py          # Explainability (RQ4)
│   └── error_analysis.py    # Failure patterns (RQ5)
├── scripts/
│   └── create_demo_dataset.py
└── outputs/
    ├── models/
    ├── figures/
    ├── reports/
    └── logs/
```

## Setup

```bash
cd urban_green_cover
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Run the Pipeline

```bash
# Full pipeline: prepare → train → evaluate → Grad-CAM → error analysis
python run_pipeline.py --step all

# Individual steps
python run_pipeline.py --step prepare
python run_pipeline.py --step train
python run_pipeline.py --step evaluate
python run_pipeline.py --step explain
python run_pipeline.py --step errors

# Train/evaluate one model
python run_pipeline.py --step train --model mobilenetv2

# Generate report figures (training curves, ROC, comparison table)
python scripts/generate_report_figures.py
```

## Phase 2 Submission Checklist

| Item | Location |
|------|----------|
| Proposal document | `docs/PROPOSAL.md` |
| Source code | `src/`, `run_pipeline.py` |
| Dataset link | See Dataset section above |
| README | This file |
| Output figures | `outputs/figures/` |
| Evaluation reports | `outputs/reports/` |

**Author:** Penthala Harsha Vardhan

**GitHub:** https://github.com/harshava123/urban_green_cover

**Teams submission title:** `Phase 2 - Proposal and Code Implementation - Penthala Harsha Vardhan`

## Models Compared

| Model | Type | Purpose |
|-------|------|---------|
| `custom_cnn` | Built from scratch | Baseline for teaching CNN fundamentals |
| `mobilenetv2` | Transfer learning | Lightweight, mobile-friendly |
| `resnet50` | Transfer learning | Strong accuracy baseline |
| `efficientnetb0` | Transfer learning | Efficiency vs. accuracy trade-off |

## Evaluation Metrics

- Accuracy, precision, recall, F1 (macro and weighted)
- Confusion matrix
- ROC-AUC (one-vs-rest)
- Inference time per image
- Training time and parameter count

Results are saved to `outputs/reports/model_comparison.json`.

## Explainability

Grad-CAM heatmaps are generated for:

- Correct predictions
- False positives
- False negatives

Saved under `outputs/figures/grad_cam/<model_name>/`.

## Methodology Alignment

| Step | Implementation |
|------|----------------|
| Dataset selection | UC Merced Land Use (Kaggle) with academic permissions |
| Cleaning | Corrupt/duplicate removal, class balance summary |
| Preprocessing | Resize to 224×224, normalize, domain augmentation |
| Models | 1 custom CNN + 3 transfer learning models |
| Evaluation | Full metric suite + inference timing |
| Explainability | Grad-CAM on correct/incorrect cases |
| Deployment framing | Error analysis with screening-role guidance |

## Deployment Discussion (RQ5)

This system is designed as a **screening framework**, not an autonomous decision maker:

- **Role**: Flag neighborhoods or image batches for expert review in urban planning dashboards.
- **Limitations**: Aerial patch resolution, geographic bias, seasonal variation, adjacent-class confusion.
- **Ethical concerns**: Automated greenery scores should not replace community input in planning decisions.
- **Validation**: Requires prospective testing on locally captured images with expert adjudication.

## Publication Checklist

- [ ] Report benchmark table across all four models
- [x] Include ablation: with/without augmentation and class weights (RQ3)
- [x] Show Grad-CAM panels for each error type (RQ4)
- [ ] Summarize top confusion pairs from error analysis (RQ5)
- [ ] Discuss compute budget: params vs. accuracy vs. inference ms

## Citation

If using UC Merced, cite Yang & Newsam (2010). See the [official dataset page](http://weegee.vision.ucmerced.edu/datasets/landuse.html).

## License

Code: MIT. Dataset: follow UC Merced / Kaggle terms of use.
