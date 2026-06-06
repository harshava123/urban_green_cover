# Phase 2 — Teams Submission Guide

## Submission Title (copy exactly, replace name)

```
Phase 2 - Proposal and Code Implementation - Your Name
```

---

## What to Submit on Teams

| # | Material | What you submit |
|---|----------|-----------------|
| 1 | **Proposal document** | Export `docs/PROPOSAL.md` as PDF or Word |
| 2 | **GitHub repository link** | Public repo URL (see setup below) |
| 3 | **Kaggle Notebook link** | Optional — link if you upload `notebooks/` |
| 4 | **README** | Included in GitHub repo (`README.md`) |

---

## GitHub Repository Setup

1. Create a new public repository on GitHub (e.g. `urban-green-cover-cnn`)
2. Push this project:

```bash
cd urban_green_cover
git init
git add .
git commit -m "Phase 2: Urban Green Cover CNN project"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/urban-green-cover-cnn.git
git push -u origin main
```

3. **Do not upload** large dataset files — only include the dataset link in README
4. Add `data/raw/` to `.gitignore` (large images)

---

## Repository Must Include

- [x] `src/` — all Python source code
- [x] `run_pipeline.py` — main runner
- [x] `config.yaml` — configuration
- [x] `requirements.txt` — dependencies
- [x] `README.md` — setup and run instructions
- [x] `docs/PROPOSAL.md` — full proposal
- [x] `outputs/figures/` — generated figures (optional but recommended)
- [x] `outputs/reports/` — JSON evaluation reports
- [x] Dataset link (not the full 2,100 images)

---

## Kaggle Notebook (Optional)

1. Go to [kaggle.com/code](https://www.kaggle.com/code) → New Notebook
2. Add dataset: `abdulhasibuddin/uc-merced-land-use-dataset`
3. Upload project files or clone from GitHub
4. Set notebook settings: GPU optional (CPU is fine for this dataset)
5. Copy public notebook URL into Teams submission

---

## Proposal → PDF

**Option A:** Open `docs/PROPOSAL.md` in VS Code → export with Markdown PDF extension

**Option B:** Paste into Google Docs / Word → format → export PDF

**Before submitting:** Replace `[Your Name]` in the proposal with your full name.

---

## Key Results to Mention in Presentation

- **Best model:** MobileNetV2 — 93.3% accuracy, 11.9 ms/image
- **Dataset:** 2,099 aerial images, 4 greenery classes
- **Methods:** Custom CNN + 3 transfer learning models
- **Extras:** Grad-CAM, ROC curves, error analysis
