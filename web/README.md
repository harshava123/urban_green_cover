# Urban Green Cover — Web Demo

Next.js frontend for live greenery classification. Runs **MobileNetV2** inference in the browser via TensorFlow.js — no backend API required (ideal for Vercel).

## How it works

1. **MobileNetV2 backbone** — loaded from the official TF.js model CDN (~4 MB, cached after first visit)
2. **Classification head** — your trained dense layer weights in `public/model/head_weights.json` (~112 KB)

This matches your Keras pipeline: frozen ImageNet MobileNetV2 → 1280-d embedding → dense softmax (4 classes).

## Prerequisites

- Trained weights: `outputs/models/mobilenetv2.weights.h5`
- Node.js 18+

## Setup

### 1. Export classification head weights

From the project root (only needs Python + h5py):

```bash
pip install h5py
python scripts/export_head_weights.py
```

Creates `web/public/model/head_weights.json`.

### 2. Install and run locally

```bash
cd web
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Deploy to Vercel

### Option A — Vercel CLI

```bash
cd web
npx vercel
```

### Option B — GitHub + Vercel Dashboard

1. Push the repo to GitHub (include `web/public/model/head_weights.json`)
2. Go to [vercel.com/new](https://vercel.com/new)
3. Import your repository
4. Set **Root Directory** to `web`
5. Framework Preset: **Next.js**
6. Deploy

Live URL example: `https://urban-green-cover-demo.vercel.app`

## Notes

- First visit downloads the MobileNetV2 backbone from Google CDN; head weights are served from your deployment.
- Upload aerial scene images similar to UC Merced (forests, residential, buildings).
- For presentation: demo 2–3 images showing dense_green, minimal_green, and a borderline sparse_green case.

## Optional: full TF.js model export

If you prefer a single bundled model file instead of backbone + head, use `scripts/export_tfjs_model.py` (requires TensorFlow + tensorflowjs on Linux/macOS or a clean Python 3.10 venv on Windows).
