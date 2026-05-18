# Python to AI — Interactive Learning Platform

[![Live Site](https://img.shields.io/badge/Live%20Site-GitHub%20Pages-blue?logo=github)](https://ikathuria.github.io/python-to-ai/)
[![Deploy](https://github.com/ikathuria/python-to-ai/actions/workflows/static.yml/badge.svg)](https://github.com/ikathuria/python-to-ai/actions/workflows/static.yml)

A free, self-hostable tutorial platform taking you from Python basics to advanced AI — with **live in-browser ML model demos** powered by [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/). No server required; everything runs in your browser.

**Live:** [ikathuria.github.io/python-to-ai](https://ikathuria.github.io/python-to-ai/)

---

## Learning Path

| # | Module | Topics | Demo |
|---|--------|---------|------|
| 0 | [Python Basics](app/pages/python.html) | Data types, structures, functions, OOP | — |
| 1 | [ML Basics](app/pages/ml_basics.html) | NumPy, Pandas, data preprocessing | — |
| 2 | [Supervised Learning](app/pages/supervised_learning.html) | Classification, regression, KNN, Decision Trees | Live prediction demo |
| 3 | [Unsupervised Learning](app/pages/unsupervised_learning.html) | K-Means clustering, PCA | Cluster assignment demo |
| 4 | [Recommendation Systems](app/pages/recommendation_system.html) | Content-based & collaborative filtering | — |
| 5 | [Deep Learning](app/pages/deep_learning.html) | PyTorch, backprop, IrisNet | — |
| 6 | [Computer Vision](app/pages/computer_vision.html) | CNNs, convolution, image classification | 🎨 Colour classifier (ONNX) |
| 7 | [Time Series](app/pages/time_series.html) | Stationarity, ARIMA, LSTM forecasting | — |
| 8 | [NLP](app/pages/natural_language_processing.html) | Word2Vec, PMI, language models | — |

---

## Features

- **Live ONNX demos** — 9 trained models run entirely in the browser (no server, no Python needed)
- **Interactive quizzes** — 3 knowledge-check questions per topic with instant feedback
- **Copy buttons** — one-click copy on every code block
- **"Try it yourself"** — collapsible hints for hands-on exercises
- **Mobile responsive** — hamburger nav, stacked layouts on small screens

---

## Tech Stack

| Layer | Choice |
|-------|--------|
| Frontend | Vanilla HTML/CSS/JS + Tailwind CDN |
| Model inference | [onnxruntime-web](https://cdn.jsdelivr.net/npm/onnxruntime-web) (runs in browser) |
| Syntax highlighting | Highlight.js |
| Hosting | GitHub Pages (free, always-on) |

---

## Running Locally

```bash
git clone https://github.com/ikathuria/python-to-ai.git
cd python-to-ai
python -m http.server 8000
# open http://localhost:8000
```

No build step, no Node.js, no dependencies needed to view the site.

---

## Re-training / Exporting Models

The ONNX models in `app/onnx_models/` were exported from the Jupyter notebooks in the repo root. To retrain the colour CNN:

```bash
pip install torch torchvision onnx onnxsim pillow numpy
python scripts/export_colour_cnn.py
```

---

## For more tutorials

- Articles on [Medium](https://medium.com/@ishani-kathuria)
- GitHub [Wiki](https://github.com/ikathuria/python-to-ai/wiki)
