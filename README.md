# Python to AI — Interactive Learning Platform

[![Live Site](https://img.shields.io/badge/Live%20Site-GitHub%20Pages-blue?logo=github)](https://ikathuria.github.io/python-to-ai/)
[![Deploy](https://github.com/ikathuria/python-to-ai/actions/workflows/static.yml/badge.svg)](https://github.com/ikathuria/python-to-ai/actions/workflows/static.yml)

A free, self-hostable tutorial platform taking you from Python basics to generative and adversarial AI — with **live in-browser ML model demos** powered by [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/). No server required; everything runs in your browser.

**Live:** [ikathuria.github.io/python-to-ai](https://ikathuria.github.io/python-to-ai/)

---

## Learning Path

| # | Lesson | Topics | Try it in the browser |
|---|--------|--------|------|
| — | [History of AI](app/pages/history_of_ai.html) | 80 years of AI, hype vs. reality (no code) | Animated timeline |
| 00 | [Python basics](app/pages/python.html) | Data types, structures, functions, OOP | — |
| 01 | [ML basics](app/pages/ml_basics.html) | NumPy, Pandas, data preprocessing | — |
| 02 | [Supervised learning](app/pages/supervised_learning.html) | Classification, regression, KNN, decision trees | Live predictions (ONNX) |
| 03 | [Unsupervised learning](app/pages/unsupervised_learning.html) | K-means clustering, PCA | Cluster assignment (ONNX) |
| 04 | [Recommendation systems](app/pages/recommendation_system.html) | Content-based & collaborative filtering | Movie recommender (ONNX) |
| 05 | [Deep learning](app/pages/deep_learning.html) | PyTorch, backprop, neural networks | Gradient descent & activation playgrounds |
| 06 | [Computer vision](app/pages/computer_vision.html) | CNNs, convolution, image classification | Colour classifier (ONNX) |
| 07 | [Time series](app/pages/time_series.html) | Stationarity, ARIMA, LSTM forecasting | Moving-average smoother |
| 08 | [Language (NLP)](app/pages/natural_language_processing.html) | Word2Vec, PMI, language models | Cosine similarity explorer |
| 09 | [Generative AI](app/pages/generative_ai.html) | Next-token prediction, Baby GPT, RAG, diffusion | Be the language model |
| 10 | [Adversarial AI](app/pages/adversarial_ai.html) | Adversarial examples, FGSM, prompt injection, defences | Fool a classifier |

---

## Repo Map

| Path | What's in it |
|------|--------------|
| `index.html`, `app/pages/` | The website (static HTML, served by GitHub Pages) |
| `app/onnx_models/` | Trained models the site runs in the browser |
| `app/styles/` | Brand tokens, shared styles and scripts, built lesson CSS, `episodes.js` (video list) |
| `0 Basic_Python_Concepts/` … `8 Adversarial Threats/` | Notebooks behind each topic. Each folder has a README linking to its lesson |
| `scripts/` | Export models to ONNX, keep the nav in sync, make share images |
| `tests/` | Site checks that run in CI |
| [`CLAUDE.md`](CLAUDE.md) | Conventions for working on the repo (datasets, adding lessons) |
| [`PLAN.md`](PLAN.md) | Roadmap |
| [`VIDEOS.md`](VIDEOS.md) | YouTube episode plan linked to each lesson |
| [`BRAND.md`](BRAND.md) | Brand guidelines |
| `design/` | Brand exploration: the 8 directions and the brand sheet |

Datasets aren't stored in the repo: notebooks download them from Kaggle on first run with [`kagglehub`](https://github.com/Kaggle/kagglehub) (public datasets, no account needed).

---

## Features

- **Live demos**: 15 ONNX models plus small JavaScript demos run entirely in the browser (no server, no Python needed)
- **Learning path**: lessons grouped into four stages, with a progress bar and "Continue" button saved in your browser
- **Knowledge checks**: a quiz at the end of every lesson; a perfect score marks the lesson complete
- **Copy buttons** on every code block and **"Try it yourself"** exercises
- **Light and dark mode**, mobile friendly

---

## Tech Stack

| Layer | Choice |
|-------|--------|
| Frontend | Static HTML/CSS/JS; lesson styles compiled with Tailwind CSS |
| Model inference | [onnxruntime-web](https://onnxruntime.ai/docs/tutorials/web/) (runs in the browser) |
| Syntax highlighting | Highlight.js |
| Hosting | GitHub Pages |

---

## Running Locally

**View the site** (no build step, no Node.js):

```bash
git clone https://github.com/ikathuria/python-to-ai.git
cd python-to-ai
python -m http.server 8000
# open http://localhost:8000
```

**Run the notebooks** (macOS, Linux or Windows):

```bash
pip install -r requirements.txt
jupyter notebook
```

**Working on the site:**

```bash
python scripts/sync_nav.py   # after adding/reordering lessons in scripts/lessons.py
npm install && npm run build:css   # after changing Tailwind classes in app/pages/*.html
pip install -r requirements-ci.txt && pytest   # the same checks CI runs
```

---

## Re-training / Exporting Models

The ONNX models in `app/onnx_models/` were exported from the notebooks. To re-export them:

```bash
python scripts/export_tutorial_onnx.py     # scikit-learn models
python scripts/export_recommender_onnx.py  # movie recommender
python scripts/export_colour_cnn.py        # colour CNN (needs torch)
```

---

## For more tutorials

- Videos on [YouTube](https://www.youtube.com/@DrIshaniKathuria)
- My website: [ishani.kathuria.net](https://ishani.kathuria.net)
- Connect on [LinkedIn](https://linkedin.com/in/ishani-kathuria)
- Articles on [Medium](https://medium.com/@ishani-kathuria)
- GitHub [Wiki](https://github.com/ikathuria/python-to-ai/wiki)
