# Python to AI: plan and roadmap

> A free, static learning site that goes from Python basics to generative AI, with models running live in the browser (ONNX Runtime Web), backed by the notebooks in this repo.

The original build plan (Flask → static site, ONNX demos, UI overhaul, deploy) is finished. This file now tracks what's left. Finished work lives in git history.

## Where things stand

- **Site:** static HTML on GitHub Pages, deployed by `.github/workflows/static.yml`. 11 lessons (00–10) plus History of AI and Videos, all on one brand ([BRAND.md](BRAND.md)), with light and dark mode and mobile layouts.
- **Demos:** 15 ONNX models in `app/onnx_models/`, loaded only when a demo is used. Smaller demos (next-word prediction, fooling a classifier, cosine similarity) are plain JavaScript.
- **Learning flow:** lessons grouped into four stages on the homepage, a progress bar and "Continue" button (stored in the visitor's browser), a quiz at the end of every lesson, and previous/next links in lesson order.
- **Notebooks:** one numbered folder per topic (`0 …` to `8 …`). Datasets download from Kaggle with `kagglehub`.
- **Videos:** two YouTube series planned in [VIDEOS.md](VIDEOS.md). Episodes appear on the site when their ID is added to `app/styles/episodes.js`.

## Next up

- [ ] Publish the first episodes and add their IDs to `app/styles/episodes.js`
- [ ] Upload the Baby GPT recipes dataset to Kaggle and switch `baby_gpt.ipynb` to `kagglehub`
- [ ] "What you'll learn" box at the top of each lesson and a "Key takeaways" box at the end
- [ ] "See the notebook" link on every lesson (4 of 11 have one)
- [ ] Difficulty badge (beginner / intermediate / advanced) on each lesson
- [ ] Time series lesson: add a forecasting demo (it has a moving-average smoother today)
- [ ] Expand the adversarial notebooks (FGSM on a real model, prompt injection examples) to match lesson 10
- [ ] Research episode and page: the real risks of AI

## Decisions

- **Static, no server.** GitHub Pages is free and always on. Inference runs in the browser, so visitors' data never leaves their device.
- **ONNX Runtime Web over hosted demos.** Free-tier Gradio/Streamlit apps go to sleep; in-browser models start instantly.
- **Tailwind is compiled, not loaded from the CDN.** The Play CDN builds styles in the browser and isn't meant for production. `npm run build:css` regenerates `app/styles/lessons.css`; the built file is committed, so viewing or editing content needs no Node.js.
- **One lesson list, generated nav.** `scripts/sync_nav.py` writes the menu, mobile menu and previous/next links on every page from a single list. A test fails if a page drifts.
- **No accounts.** Progress is saved in the visitor's browser only.
