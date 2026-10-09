# Python to AI

Free lessons site (GitHub Pages) plus the Jupyter notebooks behind each topic.

## Layout

- `index.html`, `404.html`, `app/` — the website. `app/pages/*.html` are the lessons; `app/styles/episodes.js` is the single list of YouTube episodes.
- `0 …` to `8 …` — topic folders with notebooks. Numbered with no gaps; keep it that way when adding or removing topics.
- `design/` — brand and redesign explorations (brand sheet, theme options). **Keep it.** It isn't deployed or referenced by the site, but it's the design history. Don't delete or "clean up" this folder.
- `.resources/data/colours/` and `drug200.csv` — used by the vision demo and Naive Bayes notebook.
- `BRAND.md` (visual rules), `VIDEOS.md` (video plan), `PLAN.md`.

## Rules

- **Datasets:** don't commit new datasets. Fetch them in code from Kaggle with `kagglehub` (or from the original source, e.g. `tf.keras.utils.get_file`), so notebooks download what they need on first run. Ishani is moving her own datasets to Kaggle too. Small, hand-made files (a few KB) can stay in the repo.
- **Adding a lesson:** every page has its own copy of the nav, so add the new page to the nav in `index.html` and every `app/pages/*.html`, plus the prev/next buttons, `sitemap.xml`, `scripts/make_og_images.py` and `tests/test_site.py`.
- Tests and lint match CI: `flake8 scripts/ tests/` and `pytest` with `requirements-ci.txt`.
