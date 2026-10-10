# Python to AI

Free lessons site (GitHub Pages) plus the Jupyter notebooks behind each topic.

## Layout

- `index.html`, `404.html`, `app/` — the website. `app/pages/*.html` are the lessons; `app/styles/episodes.js` is the single list of YouTube episodes.
- `0 …` to `8 …` — topic folders with notebooks. Numbered with no gaps; keep it that way when adding or removing topics.
- `design/` — brand and redesign explorations (brand sheet, theme options). **Keep it.** It isn't deployed or referenced by the site, but it's the design history. Don't delete or "clean up" this folder.
- `.resources/data/colours/` — images used by the colour CNN demo and `scripts/export_colour_cnn.py`.
- `BRAND.md` (visual rules), `VIDEOS.md` (video plan), `PLAN.md`.

## Rules

- **Datasets:** don't commit new datasets. Fetch them in code from Kaggle with `kagglehub` (or from the original source, e.g. `tf.keras.utils.get_file`), so notebooks download what they need on first run. Ishani is moving her own datasets to Kaggle too. Pattern used in the notebooks:
  ```python
  import kagglehub
  path = kagglehub.dataset_download("owner/dataset")  # downloads once, then cached
  df = pd.read_csv(f"{path}/file.csv")
  ```
  Public datasets need no Kaggle login. Small, hand-made files (a few KB) can stay in the repo. Still local for now: Titanic `train.csv`/`test.csv` (the test set is only on the login-gated competition page) and `7 Generative AI/Baby_GPT/data/allrecipes_data.txt` (Ishani's own scrape, to be uploaded to Kaggle).
- **Adding a lesson:** add it to `scripts/lessons.py`, then run `python scripts/sync_nav.py` to rewrite the menus and previous/next links on every page (a test fails if they drift). Also add a homepage card in `index.html`, plus `sitemap.xml`, `scripts/make_og_images.py` (then run it) and the episode in `app/styles/episodes.js`. Lesson pages end with a quiz (`QUIZ_DATA`) and a completion key `p2ai_done_<name>` that the homepage card's `data-done-key` must match.
- **Lesson styles:** lesson pages use Tailwind classes compiled into `app/styles/lessons.css` (config in `tailwind.config.js`). After adding classes that aren't used elsewhere, run `npm run build:css` and commit the result; CI fails if it's stale. Don't bring back the Tailwind CDN script. Shared, non-Tailwind styles go in `app/styles/site.css`.
- Tests and lint match CI: `flake8 scripts/ tests/` and `pytest` with `requirements-ci.txt`.
- `requirements.txt` must stay installable on macOS, Linux and Windows: lower-bound pins only, no platform-specific packages (e.g. `pywin32`, `tensorflow_intel`, `+cu130` builds).
