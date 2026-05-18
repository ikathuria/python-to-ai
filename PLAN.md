# Python to AI — Self-Learning Platform

> A free, self-hostable, one-stop learning platform covering Python basics through Generative AI, with live in-browser ML model demos powered by ONNX Runtime Web.

---

## Viability Summary

| | |
|---|---|
| **Market** | Niche gap — platforms like Kaggle Learn, fast.ai, and DeepLearning.AI exist, but none combine structured tutorials + live in-browser model inference in a single free static site |
| **Feasibility** | Medium — core content and 8 ONNX models already exist; main work is restructuring to static, wiring onnxruntime-web, and polishing UX |
| **Free to build** | Yes — GitHub Pages (free), ONNX Runtime Web (free/open-source, runs in browser), no server required |
| **Monetization** | Portfolio project |

---

## Tech Stack

| Layer | Choice | Reason |
|---|---|---|
| Frontend | Vanilla HTML/CSS/JS + Tailwind CDN | Matches existing codebase, no build step, works with GitHub Pages |
| Model Inference | onnxruntime-web (CDN) | Runs the 8 existing ONNX models entirely in the browser — zero server cost |
| Syntax Highlighting | Prism.js (CDN) | Lightweight, CDN-available, supports Python |
| Hosting | GitHub Pages | 100% free, already a GitHub repo, supports custom domains |
| Backend | None (eliminated) | All inference moves client-side; Flask is no longer needed |
| Auth | None | Portfolio/public learning platform, no login required |

---

## Environment Variables

None required — fully static, no secrets or API keys needed.

---

## Current State (as of planning)

The repo already has:
- ✅ 9 topic modules (0–8): Basic Python → Generative AI
- ✅ 26 Jupyter notebooks with implementations
- ✅ 8 ONNX models in `app/onnx_models/`
- ✅ Flask app with 10 HTML topic pages in `app/pages/`
- ✅ Landing `index.html`

The plan converts the Flask app to a pure static site deployable via GitHub Pages, adds browser-based ONNX inference, and polishes the UX across all pages.

---

## Milestones

---

### Milestone 1: Static Site Restructure
**Goal:** Repo runs as a pure static site locally (via `python -m http.server` or Live Server), with all existing pages accessible and GitHub Pages deployment configured.

Tasks:
- [ ] Create a clean static folder layout: move all HTML pages from `app/pages/` into `pages/`, move ONNX models from `app/onnx_models/` into `models/`, move any CSS/JS assets into `assets/css/` and `assets/js/` — Done when: all files are in their new locations and no broken relative paths remain
- [ ] Update `index.html` navigation links to point to the new `pages/` paths — Done when: every topic link on the landing page opens the correct page
- [ ] Add a `404.html` page that redirects to `index.html` (required for GitHub Pages SPA routing) — Done when: file exists with a meta-refresh or JS redirect
- [ ] Create `.github/workflows/deploy.yml` using the `actions/deploy-pages` action to auto-deploy `main` branch to GitHub Pages — Done when: workflow file is valid YAML and would trigger on push to main
- [ ] Delete Flask-specific files (`app/main.py`, `wsgi.py`, `requirements.txt` Flask entries) or move them to a `_archive/` folder so they don't confuse the static structure — Done when: root directory has no Flask entry points

---

### Milestone 2: ONNX Browser Inference
**Goal:** All 8 existing ONNX models are runnable directly in the browser — users can input values and see predictions without any server.

Tasks:
- [ ] Create `assets/js/onnx-inference.js` — a shared helper module that (a) loads an ONNX model file from a given path using `onnxruntime-web` via CDN, (b) runs inference given a Float32Array input, and (c) returns the output tensor as a plain JS array — Done when: the module exports `loadModel(path)` and `runInference(session, inputData, inputShape)` functions
- [ ] Add ONNX Runtime Web via CDN (`<script src="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js">`) to all topic pages that have a demo — Done when: `ort` global is available on those pages
- [ ] Wire up the **Classification demo** (KNN or Logistic Regression model) on `pages/supervised_learning.html`: add an input form (feature sliders or number inputs for Iris dataset features), run inference on submit, display the predicted class — Done when: entering valid Iris feature values returns a class label (Setosa/Versicolor/Virginica)
- [ ] Wire up the **Linear Regression demo** on `pages/supervised_learning.html` or a dedicated regression section: input one or two feature values, display predicted output — Done when: changing the input changes the predicted value
- [ ] Wire up the **K-Means clustering demo** on `pages/unsupervised_learning.html`: accept 2D point coordinates, run inference, display which cluster the point belongs to — Done when: different input coordinates return different cluster labels
- [ ] Wire up remaining ONNX models (Naive Bayes, Decision Tree, Autoencoder, any others in `models/`) to their respective topic pages with matching input forms — Done when: each model has at least one working input/output demo on its page
- [ ] Add a loading spinner and error message for each demo (model file not found, inference error) — Done when: a fake bad model path shows an error message instead of a crash

---

### Milestone 3: UI/UX Overhaul
**Goal:** All pages share a consistent, clean design with working navigation, a learning path flow, and mobile-friendly layout.

Tasks:
- [ ] Create `assets/css/main.css` with a shared design system: CSS variables for colors/spacing, base typography, utility classes — Done when: all pages import this file and render consistently
- [ ] Add Tailwind CDN to all pages as the primary styling utility (alongside `main.css` for custom overrides) — Done when: Tailwind classes work on all pages without a build step
- [ ] Build a shared `<nav>` component (copy-paste HTML) with links to all 9 topic pages and a "Home" link — include it at the top of every page — Done when: navigation is present and functional on all 10 pages
- [ ] Build a shared `<footer>` with GitHub repo link and "built with ONNX Runtime Web" credit — Done when: footer appears on all pages
- [ ] Add a learning path progress bar or topic index on `index.html` showing the 9 modules in order (0 → 8) with visual links — Done when: landing page clearly communicates the learning sequence
- [ ] Make all pages mobile-responsive: navigation collapses to a hamburger menu on small screens, demo inputs stack vertically — Done when: pages render without horizontal scroll on a 375px wide viewport
- [ ] Add topic difficulty badges (Beginner / Intermediate / Advanced) to each page's header — Done when: every topic page shows a badge
- [ ] Add syntax highlighting via Prism.js CDN to all code blocks across all pages — Done when: Python code blocks render with color highlighting

---

### Milestone 4: Content Completion
**Goal:** All 9 modules have complete, substantive page content — no thin or empty pages.

Tasks:
- [ ] Audit all 10 HTML pages and list which sections are missing or placeholder text — Done when: a comment block at the top of each page lists what was added/confirmed complete
- [ ] Complete `pages/time_series.html` (Module 6 — currently thin): add theory section (what is a time series, stationarity, AR/MA/ARIMA concepts), a code example, and a simple demo — Done when: page has at minimum a theory section, one code snippet, and one visual or interactive element
- [ ] Add "Key Concepts" summary boxes to each topic page (3–5 bullet points of the most important takeaways from that module) — Done when: every page has a Key Concepts section before the demo
- [ ] Add links from each topic page to the relevant Jupyter notebooks in the repo (GitHub raw links or nbviewer links) — Done when: every page has at least one "See the notebook" link
- [ ] Add an "What you'll learn" intro section at the top of each topic page — Done when: every page opens with a brief learning objective paragraph

---

### Milestone 5: Interactive Learning Features
**Goal:** Users can engage actively with code and test their understanding beyond just reading.

Tasks:
- [ ] Add copy-to-clipboard buttons on all `<pre><code>` blocks — Done when: clicking the button copies code and shows a brief "Copied!" confirmation
- [ ] Add a "Try it yourself" expandable section on each topic page with a minimal reproducible Python snippet users can copy and run — Done when: every page has at least one such section
- [ ] Add simple knowledge-check questions (3 multiple-choice questions per topic) as collapsible `<details>` elements with the answer hidden — Done when: all 9 topic pages have at least 3 questions with hidden answers
- [ ] Add smooth scroll-to-demo anchor links in the page header so users can jump directly to the interactive demo — Done when: clicking "Try the Demo →" scrolls smoothly to the demo section

---

### Milestone 6: Deploy & Polish
**Goal:** Site is live at `https://[username].github.io/python-to-ai` with no broken links or console errors.

Tasks:
- [ ] Enable GitHub Pages in the repo settings (source: GitHub Actions) and push to trigger the first deploy — Done when: site loads at the GitHub Pages URL
- [ ] Run a link checker (e.g., `htmlproofer` or manual check) across all pages and fix any broken internal links — Done when: no 404s from internal navigation
- [ ] Add `<meta>` SEO tags (title, description, og:title, og:description) to every page — Done when: all pages have at minimum title and description tags
- [ ] Add a `robots.txt` and `sitemap.xml` to the repo root — Done when: both files exist with correct content
- [ ] Lazy-load ONNX model files (only fetch the `.onnx` file when the user clicks "Run Demo", not on page load) — Done when: network tab shows model file loads only after button click
- [ ] Update `README.md`: add live site URL badge, project screenshot, one-paragraph description, and "Learning Path" section listing all 9 modules — Done when: README renders well on GitHub and includes the live URL

---

## Claude Code Commands

**Start from Milestone 1:**
```
claude "Read PLAN.md and complete Milestone 1. Mark tasks done as you go. Stop after Milestone 1 and commit."
```

**Resume from any point:**
```
claude "Read PLAN.md, find the first incomplete task, and continue. Mark tasks done as you go. Commit when a milestone is complete."
```

**Test current state:**
```
claude "Read PLAN.md. Without building anything new, test everything that's marked done. Report what works and what's broken."
```

---

## Notes & Decisions

- **Why static over Flask?** Flask requires a server (cold starts, uptime limits on free tiers). GitHub Pages is always-on, zero-cost, and the content is educational — no server-side logic is needed.
- **Why ONNX Runtime Web over Hugging Face Spaces / Gradio?** Gradio/Streamlit demos on HF Spaces go to sleep on the free tier. In-browser inference via `onnxruntime-web` is instant, private (data never leaves the browser), and costs nothing.
- **8 ONNX models already exist** in `app/onnx_models/` — Milestone 2 is about wiring them up, not converting models.
- **Tailwind via CDN** avoids a build step entirely, which keeps the project accessible to contributors without Node.js installed.
- **No auth** — this is a public learning resource, no login required.
