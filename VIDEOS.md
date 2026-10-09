# YouTube Video Plan

Channel: [youtube.com/@DrIshaniKathuria](https://www.youtube.com/@DrIshaniKathuria)

Two tracks per topic, linked to each other:

- **Plain-English track:** 10–12 min, no code on screen, for curious non-tech viewers. Goal: AI is less scary once you see what it actually is, and it's much more than chatbots.
- **Under-the-hood track:** walks through this repo's notebooks and pages for engineers.

Videos are filmed talking to camera, unscripted, with Claude Motion animations on screen. Each episode below lists **talking points to glance at, not a script**.

**Publishing an episode on the site:** open [`app/styles/episodes.js`](app/styles/episodes.js), find the episode, and paste the video's YouTube ID into `id` (the part after `watch?v=` in its URL). Its "coming soon" card becomes a player. Everything you upload also appears automatically in the "Latest from the channel" playlist on that page.

**Status key:** 💡 idea · 📝 outlined · 🎬 filmed · ✂️ editing · ✅ published

---

## Series 1: AI Is Not the Terminator (plain-English)

| # | Title (working) | Status | Site page | Video |
|---|---|---|---|---|
| 1 | AI is older than your grandparents' TV | 📝 | [History of AI](app/pages/history_of_ai.html) | — |
| 2 | You've used AI 50 times today | 💡 | [Recommenders](app/pages/recommendation_system.html) | — |
| 3 | How a computer learns: it's just practice | 💡 | [Supervised ML](app/pages/supervised_learning.html) | — |
| 4 | Sorting socks without instructions | 💡 | [Unsupervised ML](app/pages/unsupervised_learning.html) | — |
| 5 | How computers "see" | 💡 | [Computer Vision](app/pages/computer_vision.html) | — |
| 6 | What ChatGPT actually does | 💡 | [Generative AI](app/pages/generative_ai.html) | — |
| 7 | How to fool an AI | 💡 | `8 Adversarial Threats/` (page to come) | — |
| 8 | The real risks of AI (research episode) | 💡 | — | — |

### Ep 1: AI is older than your grandparents' TV

**Angle:** AI has been researched since the 1940s, and the hype has been wrong many times. The scary sci-fi version has been "20 years away" for 70 years. Cliffhanger: every era built a *different kind* of AI → episode 2.

**Format:** to camera, with an era filter per section. Use the timeline page's presentation mode as backup B-roll.

| Time | Era look | Beat |
|---|---|---|
| 0:00 | Modern | Hook: "AI isn't new. It's older than colour TV." |
| 0:45 | Greyscale + monocle | **1943**: McCulloch & Pitts, the brain as on/off switches |
| 2:00 | Greyscale | **1950**: Turing asks "Can machines think?" Ask viewers to comment: would you be able to tell? |
| 3:00 | Greyscale | **1956**: Dartmouth names "AI" and thinks one summer is enough |
| 4:00 | Sepia + newspaper | **1958**: Perceptron headlines (walk, talk, be conscious!) vs. what it actually did |
| 5:30 | Frosty blue | **AI winters**: the hype crashes twice (1970s, late 80s) |
| 6:30 | 90s VHS | **1997**: Deep Blue beats Kasparov; the world didn't end, chess got more popular |
| 7:30 | Clean modern | **2012–17**: AlexNet and Transformers; the 1943 idea finally has enough data and computing power |
| 8:45 | Modern | **2022**: ChatGPT, fancy autocomplete |
| 9:45 | Today | **2026**: real issues (bias, privacy, misuse, jobs) → tease research episode |
| 11:00 | Today | Cliffhanger + link to the site |

**Motion animation ideas**
- Vertical timeline 1943 → 2026 that ages from sepia to modern
- Recreated 1950s newspaper front page (label it as a recreation; don't show the real NYT clipping)
- "Hype vs. reality" rollercoaster chart, with the dips labelled "AI winter"

**Before publishing**
- [ ] Fact-check: Deep Blue's ~200M positions/sec; 1958 newspaper wording; 1960s "20 years" predictions
- [ ] Confirm the "including me!" research line on the 2026 card
- [ ] Description links: site, History page, repo

---

## Series 2: Python to AI (under-the-hood)

| # | Topic | Status | Site page | Repo folder |
|---|---|---|---|---|
| 0 | Python for people who've never coded | 💡 | [Python](app/pages/python.html) | `0 Basic_Python_Concepts/` |
| 1 | NumPy, Pandas and PyTorch basics | 💡 | [ML Basics](app/pages/ml_basics.html) | `1 Basic ML Concepts/` |
| 2 | Supervised learning | 💡 | [Supervised](app/pages/supervised_learning.html) | `2 Machine Learning/S*` |
| 3 | Unsupervised learning | 💡 | [Unsupervised](app/pages/unsupervised_learning.html) | `2 Machine Learning/US*` |
| 4 | Recommendation systems | 💡 | [Recommenders](app/pages/recommendation_system.html) | `3 Recommendation Systems/` |
| 5 | Neural networks from scratch | 💡 | [Deep Learning](app/pages/deep_learning.html) | `4 Deep Learning/` |
| 6 | CNNs and the colour classifier | 💡 | [CV](app/pages/computer_vision.html) | `5 Computer Vision/` |
| 7 | Time series | 💡 | [Time Series](app/pages/time_series.html) | — |
| 8 | Word2Vec to language models | 💡 | [NLP](app/pages/natural_language_processing.html) | `6 Natural Language Processing/` |
| 9 | Build a baby GPT, RAG | 💡 | [Generative AI](app/pages/generative_ai.html) | `7 Generative AI/` |
| 10 | Adversarial attacks | 💡 | — | `8 Adversarial Threats/` |

**Tip:** tag the commit used in each technical video (`git tag ep-05`) and link the tag in the description, so viewers see the same code even after the repo changes.
