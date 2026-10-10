"""The lesson list: the one place to add, remove or reorder lessons.

Each entry is (number, page file, menu label, short label for previous/next buttons).
After editing, run `python scripts/sync_nav.py` to update every page.
"""

LESSONS = [
    ("00", "python.html", "Python basics", "Python"),
    ("01", "ml_basics.html", "ML basics", "ML Basics"),
    ("02", "supervised_learning.html", "Supervised learning", "Supervised ML"),
    ("03", "unsupervised_learning.html", "Unsupervised learning", "Unsupervised"),
    ("04", "recommendation_system.html", "Recommendation systems", "Recommenders"),
    ("05", "deep_learning.html", "Deep learning", "Deep Learning"),
    ("06", "computer_vision.html", "Computer vision", "Computer Vision"),
    ("07", "time_series.html", "Time series", "Time Series"),
    ("08", "natural_language_processing.html", "Language (NLP)", "NLP"),
    ("09", "generative_ai.html", "Generative AI", "Generative AI"),
    ("10", "adversarial_ai.html", "Adversarial AI", "Adversarial AI"),
]
