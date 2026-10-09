"""Apply the shared brand chrome (head assets, menu bar, footer, a11y fixes) to site pages.

Idempotent: pages already carrying the menu bar are skipped. See BRAND.md.
"""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
PAGES = ROOT / "app" / "pages"

LESSONS = [
    ("00", "python.html", "Python basics"),
    ("01", "ml_basics.html", "ML basics"),
    ("02", "supervised_learning.html", "Supervised learning"),
    ("03", "unsupervised_learning.html", "Unsupervised learning"),
    ("04", "recommendation_system.html", "Recommendation systems"),
    ("05", "deep_learning.html", "Deep learning"),
    ("06", "computer_vision.html", "Computer vision"),
    ("07", "time_series.html", "Time series"),
    ("08", "natural_language_processing.html", "Language (NLP)"),
]
YOUTUBE = "https://www.youtube.com/@DrIshaniKathuria"
GITHUB = "https://github.com/ikathuria/python-to-ai"


def head_assets(prefix):
    return (
        f'\t<link rel="icon" href="{prefix}favicon.svg" type="image/svg+xml">\n'
        f'\t<link rel="stylesheet" href="{prefix}app/styles/brand.css">\n'
        f'\t<link rel="stylesheet" href="{prefix}app/styles/site.css">\n'
        f'\t<script src="{prefix}app/styles/site.js"></script>\n'
    )


def menubar(prefix, pages_prefix, current):
    def cur(f):
        return ' aria-current="page"' if f == current else ""

    items = "\n".join(
        f'\t\t\t\t\t<a href="{pages_prefix}{f}"{cur(f)}><span>{n}</span>{t}</a>' for n, f, t in LESSONS
    )
    mobile = "\n".join(
        f'\t\t<a href="{pages_prefix}{f}"{cur(f)}><span>{n}</span>{t}</a>' for n, f, t in LESSONS
    )
    return f'''<a class="skip-link" href="#main">Skip to lesson</a>
\t<header class="menubar">
\t\t<div class="menubar-inner">
\t\t\t<a class="wordmark" href="{prefix}index.html"><i aria-hidden="true"></i>Python to AI</a>
\t\t\t<div class="mb-desktop">
\t\t\t\t<a class="mb-link" href="{pages_prefix}history_of_ai.html"{cur("history_of_ai.html")}>History</a>
\t\t\t\t<details>
\t\t\t\t\t<summary>Lessons ▾</summary>
\t\t\t\t\t<nav class="dropdown" aria-label="Lessons">
{items}
\t\t\t\t\t</nav>
\t\t\t\t</details>
\t\t\t\t<a class="mb-link" href="{YOUTUBE}" target="_blank" rel="noopener">Videos</a>
\t\t\t\t<a class="mb-link" href="{GITHUB}" target="_blank" rel="noopener">GitHub</a>
\t\t\t</div>
\t\t\t<span class="mb-spacer"></span>
\t\t\t<button class="theme-toggle" type="button" aria-label="Toggle dark mode"><span class="when-light">◐ <span class="label">Dark</span></span><span class="when-dark">◑ <span class="label">Light</span></span></button>
\t\t\t<button class="mb-mobile-btn" type="button" aria-expanded="false" aria-controls="mobile-panel">Menu</button>
\t\t</div>
\t\t<nav id="mobile-panel" class="mobile-panel" aria-label="Site">
\t\t<a href="{pages_prefix}history_of_ai.html"{cur("history_of_ai.html")}><span>--</span>History of AI</a>
{mobile}
\t\t<a href="{YOUTUBE}" target="_blank" rel="noopener"><span>▶</span>Videos</a>
\t\t<a href="{GITHUB}" target="_blank" rel="noopener"><span>{{}}</span>GitHub</a>
\t\t</nav>
\t</header>'''


FOOTER = f'''<footer class="site-footer">
\t\t<div class="band-thin"></div>
\t\t<div class="inner"><div class="row">
\t\t\t<span>&gt; built by Ishani Kathuria_</span>
\t\t\t<span class="links">
\t\t\t\t<a href="{YOUTUBE}" target="_blank" rel="noopener">YouTube</a>
\t\t\t\t<a href="{GITHUB}" target="_blank" rel="noopener">GitHub</a>
\t\t\t\t<a href="https://linkedin.com/in/ishani-kathuria" target="_blank" rel="noopener">LinkedIn</a>
\t\t\t\t<a href="https://ishani.kathuria.net" target="_blank" rel="noopener">Website</a>
\t\t\t</span>
\t\t</div></div>
\t</footer>'''


def label_inputs(html):
    """Point each <label> at the input/select that follows it in the same block."""
    pattern = re.compile(
        r'<label(?![^>]*\bfor=)([^>]*)>(.*?)</label>(\s*(?:<[^>]+>\s*)*?)<(input|select|textarea)\b([^>]*?)\bid="([^"]+)"',
        re.S,
    )

    def repl(m):
        attrs, inner, between, tag, pre, id_ = m.groups()
        # Only bridge simple wrappers, not whole sections.
        if len(between) > 200 or "<label" in between:
            return m.group(0)
        return f'<label for="{id_}"{attrs}>{inner}</label>{between}<{tag}{pre}id="{id_}"'

    return pattern.sub(repl, html)


def migrate(path):
    html = path.read_text()
    if 'class="menubar"' in html:
        return False
    name = path.name

    # Head: drop Nunito, add brand assets and Tailwind token mapping.
    html = re.sub(r'\s*<link href="https://fonts.googleapis.com/css2\?family=Nunito[^>]*>', "", html)
    html = html.replace(
        '<script src="https://cdn.tailwindcss.com"></script>',
        '<script src="https://cdn.tailwindcss.com"></script>\n\t<script src="../styles/tw-config.js"></script>',
        1,
    )
    html = html.replace("</head>", head_assets("../../") + "</head>", 1)
    html = re.sub(r"font-family: 'Nunito', sans-serif;", "", html)

    # Title format: "<Lesson> · Python to AI".
    html = re.sub(r"<title>(.*?) - Python to AI</title>", r"<title>\1 · Python to AI</title>", html)

    # Top nav (+ its mobile menu) -> menu bar.
    html = re.sub(
        r'<!-- NAVIGATION -->\s*<nav class="bg-white shadow-sm sticky.*?</nav>',
        menubar("../../", "", name),
        html,
        count=1,
        flags=re.S,
    )
    if 'class="menubar"' not in html:
        html = re.sub(
            r'<nav class="bg-white shadow-sm sticky.*?</nav>', menubar("../../", "", name), html, count=1, flags=re.S
        )

    # Old hamburger script targets removed ids.
    html = re.sub(
        r"<script>\s*document\.getElementById\('mobile-menu-btn'\).*?</script>", "", html, count=1, flags=re.S
    )

    html = re.sub(r"<footer.*?</footer>", FOOTER, html, count=1, flags=re.S)
    html = re.sub(r'<main class="', '<main id="main" class="', html, count=1)
    html = html.replace("🔵", "")
    html = label_inputs(html)
    path.write_text(html)
    return True


if __name__ == "__main__":
    targets = [PAGES / a for a in sys.argv[1:]] or [PAGES / f for _, f, _ in LESSONS]
    for p in targets:
        print(("updated " if migrate(p) else "skipped ") + p.name)
