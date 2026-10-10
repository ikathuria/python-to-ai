"""Rewrite the menu bar and previous/next links on every page from scripts/lessons.py.

    python scripts/sync_nav.py           # update pages
    python scripts/sync_nav.py --check   # exit 1 if any page is out of date (used by the tests)
"""
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from apply_brand import menubar  # noqa: E402
from lessons import LESSONS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PAGES = ROOT / "app" / "pages"
SKIP = {"machine_learning.html"}  # redirect stub, no nav

MENUBAR = re.compile(r'<a class="skip-link".*?</header>', re.S)
PREV_NEXT = re.compile(r'(<!-- Module Progress & Navigation -->.*?<div class="flex gap-3">\n)(.*?)(\n\t*</div>)', re.S)
LINK = "flex items-center gap-2 px-4 py-2"
PREV = ('<a href="{}" class="' + LINK + ' border border-gray-200 rounded-xl text-sm text-gray-600 '
        'hover:bg-gray-50 hover:border-indigo-300 transition"><i class="fa-solid fa-arrow-left text-xs"></i> {}</a>')
NEXT = ('<a href="{}" class="' + LINK + ' bg-indigo-600 text-white rounded-xl text-sm font-semibold '
        'hover:bg-indigo-700 transition">{} <i class="fa-solid fa-arrow-right text-xs"></i></a>')
HOME = ('<a href="../../index.html" class="' + LINK + ' bg-indigo-600 text-white rounded-xl text-sm font-semibold '
        'hover:bg-indigo-700 transition">Back to Home <i class="fa-solid fa-house text-xs"></i></a>')


def prev_next(name):
    files = [f for _, f, _, _ in LESSONS]
    if name not in files:
        return None
    i = files.index(name)
    links = []
    if i > 0:
        links.append(PREV.format(LESSONS[i - 1][1], LESSONS[i - 1][3]))
    links.append(NEXT.format(LESSONS[i + 1][1], LESSONS[i + 1][3]) if i + 1 < len(LESSONS) else HOME)
    return "\n".join("\t\t\t\t\t\t\t" + link for link in links)


def render(path):
    html = path.read_text(encoding="utf-8")
    if path.parent == ROOT:
        bar = menubar("", "app/pages/", None)
    else:
        bar = menubar("../../", "", path.name)
    html = MENUBAR.sub(lambda m: bar, html, count=1)
    links = prev_next(path.name)
    if links is not None:
        html = PREV_NEXT.sub(lambda m: m.group(1) + links + m.group(3), html, count=1)
    return html


def pages():
    yield ROOT / "index.html"
    for p in sorted(PAGES.glob("*.html")):
        if p.name not in SKIP:
            yield p


def main(check):
    stale = []
    for p in pages():
        new = render(p)
        if new != p.read_text(encoding="utf-8"):
            stale.append(p.relative_to(ROOT))
            if not check:
                p.write_text(new, encoding="utf-8")
    for s in stale:
        print(("out of date: " if check else "updated: ") + str(s))
    missing = [f for _, f, _, _ in LESSONS if not (PAGES / f).is_file()]
    for f in missing:
        print("missing page: app/pages/" + f)
    return 1 if check and (stale or missing) else 0


if __name__ == "__main__":
    sys.exit(main("--check" in sys.argv))
