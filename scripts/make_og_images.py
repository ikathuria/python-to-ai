"""Generate branded social share images (1200x630) and the apple-touch-icon. See BRAND.md."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "app" / "images" / "og"
BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
MONO = "/System/Library/Fonts/Menlo.ttc"
STRIPES = ["#61BB46", "#FDB827", "#F5821F", "#E03A3E", "#963D97", "#009DDC"]
INK, BG = "#111111", "#F4F4F1"

PAGES = {
    "home": ("AI is older than your computer.", "Let's boot it up."),
    "history_of_ai": ("History of AI", "80 years of hype vs. reality"),
    "python": ("Python basics", "lesson 00 · never coded? start here"),
    "ml_basics": ("ML basics", "lesson 01 · NumPy, Pandas, data"),
    "supervised_learning": ("Teach a machine", "lesson 02 · supervised learning"),
    "unsupervised_learning": ("Find hidden groups", "lesson 03 · unsupervised learning"),
    "recommendation_system": ("Why Netflix knows you", "lesson 04 · recommendation systems"),
    "deep_learning": ("Neural networks", "lesson 05 · deep learning"),
    "computer_vision": ("How computers see", "lesson 06 · computer vision"),
    "time_series": ("Predicting tomorrow", "lesson 07 · time series"),
    "natural_language_processing": ("Fancy autocomplete", "lesson 08 · language & NLP"),
}


def wrap(draw, text, font, width):
    words, lines, line = text.split(), [], ""
    for w in words:
        trial = (line + " " + w).strip()
        if draw.textlength(trial, font=font) <= width:
            line = trial
        else:
            lines.append(line)
            line = w
    return lines + [line]


def stripes(draw, x, y, w, h, vertical=False):
    for i, c in enumerate(STRIPES):
        if vertical:
            draw.rectangle([x + i * w / 6, y, x + (i + 1) * w / 6, y + h], fill=c)
        else:
            draw.rectangle([x, y + i * h / 6, x + w, y + (i + 1) * h / 6], fill=c)


def og(slug, title, sub):
    im = Image.new("RGB", (1200, 630), BG)
    d = ImageDraw.Draw(im)
    # Menu bar
    d.rectangle([0, 0, 1200, 64], fill="#FFFFFF")
    d.line([0, 64, 1200, 64], fill=INK, width=3)
    d.rectangle([40, 20, 66, 46], fill=INK)
    stripes(d, 43, 23, 20, 20)
    d.text((80, 18), "Python to AI", font=ImageFont.truetype(BOLD, 26), fill=INK)
    # Title
    f = ImageFont.truetype(BOLD, 78)
    y = 120
    for line in wrap(d, title, f, 1080)[:3]:
        d.text((60, y), line, font=f, fill=INK)
        y += 90
    d.text((60, y + 16), "> " + sub, font=ImageFont.truetype(MONO, 32), fill="#4A4A4A")
    # Rainbow band
    d.line([0, 528, 1200, 528], fill=INK, width=3)
    stripes(d, 0, 530, 1200, 100)
    im.save(OUT / f"{slug}.png", optimize=True)


def touch_icon():
    im = Image.new("RGB", (180, 180), INK)
    stripes(ImageDraw.Draw(im), 20, 20, 140, 140)
    im.save(OUT / "apple-touch-icon.png", optimize=True)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for slug, (t, s) in PAGES.items():
        og(slug, t, s)
    touch_icon()
    print(f"wrote {len(PAGES) + 1} images to {OUT}")
