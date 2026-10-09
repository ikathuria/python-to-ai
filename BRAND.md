# Python to AI Brand Guidelines

> Source of truth for tokens: [`app/styles/brand.css`](app/styles/brand.css). If this doc and the tokens disagree, fix one.

## Essence
- **One line:** Free, friendly lessons that show non-tech people (and engineers) what AI really is, from the first line of Python to models running in the browser. Companion site to [youtube.com/@DrIshaniKathuria](https://www.youtube.com/@DrIshaniKathuria).
- **Personality:** witty, warm, demystifying. Never ominous, smug or hype-y.
- **Reference:** the 1977 Apple rainbow logo, classic Mac windows, 80s school computer labs. Dark mode: old-school movie hacker terminals (black screen, neon green text).
- **Energy level:** 4 · **Density:** balanced (lessons stay readable, hero and section breaks are loud)

## Voice & copy
- Tone: a friendly teacher with jokes. Plain English first, jargon second (and always explained).
- Do: "AI is older than your computer. Let's boot it up." · "Result: it's math. Relax." · "Never coded? Start here."
- Don't: "Unlock", "Master", "Harness the power", "Revolutionize", doom framing, hype claims.
- Casing: sentence case everywhere. File-name style labels (`02_supervised.py`) are lowercase.

## Logo
- Wordmark: "Python to AI" in Work Sans 800, preceded by a small six-stripe rainbow square (16×16, stripes per `--rainbow`).
- Dark mode: same wordmark, the square becomes solid neon green.
- Don't: recolour the stripes, reorder them, or put the wordmark on a rainbow block.

## Color
| Token | Light | Dark | Use |
|---|---|---|---|
| `--bg` | #F4F4F1 | #050805 | page background |
| `--surface` | #FFFFFF | #0C140D | windows, cards, inputs |
| `--case` | #E9E6DD | #0F1B10 | secondary surfaces, computer case |
| `--text` | #111111 | #C6F7C0 | body text |
| `--muted` | #4A4A4A | #7FBF78 | secondary text |
| `--border` | #111111 | #3F8F3A | 2px outlines |
| `--accent` | #C8282E | #39FF14 | primary buttons |
| `--link` | #0069A0 | #39FF14 | inline links |
| `--rb-*` | green #61BB46 · yellow #FDB827 · orange #F5821F · red #E03A3E · purple #963D97 · blue #009DDC | n/a | colour blocks, stripes, highlights, chart series |
| `--rainbow` | six stripes | neon CRT scanlines on black | the band (signature move 1) |
| `--title` / `--title-glow` | #111 / none | #39FF14 / soft glow | the page H1 only |
| `--screen` / `--screen-text` | #111 / #F4F4F4 | #000 / #39FF14 | terminal panels |

Rule: rainbow colours are **surfaces, never text colours**. Text on them is always `#111`.
Dark mode: neon green is for actions, links, the page H1 (with glow) and band text. Body text and H2/H3 stay pale green (`--text`) so long lessons don't strain the eyes.

## Typography
| Role | Font | Size / line-height | Weight | Tracking |
|---|---|---|---|---|
| Display (H1) | Work Sans | `--fs-h1` clamp(40–76px) / 1.02 | 800 | -0.025em |
| H2 / H3 | Work Sans | `--fs-h2` / `--fs-h3`, 1.1 | 800 | -0.015em |
| Body | Work Sans | 17px / 1.6, max 72ch | 400 (600 for emphasis) | 0 |
| Labels, file names, nav | Space Mono | 13–15px | 700 | 0.04em |
| Terminal screens | VT323 | 22–26px / 1.3 | 400 | 0 |
| Code blocks | Space Mono | 14–15px | 400 | 0 |

Loading: Google Fonts with `display=swap` (in `brand.css`). Three families, max 5 weights total.

## Layout & spacing
- Base unit 8px: `--s-1`…`--s-8` (4, 8, 16, 24, 32, 48, 64, 96).
- Max widths: `--wide` 1180px for pages, `--content` 72ch for lesson text.
- Section rhythm: loud full-bleed rainbow or terminal band → calm white lesson section → repeat. Never two loud bands in a row.
- Breakpoints: 640 / 860 / 1024 / 1280.
- Responsive: menu bar collapses to wordmark + menu button under 860px; lesson sidebar becomes a "Contents" dropdown under 1024px (never a fixed column on phones); window grids go 3 → 2 → 1 columns; hero type uses `clamp()`.

## Shape & depth
- Radius: `--radius-sm` 4px for buttons, inputs, windows. `--radius-lg` 14px only for the computer-case illustration.
- Elevation: hard offset shadows only (`--shadow-pop`), no blur. Hover moves to `--shadow-pop-hover`.
- Borders: 2px `--border` on all interactive surfaces.

## Signature moves
1. **Rainbow band:** a full-bleed six-stripe band (`--rainbow`) for big statements, with the headline on a white highlight. In dark mode it becomes neon CRT scanlines with glowing text (`--band-text`, `--band-glow`). Once per page at most.
2. **Lesson windows:** cards look like classic Mac windows: striped title bar, close box, a file name (`06_vision.py`), hard shadow.
3. **Terminal moments:** a `--screen` panel that "types" a witty exchange (`is_ai_going_to_take_over()` → `it's math. Relax.`). Use for heroes, quiz results and demo outputs.
4. **Menu bar nav:** a thin top bar like the classic Mac menu bar, with mono labels.

## Imagery & iconography
- Imagery: simple flat illustrations in the rainbow palette with 2px black outlines (computers, windows, floppy disks). Screenshots of live demos inside window frames.
- Icons: Font Awesome solid at 16–20px, always `--text` coloured, never multicolour; no emoji as icons.

## Motion
- `--dur` 150ms, `--ease`. Blinking terminal cursor, typed terminal lines, hover shadow shift.
- No fade-up on every section. All motion off with `prefers-reduced-motion`.

## Components
- **Buttons:** primary = `--accent` fill + `--on-accent` text + 2px border; secondary = `--surface` fill. Mono label, min 44px tall. Hover: invert to `--text` background.
- **Inputs/selects:** `--surface`, 2px border, visible `<label>` always, 44px tall, focus ring.
- **Cards:** lesson windows (above). No nested cards.
- **Code blocks:** `--screen` background, Space Mono, copy button in the title bar.
- **States:** loading = blinking cursor + "Loading model…"; error = terminal panel with `ERROR:` line in plain English; empty = friendly one-liner.

## Do / Don't
| Do | Don't |
|---|---|
| Use one rainbow band per page as the loud moment | Rainbow gradients on text or buttons |
| Black text on rainbow stripes | Indigo/purple gradient "AI" look |
| Joke, then explain | Doom or "master AI" hype copy |
| Window frames for lessons and demos | `rounded-2xl` soft cards with blurred shadows |

## Accessibility
- Focus ring: 3px solid `--focus` with 2px offset on every interactive element.
- Targets ≥ 44px. Skip link "Skip to lesson" on every page.

Contrast matrix (WCAG):

| Pair | Light | Dark |
|---|---|---|
| text / bg | 17.1 | 16.7 |
| muted / bg | 8.0 | 9.2 |
| on-accent / accent | 5.5 | 14.8 |
| link / bg | 5.4 | 14.8 |
| #111 on yellow · orange · green · blue | 10.9 · 7.3 · 7.8 · 6.2 | n/a |
| #111 on rainbow red | 4.4 (large text only) | n/a |
| screen text / screen | 17.2 | 14.8 |

## SEO & metadata
- Title format: `<Lesson> · Python to AI` (30–60 chars). Description: plain English, specific, 70–160 chars.
- OG image: 1200×630, wordmark + rainbow band + lesson title, at `app/images/og/<page>.png`. Favicon: rainbow square (`favicon.svg`, 180px apple-touch-icon).
- One H1 per page, no skipped heading levels.
