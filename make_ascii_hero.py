#!/usr/bin/env python3
"""
make_ascii_hero.py  -  turns your avatar into animated @!$% character art
on a neon "Matrix" code-rain background, saved as assets/hero-ascii.svg

Usage:
    pip install pillow numpy scipy
    python make_ascii_hero.py avatar.jpg

Re-run it whenever you change the picture or the text below.
"""
import html
import os
import random
import sys

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

# ------------------------- EDIT THESE -------------------------
SRC = sys.argv[1] if len(sys.argv) > 1 else "avatar.jpg"
OUT = "assets/hero-ascii.svg"
USER = "edutanvir"

LEFT = [  # (kind, text)   kind: cmd = prompt+command, out = output line
    ("cmd", "whoami"),
    ("out", "> Azure Data Engineer"),
    ("out", "> Pipelines that scale"),
    ("gap", ""),
    ("cmd", "status"),
    ("live", "> OPEN_TO_WORK = true"),
    ("gap", ""),
    ("cmd", "learning"),
    ("cursor", "> AI | LLMs | RAG "),
]
RIGHT = [
    "+-- adf.pipelines",
    "+-- databricks.pyspark",
    "+-- synapse.adls_gen2",
    "+-- fabric.power_bi",
    "+-- azure_sql.spark_sql",
    "+-- keyvault.rbac.mi",
    "+-- devops.ci_cd",
    "`-- ai.llm.rag [learning]",
]
# --------------------------------------------------------------

W, H = 1000, 612
COLS = 96
CW, CH = 3.8, 6.3            # size of one character cell
GROUPS = [".", ",'", ":;", "!|", "+=", "*#", "&$", "%@"]   # sparse -> dense, picked at random per group
random.seed(11)
esc = html.escape

# ---------- 1. cut the person out of the background ----------
im = Image.open(SRC).convert("RGB")
a = np.asarray(im).astype(float)
h0, w0 = a.shape[:2]
probe = [(w0 // 2, int(h0 * .06)), (int(w0 * .12), h0 // 2),
         (w0 // 2, int(h0 * .88)), (int(w0 * .86), h0 // 2)]
bg = np.median([a[y, x] for x, y in probe], axis=0)
dist = np.sqrt(((a - bg) ** 2).sum(2))
m = ndi.binary_opening(dist > 14, iterations=2)
m = ndi.binary_closing(m, iterations=6)
lab, _ = ndi.label(m)
m = ndi.binary_fill_holes(lab == lab[h0 // 2, w0 // 2])

ys, xs = np.where(m)
x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
bw, bh = x1 - x0, y1 - y0
ROWS = int(round(COLS * (bh / bw) * (CW / CH)))
box = (x0, y0, x1, y1)

small = np.asarray(im.crop(box).resize((COLS, ROWS), Image.BOX)).astype(float)
frac = np.asarray(Image.fromarray((m * 255).astype("uint8")).crop(box)
                  .resize((COLS, ROWS), Image.BOX)).astype(float) / 255
inc = frac > .5

R, G, B = small[..., 0], small[..., 1], small[..., 2]
L0 = .299 * R + .587 * G + .114 * B
L = np.clip(L0 + 1.3 * (L0 - ndi.gaussian_filter(L0, 1.1)), 0, 255)   # sharpen features
skin = inc & (L >= 110)
lo, hi = np.percentile(L[skin], [3, 97])
t = np.clip((L - lo) / (hi - lo), 0, 1)
dark = L < 110
d = np.where(dark, .78 + .22 * (1 - L / 110), .72 - .42 * t)

def cls(i, j):
    r, g, b, l = R[i, j], G[i, j], B[i, j], L[i, j]
    if l > 232:
        return "w"                       # specular highlights / teeth
    in_mouth_area = .68 < i / ROWS < .9 and .25 < j / COLS < .75
    if in_mouth_area and l < 110 and (r - g) / max(r, 1) > .3:
        return "p"                       # mouth interior
    if l < 55:
        return "a"
    if l < 110:
        return "b"
    tt = t[i, j]
    return "c" if tt < .35 else ("d" if tt < .75 else "e")

# ---------- 2. portrait rows ----------
pw, ph = COLS * CW, ROWS * CH
PX, PY = (W - pw) / 2, (H - ph) / 2
portrait = []
cells = []
for i in range(ROWS):
    row, j = "", 0
    while j < COLS:
        if not inc[i, j]:
            j += 1
            continue
        k = j
        while k < COLS and inc[i, k]:
            k += 1
        n = k - j
        x = PX + j * CW
        y = PY + i * CH
        row += (f'<rect x="{x:.1f}" y="{y:.1f}" width="{n * CW:.1f}" '
                f'height="{CH + .4:.1f}" fill="#02130a" opacity=".9"/>')
        spans, cur, buf = [], None, ""
        for jj in range(j, k):
            c = cls(i, jj)
            grp = GROUPS[min(len(GROUPS) - 1, max(0, int(d[i, jj] * len(GROUPS))))]
            ch = random.choice(grp)
            cells.append((x + (jj - j) * CW, y, c))
            if c != cur and buf:
                spans.append((cur, buf))
                buf = ""
            cur = c
            buf += ch
        spans.append((cur, buf))
        tsp = "".join(f'<tspan class="{c}">{esc(s)}</tspan>' for c, s in spans)
        row += (f'<text x="{x:.1f}" y="{y + CH - 1.6:.1f}" textLength="{n * CW:.1f}" '
                f'lengthAdjust="spacing" class="pt">{tsp}</text>')
        j = k
    portrait.append(f'<g class="pr" style="animation-delay:{1.0 + i * .022:.2f}s">{row}</g>')

# ---------- 3. matrix rain ----------
RAIN = "01@!$%#&*+=<>/\\|;:~^"
TAIL = 24
rain = []
x = 6.0
while x < W:
    chars = [random.choice(RAIN) for _ in range(TAIL)]
    grad = random.choices(["tg", "tc", "tm"], weights=[78, 17, 5])[0]
    tsp = "".join(f'<tspan x="{x:.1f}" dy="14">{esc(c)}</tspan>' for c in chars)
    dur = random.uniform(5.5, 12)
    delay = -random.uniform(0, 12)
    op = random.uniform(.4, 1)
    rain.append(f'<g class="rc" style="animation-duration:{dur:.1f}s;animation-delay:{delay:.1f}s" '
                f'opacity="{op:.2f}"><text y="0" fill="url(#{grad})" class="rt">{tsp}</text></g>')
    x += random.uniform(11, 15)

# ---------- 4. twinkling sparkles on the portrait ----------
sparks = []
for (sx, sy, c) in random.sample(cells, 60):
    ch = random.choice("01@$%!#")
    col = random.choice(["#ffffff", "#00e5ff", "#ff2bd6", "#39ff14", "#ffe600"])
    dur, beg = random.uniform(1.6, 4.2), random.uniform(2.5, 8)
    sparks.append(f'<text x="{sx:.1f}" y="{sy + CH - 1:.1f}" fill="{col}" class="sp" opacity="0">{ch}'
                  f'<animate attributeName="opacity" values="0;1;0" dur="{dur:.1f}s" begin="{beg:.1f}s" '
                  f'repeatCount="indefinite"/></text>')

# ---------- 5. terminal panels with typing animation ----------
clips, ltxt = [], []
def typed(idx, x, y, inner, nchars, begin, dur):
    cid = f"k{idx}"
    clips.append(f'<clipPath id="{cid}"><rect x="{x - 2}" y="{y - 15}" width="0" height="20">'
                 f'<animate attributeName="width" from="0" to="{nchars * 8 + 6}" begin="{begin:.2f}s" '
                 f'dur="{dur:.2f}s" fill="freeze"/></rect></clipPath>')
    return f'<text x="{x}" y="{y}" class="tt" clip-path="url(#{cid})">{inner}</text>'

LX, LY, LH = 40, 196, 22
t_cur = 1.0
idx = 0
for n, (kind, txt) in enumerate(LEFT):
    y = LY + n * LH
    if kind == "gap":
        continue
    if kind == "cmd":
        inner = f'<tspan fill="#39ff14">root@{USER}:~$</tspan> <tspan fill="#ffffff">{esc(txt)}</tspan>'
        nch = len(f"root@{USER}:~$ {txt}")
    else:
        colr = {"out": "#00e5ff", "live": "#ffe600", "cursor": "#ff7bd5"}[kind]
        inner = f'<tspan fill="{colr}">{esc(txt)}</tspan>'
        nch = len(txt)
    dur = max(.5, nch * .035)
    ltxt.append(typed(idx, LX, y, inner, nch, t_cur, dur))
    if kind == "live":
        ltxt.append(f'<circle cx="{LX + 190}" cy="{y - 4}" r="4" fill="#39ff14" opacity="0">'
                    f'<animate attributeName="opacity" values="0;0;1;.2;1" dur="1.6s" begin="{t_cur + dur:.2f}s" '
                    f'repeatCount="indefinite"/></circle>')
    if kind == "cursor":
        cx = LX + nch * 7.85
        ltxt.append(f'<rect x="{cx:.1f}" y="{y - 12}" width="8" height="15" fill="#ff7bd5" opacity="0">'
                    f'<animate attributeName="opacity" values="0;1;1;0;0" keyTimes="0;.01;.5;.51;1" dur="1s" '
                    f'begin="{t_cur + dur:.2f}s" repeatCount="indefinite"/></rect>')
    t_cur += dur + .25
    idx += 1

rcolors = ["#ff2bd6", "#00e5ff", "#39ff14", "#ffe600", "#ff9f1c", "#b388ff", "#00ffa3", "#ff5c5c"]
RX, RY = 722, 176
rtxt = []
t_cur = 1.6
rtxt.append(typed(idx, RX, RY - 4, f'<tspan fill="#39ff14">$</tspan> <tspan fill="#fff">ls ./skills</tspan>', 14, t_cur, .5))
idx += 1
t_cur += .6
for n, line in enumerate(RIGHT):
    y = RY + 24 + n * LH
    inner = (f'<tspan fill="#4b6b57">{esc(line[:4])}</tspan>'
             f'<tspan fill="{rcolors[n % len(rcolors)]}">{esc(line[4:])}</tspan>')
    dur = max(.45, len(line) * .03)
    rtxt.append(typed(idx, RX, y, inner, len(line), t_cur, dur))
    t_cur += dur + .15
    idx += 1

lp_h = LY - 52 + (len(LEFT)) * LH + 4
rp_y = RY - 56
rp_h = 24 + (len(RIGHT) + 1) * LH + 46

def panel(x, y, w, h, title):
    return (f'<g><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="#02100a" opacity=".82" '
            f'stroke="#1f7a45" stroke-width="1.2"/>'
            f'<circle cx="{x + 16}" cy="{y + 15}" r="4.5" fill="#ff5f56"/><circle cx="{x + 32}" cy="{y + 15}" r="4.5" fill="#ffbd2e"/>'
            f'<circle cx="{x + 48}" cy="{y + 15}" r="4.5" fill="#27c93f"/>'
            f'<text x="{x + w - 14}" y="{y + 19}" text-anchor="end" class="ttl">{esc(title)}</text>'
            f'<line x1="{x}" y1="{y + 30}" x2="{x + w}" y2="{y + 30}" stroke="#1f7a45" stroke-opacity=".6"/></g>')

svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
<defs>
  <linearGradient id="tg" gradientUnits="userSpaceOnUse" x1="0" y1="0" x2="0" y2="{TAIL * 14}">
    <stop offset="0" stop-color="#39ff14" stop-opacity="0"/><stop offset=".7" stop-color="#39ff14" stop-opacity=".75"/>
    <stop offset=".93" stop-color="#9dffb0"/><stop offset="1" stop-color="#ffffff"/>
  </linearGradient>
  <linearGradient id="tc" gradientUnits="userSpaceOnUse" x1="0" y1="0" x2="0" y2="{TAIL * 14}">
    <stop offset="0" stop-color="#00e5ff" stop-opacity="0"/><stop offset=".7" stop-color="#00e5ff" stop-opacity=".75"/>
    <stop offset=".93" stop-color="#a8f6ff"/><stop offset="1" stop-color="#ffffff"/>
  </linearGradient>
  <linearGradient id="tm" gradientUnits="userSpaceOnUse" x1="0" y1="0" x2="0" y2="{TAIL * 14}">
    <stop offset="0" stop-color="#ff2bd6" stop-opacity="0"/><stop offset=".7" stop-color="#ff2bd6" stop-opacity=".75"/>
    <stop offset=".93" stop-color="#ffb3f1"/><stop offset="1" stop-color="#ffffff"/>
  </linearGradient>
  <radialGradient id="vig" cx=".5" cy=".5" r=".75">
    <stop offset=".45" stop-color="#000" stop-opacity="0"/><stop offset="1" stop-color="#000" stop-opacity=".7"/>
  </radialGradient>
  <linearGradient id="scan" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0" stop-color="#39ff14" stop-opacity="0"/><stop offset=".5" stop-color="#b9ffd0" stop-opacity=".55"/>
    <stop offset="1" stop-color="#39ff14" stop-opacity="0"/>
  </linearGradient>
  <filter id="glow" x="-5%" y="-3%" width="110%" height="106%">
    <feGaussianBlur stdDeviation=".8" result="b"/>
    <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
  </filter>
  <clipPath id="frame"><rect width="{W}" height="{H}" rx="16"/></clipPath>
  {"".join(clips)}
</defs>
<style>
  text{{font-family:"DejaVu Sans Mono",Consolas,Menlo,"Liberation Mono",monospace}}
  .rt{{font-size:13px;font-weight:700}}
  .rc{{animation:fall linear infinite}}
  @keyframes fall{{from{{transform:translateY(-{TAIL * 14 + 10}px)}}to{{transform:translateY({H + 20}px)}}}}
  .pt{{font-size:{CH}px;font-weight:700}}
  .pr{{opacity:0;animation:rv .4s ease forwards}}
  @keyframes rv{{from{{opacity:0}}to{{opacity:1}}}}
  .a{{fill:#12d64a}} .b{{fill:#39ff14}} .c{{fill:#00ffc8}} .d{{fill:#00e5ff}} .e{{fill:#c4fff2}}
  .w{{fill:#fff}} .p{{fill:#ff2bd6}}
  .tt{{font-size:13px}} .ttl{{font-size:10px;fill:#4b8a62}} .sp{{font-size:10px;font-weight:700}}
  .sweep{{animation:sw 7s linear infinite;opacity:0}}
  @keyframes sw{{0%{{transform:translateY(0);opacity:0}}4%{{opacity:1}}42%{{transform:translateY({ph:.0f}px);opacity:1}}46%,100%{{transform:translateY({ph:.0f}px);opacity:0}}}}
</style>

<g clip-path="url(#frame)">
  <rect width="{W}" height="{H}" fill="#010a05"/>
  <g>{"".join(rain)}</g>
  <rect width="{W}" height="{H}" fill="url(#vig)"/>

  <!-- the portrait -->
  <g filter="url(#glow)">{"".join(portrait)}</g>
  <rect class="sweep" x="{PX - 10:.0f}" y="{PY:.0f}" width="{pw + 20:.0f}" height="14" fill="url(#scan)"/>
  <g>{"".join(sparks)}</g>

  <!-- terminals -->
  {panel(24, LY - 50, 276, lp_h, f"{USER}@matrix")}
  {"".join(ltxt)}
  {panel(RX - 16, rp_y, 270, rp_h, "skills.sh")}
  {"".join(rtxt)}
</g>
<rect x="1.5" y="1.5" width="{W - 3}" height="{H - 3}" rx="15" fill="none" stroke="#39ff14" stroke-width="2">
  <animate attributeName="stroke" values="#39ff14;#00e5ff;#ff2bd6;#ffe600;#39ff14" dur="8s" repeatCount="indefinite"/>
</rect>
</svg>
'''

os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    f.write(svg)
print(f"Wrote {OUT}  ({len(svg) / 1024:.0f} KB, {COLS}x{ROWS} characters)")
