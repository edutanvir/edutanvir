#!/usr/bin/env python3
"""
make_ascii_hero.py - turns your avatar into a ROTATING 3D character-art head
(@!$% glyphs, white/silver tone) over neon code rain, saved as assets/hero-ascii.svg

How it works: the avatar is "inflated" into a 3D bust (depth from the silhouette),
ray-marched from many angles, and every angle becomes one ASCII frame inside the SVG.
CSS animation plays the frames in order, so the head spins on GitHub with no JavaScript.

Usage:
    pip install pillow numpy scipy
    python make_ascii_hero.py avatar.jpg

Re-run it whenever you change the picture or the settings below.
"""
import html
import math
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

FRAMES = 48            # angles in one full turn (more = smoother but bigger file)
FRAME_SEC = 0.15       # seconds each frame is shown -> one turn = FRAMES * FRAME_SEC
SPIN_START = 4.4       # seconds of front view (reveal + pause) before the head starts to spin
DEPTH = 0.95           # head depth relative to its width (bigger = rounder head)

SKIN_COLOR = "#aab6c3"   # silver-grey glyphs (face)
DARK_COLOR = "#f2f7fc"   # bright white glyphs (hair, beard, eyes)

LEFT = [  # (kind, text)   kind: cmd = prompt+command, out = output line
    ("cmd", "whoami"),
    ("out", "> Md Tanvir Ahmed"),
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
SS = 3                       # sub-rays per cell side (anti-aliasing)
GROUPS = [".", ",'", ":;", "!|", "+=", "*#", "&$", "%@"]   # sparse -> dense glyph groups
random.seed(11)
esc = html.escape

# ---------- 1. cut the person out of the background ----------
im = Image.open(SRC).convert("RGB")
a = np.asarray(im).astype(np.float32)
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
cwpx, chpx = bw / COLS, bh / ROWS      # source pixels per character cell
xc = (x0 + x1) / 2.0

L0 = .299 * a[..., 0] + .587 * a[..., 1] + .114 * a[..., 2]
# sharpen so eyes, brows and mouth stay crisp at character resolution
Ls = np.clip(L0 + 1.3 * (L0 - ndi.gaussian_filter(L0, (1.1 * chpx, 1.1 * cwpx))), 0, 255).astype(np.float32)

# ---------- 2. inflate the silhouette into a 3D bust ----------
dt = ndi.distance_transform_edt(m)
HMAX = DEPTH * bw / 2
RPROF = .75 * bw / 2
u = np.clip(dt / RPROF, 0, 1)
hgt = HMAX * np.sqrt(1 - (1 - u) ** 2)                       # round "pillow" profile
yy, xx = np.mgrid[0:h0, 0:w0]
nose_x, nose_y = x0 + .49 * bw, y0 + .645 * bh                 # small nose bump for a more 3D face
hgt += 55 * np.exp(-(((xx - nose_x) / 45) ** 2 + ((yy - nose_y) / 45) ** 2) / 2)
hgt = (ndi.gaussian_filter(hgt, 8) * m).astype(np.float32)
hy_, hx_ = np.gradient(hgt)
hy_, hx_ = hy_.astype(np.float32), hx_.astype(np.float32)

# hair texture for the back of the head: vertical strands
rng = np.random.RandomState(5)
NZ = ndi.gaussian_filter(rng.rand(256, 256), (10, .9), mode="wrap")
NZ = (NZ - NZ.min()) / (NZ.max() - NZ.min())

LIGHT = np.array([-.45, -.55, .70])
LIGHT /= np.linalg.norm(LIGHT)

# rays: SS x SS per cell, one slice per source row
sub_x = ((np.arange(COLS * SS) + .5) / SS) * cwpx - bw / 2          # screen x of each ray column
sub_y = y0 + ((np.arange(ROWS * SS) + .5) / SS) * chpx              # source row of each ray row
RX, RY = np.meshgrid(sub_x, sub_y)
XP = RX.ravel().astype(np.float32)
ROW = np.clip(RY.ravel().astype(int), 0, h0 - 1)
NRAY = XP.size
ZF = math.hypot(bw / 2, HMAX) * 1.05 + 10
STEP = 3.0


def inside(xp, zp, c, s, rows):
    x = xp * c - zp * s + xc
    z = xp * s + zp * c
    ok = (x >= 0) & (x <= w0 - 1.001)
    xi = np.clip(x, 0, w0 - 1.001)
    xf = xi.astype(int)
    fr = xi - xf
    hv = hgt[rows, xf] * (1 - fr) + hgt[rows, xf + 1] * fr
    return ok & (np.abs(z) <= hv)


def render(theta):
    """Ray-march the bust turned by theta. Returns per-cell coverage, luminance, shade."""
    c, s = math.cos(theta), math.sin(theta)
    hit = np.zeros(NRAY, bool)
    lo = np.zeros(NRAY, np.float32)
    hi = np.zeros(NRAY, np.float32)
    for zp in np.arange(ZF, -ZF, -STEP):
        new = inside(XP, zp, c, s, ROW) & ~hit
        if new.any():
            hit |= new
            hi[new] = zp
            lo[new] = zp + STEP
    idx = np.where(hit)[0]
    xp, rows = XP[idx], ROW[idx]
    l_, h_ = lo[idx], hi[idx]
    for _ in range(6):                                         # refine the surface position
        mid = (l_ + h_) / 2
        ins = inside(xp, mid, c, s, rows)
        h_ = np.where(ins, mid, h_)
        l_ = np.where(ins, l_, mid)
    zp = h_
    x = xp * c - zp * s + xc
    z = xp * s + zp * c
    front = z >= 0
    xi = np.clip(x, 0, w0 - 1.001)
    xf = xi.astype(int)
    fr = xi - xf
    hxv = hx_[rows, xf] * (1 - fr) + hx_[rows, xf + 1] * fr
    hyv = hy_[rows, xf] * (1 - fr) + hy_[rows, xf + 1] * fr
    # texture: your picture on the front, procedural hair on the back
    tex = Ls[rows, np.clip(np.rint(x).astype(int), 0, w0 - 1)]
    phi = np.arctan2(z, x - xc)
    back = 24 + 52 * NZ[(rows * .30).astype(int) % 256, (phi * 150).astype(int) % 256]
    lum = np.where(front, tex, back)
    # lighting (light fixed, head turns under it)
    nx, ny, nz = -hxv, -hyv, np.where(front, 1.0, -1.0)
    nrm = np.sqrt(nx * nx + ny * ny + nz * nz)
    nxr, nzr = nx * c + nz * s, -nx * s + nz * c
    shade = np.clip((nxr * LIGHT[0] + ny * LIGHT[1] + nzr * LIGHT[2]) / nrm / .75, 0, 1)
    full_l = np.zeros(NRAY, np.float32)
    full_s = np.zeros(NRAY, np.float32)
    full_h = np.zeros(NRAY, np.float32)
    full_l[idx], full_s[idx], full_h[idx] = lum, shade, 1
    def cellsum(v):
        return v.reshape(ROWS, SS, COLS, SS).sum(axis=(1, 3))
    hits = cellsum(full_h)
    cov = hits / (SS * SS)
    return cov, cellsum(full_l) / np.maximum(hits, 1), cellsum(full_s) / np.maximum(hits, 1)


# ---------- 3. render every angle ----------
print(f"ray-marching {FRAMES} frames ...")
frames = [render(2 * math.pi * k / FRAMES) for k in range(FRAMES)]
cov0, l0, _ = frames[0]
skin0 = (cov0 > .5) & (l0 >= 110)
lo_s, hi_s = np.percentile(l0[skin0], [3, 97])


def density(l, sh):
    t = np.clip((l - lo_s) / (hi_s - lo_s), 0, 1)
    d = np.where(l < 110, .78 + .22 * (1 - l / 110), .72 - .42 * t)
    return np.clip(d + .20 * (1 - sh), 0, 1)


pw, ph = COLS * CW, ROWS * CH
PX, PY = (W - pw) / 2, (H - ph) / 2


def build_rows(cov, l, sh):
    """-> list of (backing path piece, <text> markup) per character row"""
    inc = cov > .5
    d = density(l, sh)
    out = []
    for i in range(ROWS):
        piece, txt, j = "", "", 0
        while j < COLS:
            if not inc[i, j]:
                j += 1
                continue
            k = j
            while k < COLS and inc[i, k]:
                k += 1
            n = k - j
            x, y = PX + j * CW, PY + i * CH
            piece += f"M{x:.1f} {y:.1f}h{n * CW:.1f}v{CH + .4:.1f}h-{n * CW:.1f}z"
            segs, cur, buf = [], None, ""
            for jj in range(j, k):
                grp = GROUPS[min(len(GROUPS) - 1, max(0, int(d[i, jj] * len(GROUPS))))]
                ch = grp[((i * 73856093) ^ (jj * 19349663)) % len(grp)]
                dark = l[i, jj] < 110
                if dark != cur and buf:
                    segs.append((cur, buf))
                    buf = ""
                cur = dark
                buf += ch
            segs.append((cur, buf))
            body = "".join(f'<tspan class="h">{esc(sg)}</tspan>' if dk else esc(sg) for dk, sg in segs)
            txt += f'<text x="{x:.1f}" y="{y + CH - 1.6:.1f}" textLength="{n * CW:.1f}">{body}</text>'
            j = k
        out.append((piece, txt))
    return out


intro_rows = build_rows(*frames[0])
intro = "".join(
    f'<g class="pr" style="animation-delay:{1.0 + i * .022:.2f}s">'
    f'<path class="bk" d="{p}"/>{t}</g>' for i, (p, t) in enumerate(intro_rows))

T_SPIN = FRAMES * FRAME_SEC
spin = []
for k, fr_data in enumerate(frames):
    rows = build_rows(*fr_data)
    spin.append(f'<g class="fr" style="animation-delay:{SPIN_START + k * FRAME_SEC:.3f}s">'
                f'<path class="bk" d="{"".join(p for p, _ in rows)}"/>{"".join(t for _, t in rows)}</g>')

# cells that are part of the head in (almost) every frame -> where sparkles may appear
occ = sum((f[0] > .5).astype(int) for f in frames)
stable = [(i, j) for i in range(ROWS) for j in range(COLS) if occ[i, j] >= .85 * FRAMES]

# ---------- 4. matrix rain ----------
RAIN = "01@!$%#&*+=<>/\\|;:~^"
TAIL = 24
rain = []
x = 6.0
while x < W:
    chars = [random.choice(RAIN) for _ in range(TAIL)]
    grad = random.choices(["tg", "tc", "tw", "tm"], weights=[40, 32, 18, 10])[0]
    tsp = "".join(f'<tspan x="{x:.1f}" dy="14">{esc(c)}</tspan>' for c in chars)
    dur = random.uniform(5.5, 12)
    delay = -random.uniform(0, 12)
    op = random.uniform(.3, .85)
    rain.append(f'<g class="rc" style="animation-duration:{dur:.1f}s;animation-delay:{delay:.1f}s" '
                f'opacity="{op:.2f}"><text y="0" fill="url(#{grad})" class="rt">{tsp}</text></g>')
    x += random.uniform(11, 15)

# ---------- 5. twinkling sparkles on the head ----------
sparks = []
for (si, sj) in random.sample(stable, min(60, len(stable))):
    ch = random.choice("01@$%!#")
    col = random.choice(["#ffffff", "#00e5ff", "#ff8c00", "#39ff14", "#ffe600"])
    dur, beg = random.uniform(1.6, 4.2), random.uniform(3.5, 9)
    sparks.append(f'<text x="{PX + sj * CW:.1f}" y="{PY + si * CH + CH - 1:.1f}" fill="{col}" class="sp" opacity="0">{ch}'
                  f'<animate attributeName="opacity" values="0;1;0" dur="{dur:.1f}s" begin="{beg:.1f}s" '
                  f'repeatCount="indefinite"/></text>')

# ---------- 6. terminal panels with typing animation ----------
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
        colr = {"out": "#00e5ff", "live": "#ffe600", "cursor": "#ffb703"}[kind]
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
        ltxt.append(f'<rect x="{cx:.1f}" y="{y - 12}" width="8" height="15" fill="#ffb703" opacity="0">'
                    f'<animate attributeName="opacity" values="0;1;1;0;0" keyTimes="0;.01;.5;.51;1" dur="1s" '
                    f'begin="{t_cur + dur:.2f}s" repeatCount="indefinite"/></rect>')
    t_cur += dur + .25
    idx += 1

rcolors = ["#ff8c00", "#00e5ff", "#39ff14", "#ffe600", "#ff6b35", "#2ee6a6", "#4da3ff", "#ffb703"]
RX0, RY0 = 722, 176
rtxt = []
t_cur = 1.6
rtxt.append(typed(idx, RX0, RY0 - 4, '<tspan fill="#39ff14">$</tspan> <tspan fill="#fff">ls ./skills</tspan>', 14, t_cur, .5))
idx += 1
t_cur += .6
for n, line in enumerate(RIGHT):
    y = RY0 + 24 + n * LH
    inner = (f'<tspan fill="#4b6b57">{esc(line[:4])}</tspan>'
             f'<tspan fill="{rcolors[n % len(rcolors)]}">{esc(line[4:])}</tspan>')
    dur = max(.45, len(line) * .03)
    rtxt.append(typed(idx, RX0, y, inner, len(line), t_cur, dur))
    t_cur += dur + .15
    idx += 1

lp_h = LY - 52 + (len(LEFT)) * LH + 4
rp_y = RY0 - 56
rp_h = 24 + (len(RIGHT) + 1) * LH + 46

def panel(x, y, w, h, title):
    return (f'<g><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="#02100a" opacity=".82" '
            f'stroke="#1f7a45" stroke-width="1.2"/>'
            f'<circle cx="{x + 16}" cy="{y + 15}" r="4.5" fill="#ff5f56"/><circle cx="{x + 32}" cy="{y + 15}" r="4.5" fill="#ffbd2e"/>'
            f'<circle cx="{x + 48}" cy="{y + 15}" r="4.5" fill="#27c93f"/>'
            f'<text x="{x + w - 14}" y="{y + 19}" text-anchor="end" class="ttl">{esc(title)}</text>'
            f'<line x1="{x}" y1="{y + 30}" x2="{x + w}" y2="{y + 30}" stroke="#1f7a45" stroke-opacity=".6"/></g>')

WP = 100 / FRAMES
def tail(name, c1, c2):
    return (f'<linearGradient id="{name}" gradientUnits="userSpaceOnUse" x1="0" y1="0" x2="0" y2="{TAIL * 14}">'
            f'<stop offset="0" stop-color="{c1}" stop-opacity="0"/><stop offset=".7" stop-color="{c1}" stop-opacity=".7"/>'
            f'<stop offset=".93" stop-color="{c2}"/><stop offset="1" stop-color="#ffffff"/></linearGradient>')

svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
<defs>
  {tail("tg", "#39ff14", "#9dffb0")}
  {tail("tc", "#00e5ff", "#a8f6ff")}
  {tail("tw", "#d7e2ee", "#f4f9ff")}
  {tail("tm", "#ff8c00", "#ffd9a8")}
  <radialGradient id="vig" cx=".5" cy=".5" r=".75">
    <stop offset=".45" stop-color="#000" stop-opacity="0"/><stop offset="1" stop-color="#000" stop-opacity=".7"/>
  </radialGradient>
  <radialGradient id="halo" cx=".5" cy=".5" r=".5">
    <stop offset="0" stop-color="#cfefff" stop-opacity=".20"/><stop offset=".6" stop-color="#7fd8ff" stop-opacity=".07"/>
    <stop offset="1" stop-color="#7fd8ff" stop-opacity="0"/>
  </radialGradient>
  <linearGradient id="scan" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0" stop-color="#ffffff" stop-opacity="0"/><stop offset=".5" stop-color="#ffffff" stop-opacity=".45"/>
    <stop offset="1" stop-color="#ffffff" stop-opacity="0"/>
  </linearGradient>
  <clipPath id="frame"><rect width="{W}" height="{H}" rx="16"/></clipPath>
  {"".join(clips)}
</defs>
<style>
  text{{font-family:"DejaVu Sans Mono",Consolas,Menlo,"Liberation Mono",monospace}}
  .rt{{font-size:13px;font-weight:700}}
  .rc{{animation:fall linear infinite}}
  @keyframes fall{{from{{transform:translateY(-{TAIL * 14 + 10}px)}}to{{transform:translateY({H + 20}px)}}}}
  .head text{{font-size:{CH}px;font-weight:700;fill:{SKIN_COLOR}}}
  .head .h{{fill:{DARK_COLOR}}}
  .bk{{fill:#03070a;opacity:.93}}
  .pr{{opacity:0;animation:rv .4s ease forwards}}
  @keyframes rv{{from{{opacity:0}}to{{opacity:1}}}}
  .intro{{animation:introout {SPIN_START}s linear forwards}}
  @keyframes introout{{0%,99.9%{{visibility:visible}}100%{{visibility:hidden}}}}
  .fr{{visibility:hidden;animation:spin {T_SPIN:.3f}s linear infinite}}
  @keyframes spin{{0%{{visibility:visible}}{WP:.4f}%{{visibility:visible}}{WP + .02:.4f}%{{visibility:hidden}}100%{{visibility:hidden}}}}
  .tt{{font-size:13px}} .ttl{{font-size:10px;fill:#4b8a62}} .sp{{font-size:10px;font-weight:700}}
  .sweep{{animation:sw 7s linear infinite;opacity:0}}
  @keyframes sw{{0%{{transform:translateY(0);opacity:0}}4%{{opacity:1}}42%{{transform:translateY({ph:.0f}px);opacity:1}}46%,100%{{transform:translateY({ph:.0f}px);opacity:0}}}}
</style>

<g clip-path="url(#frame)">
  <rect width="{W}" height="{H}" fill="#010a05"/>
  <g>{"".join(rain)}</g>
  <rect width="{W}" height="{H}" fill="url(#vig)"/>
  <ellipse cx="{W / 2}" cy="{H / 2}" rx="{pw * .62:.0f}" ry="{ph * .56:.0f}" fill="url(#halo)"/>

  <!-- the head: front-view reveal, then {FRAMES} pre-rendered angles that play in a loop -->
  <g class="head">
    <g class="intro">{intro}</g>
    {"".join(spin)}
  </g>
  <rect class="sweep" x="{PX - 10:.0f}" y="{PY:.0f}" width="{pw + 20:.0f}" height="14" fill="url(#scan)"/>
  <g>{"".join(sparks)}</g>

  <!-- terminals -->
  {panel(24, LY - 50, 276, lp_h, f"{USER}@matrix")}
  {"".join(ltxt)}
  {panel(RX0 - 16, rp_y, 270, rp_h, "skills.sh")}
  {"".join(rtxt)}
</g>
<rect x="1.5" y="1.5" width="{W - 3}" height="{H - 3}" rx="15" fill="none" stroke="#39ff14" stroke-width="2">
  <animate attributeName="stroke" values="#39ff14;#00e5ff;#ff8c00;#ffe600;#39ff14" dur="8s" repeatCount="indefinite"/>
</rect>
</svg>
'''

os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    f.write(svg)
print(f"Wrote {OUT}  ({len(svg) / 1024:.0f} KB, {COLS}x{ROWS} characters, {FRAMES} frames, "
      f"one turn = {T_SPIN:.1f}s)")
