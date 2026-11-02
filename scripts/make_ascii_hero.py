#!/usr/bin/env python3
"""
make_ascii_hero.py - turns your avatar into a ROTATING 3D character-art head
(@!$% glyphs, white/silver tone like a lit 3D scan) over neon code rain,
saved as assets/hero-ascii.svg

How it works
  1. Your avatar is cut out and "inflated" into a 3D bust: a front surface (face, nose,
     brows, hair and beard volume) and a back surface (back of the head, procedural hair).
  2. The bust is ray-marched from many angles around a vertical axis with a fixed light,
     so brightness changes as it turns, like a real 3D model.
  3. Every angle becomes one ASCII frame inside the SVG; CSS plays them in a loop,
     so the head spins on GitHub with no JavaScript.

Usage:
    pip install pillow numpy scipy
    python make_ascii_hero.py avatar.jpg

The sculpting numbers (FEATURES section) are tuned for the memoji avatar. If you use
another picture, the silhouette and shading still work; tweak FEATURES for the face.
"""
import html
import math
import os
import random
import sys
from multiprocessing import Pool

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

# ------------------------- EDIT THESE -------------------------
SRC = sys.argv[1] if len(sys.argv) > 1 else "avatar.jpg"
OUT = "assets/hero-ascii.svg"
USER = "edutanvir"

FRAMES = 60            # angles in one full turn (more = smoother but bigger file)
FRAME_SEC = 0.12       # seconds each frame is shown -> one turn = FRAMES * FRAME_SEC
SPIN_START = 4.4       # seconds of front view (reveal + pause) before the head starts to spin
HEAD_DEPTH = 0.50      # front-to-back thickness of the head relative to its width
PORTRAIT_COLOR = "#eef5f8"   # glyph color: white with a faint cool tint

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
COLS_FRONT = 96              # characters across the head seen from the front
CW, CH = 3.8, 6.3            # size of one character cell
SS = 2                       # sub-rays per cell side (anti-aliasing)
N = 256                      # working resolution of the 3D model
GROUPS = [".", ",'", ":;", "-~", "=+", "*!", "#%", "$&", "@"]   # sparse -> dense glyph groups
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

# work on a small square copy: all 3D maths happens at N x N
imr = im.resize((N, N), Image.LANCZOS)
ar = np.asarray(imr).astype(np.float32)
Lr0 = .299 * ar[..., 0] + .587 * ar[..., 1] + .114 * ar[..., 2]
Lr = np.clip(Lr0 + 1.0 * (Lr0 - ndi.gaussian_filter(Lr0, 1.0)), 0, 255)     # keep eyes/brows crisp
M = np.asarray(Image.fromarray((m * 255).astype(np.uint8)).resize((N, N), Image.BILINEAR)).astype(np.float32) / 255 > .5
ys_, xs_ = np.where(M)
X0, X1, Y0, Y1 = xs_.min(), xs_.max() + 1, ys_.min(), ys_.max() + 1
BWR, BHR = X1 - X0, Y1 - Y0
CX0 = (X0 + X1) / 2.0
KX, KY = N / w0, N / h0
yy, xx = np.mgrid[0:N, 0:N].astype(np.float32)


# ---------- 2. build the 3D bust (front surface + back surface) ----------
def blob(cx, cy, sx, sy, amp):
    """Smooth bump/dent. Coordinates are in ORIGINAL 1024px avatar pixels."""
    return amp * np.exp(-(((xx - cx * KX) / (sx * KX)) ** 2 + ((yy - cy * KY) / (sy * KY)) ** 2) / 2)


dt = ndi.distance_transform_edt(M).astype(np.float32)
fade = np.clip(dt / 9.0, 0, 1)                                    # features fade out toward the silhouette
dome_f = np.sqrt(1 - (1 - np.clip(dt / (.43 * BWR), 0, 1)) ** 2)  # rounded "pillow" profile, face side
dome_b = np.sqrt(1 - (1 - np.clip(dt / (.50 * BWR), 0, 1)) ** 2)  # rounder profile for the skull

# hair + beard are the dark areas (not eyes, brows or the inside of the mouth)
dark = Lr < 100
mouth_in = (((ar[..., 0] - ar[..., 1]) / np.maximum(ar[..., 0], 1)) > .3) & (Lr < 110)
mouth_in &= (yy > 640 * KY) & (yy < 790 * KY) & (xx > 400 * KX) & (xx < 630 * KX)
eyes_brows = (yy > 365 * KY) & (yy < 565 * KY) & (xx > 330 * KX) & (xx < 705 * KX)
hair_beard = dark & ~mouth_in & ~eyes_brows & M
vol = ndi.gaussian_filter(hair_beard.astype(np.float32), 2.2)

# ---- FEATURES (original avatar pixels; amp in model pixels, + = toward viewer) ----
feats = (
    blob(510, 585, 36, 70, +20)       # nose bridge + tip
    + blob(510, 622, 44, 28, +8)      # nose ball
    + blob(462, 618, 22, 18, +5) + blob(558, 618, 22, 18, +5)      # nostril wings
    + blob(405, 500, 70, 50, -6) + blob(618, 515, 70, 50, -6)       # eye sockets
    + blob(400, 395, 72, 16, +6) + blob(632, 415, 72, 16, +6)       # brow ridges
    + blob(345, 590, 62, 62, +6) + blob(675, 600, 62, 62, +6)       # cheeks
    + blob(510, 330, 170, 70, +4)                                    # forehead
    + blob(510, 712, 92, 38, -12)                                    # mouth dent
    + blob(510, 648, 120, 22, +9)                                    # mustache
)
AF = HEAD_DEPTH * .5 * BWR * 1.05
AB = HEAD_DEPTH * .5 * BWR * 1.10
ZF = AF * dome_f + fade * (feats + 15 * vol)
ZB = -AB * dome_b
ZF = ndi.gaussian_filter(ZF, 1.1) * M
ZB = ndi.gaussian_filter(ZB, 1.1) * M
GFY, GFX = np.gradient(ndi.gaussian_filter(ZF, .8))
GBY, GBX = np.gradient(ndi.gaussian_filter(ZB, .8))
GFX, GFY, GBX, GBY = [np.clip(g, -3, 3).astype(np.float32) for g in (GFX, GFY, GBX, GBY)]
# outside the head the "solid" is empty (negative thickness)
ZFm = np.where(M, ZF, -60).astype(np.float32)
ZBm = np.where(M, ZB, 60).astype(np.float32)

# surface brightness (albedo): your picture on the front, procedural hair strands on the back
ALB_F = np.clip(.05 + .95 * (np.clip(Lr, 0, 255) / 232.0) ** .85, 0, 1).astype(np.float32)
rng = np.random.RandomState(5)
strands = ndi.gaussian_filter(rng.rand(N, N), (7, .7), mode="wrap")
strands = (strands - strands.min()) / (strands.max() - strands.min())
ALB_B = (.18 + .34 * strands).astype(np.float32)

# lights (x right, y up, z toward viewer); the light stays still while the head turns
L1 = np.array([-.50, .55, .65]); L1 /= np.linalg.norm(L1)
L2 = np.array([.80, .10, .40]); L2 /= np.linalg.norm(L2)

# ---------- 3. character grid + rays ----------
PXC = BWR / COLS_FRONT                    # model pixels per character column
PXR = PXC * CH / CW                       # model pixels per character row
ROWS = int(round(BHR / PXR))
# widest extent of the head over a full turn decides how many columns we need
pts = np.argwhere(M)[::3]
px_, py_ = pts[:, 1] - CX0, pts[:, 0]
hw = 0.0
for ang in np.arange(0, 360, 7.5):
    c_, s_ = math.cos(math.radians(ang)), math.sin(math.radians(ang))
    for zz in (ZF[pts[:, 0], pts[:, 1]], ZB[pts[:, 0], pts[:, 1]]):
        hw = max(hw, np.abs(px_ * c_ + zz * s_).max())
COLS = int(math.ceil(2 * (hw + 2) / PXC))
COLS += COLS % 2
sub_u = ((np.arange(COLS * SS) + .5) / SS) * PXC - COLS * PXC / 2
sub_v = Y0 + ((np.arange(ROWS * SS) + .5) / SS) * PXR
UU, VV = np.meshgrid(sub_u, sub_v)
UU, VV = UU.ravel().astype(np.float32), VV.ravel().astype(np.float32)
NRAY = UU.size
TMAX = float(max(ZF.max(), -ZB.min())) + 4
TSTEP = 1.0
NOISE = np.random.RandomState(9).rand(ROWS, COLS).astype(np.float32)
VARI = np.random.RandomState(4).randint(0, 1000, (ROWS, COLS))


def inside(u, v, t, c, s):
    xm = CX0 + u * c - t * s
    zm = u * s + t * c
    co = np.array([v, xm])
    return (zm <= ndi.map_coordinates(ZFm, co, order=1, mode="nearest")) & \
           (zm >= ndi.map_coordinates(ZBm, co, order=1, mode="nearest"))


def render(theta):
    """Ray-march the bust turned by theta -> per-cell (coverage, brightness 0..1)."""
    c, s = math.cos(theta), math.sin(theta)
    t_hit = np.zeros(NRAY, np.float32)
    hit = np.zeros(NRAY, bool)
    alive = np.arange(NRAY)
    for t in np.arange(TMAX, -TMAX, -TSTEP):
        if alive.size == 0:
            break
        ins = inside(UU[alive], VV[alive], t, c, s)
        if ins.any():
            got = alive[ins]
            hit[got] = True
            t_hit[got] = t
            alive = alive[~ins]
    idx = np.where(hit)[0]
    u, v = UU[idx], VV[idx]
    hi_t = t_hit[idx]
    lo_t = hi_t + TSTEP
    for _ in range(5):                                         # refine surface position
        mid = (lo_t + hi_t) / 2
        ins = inside(u, v, mid, c, s)
        hi_t = np.where(ins, mid, hi_t)
        lo_t = np.where(ins, lo_t, mid)
    t = hi_t
    xm = CX0 + u * c - t * s
    zm = u * s + t * c
    co = np.array([v, xm])
    zf = ndi.map_coordinates(ZFm, co, order=1, mode="nearest")
    zb = ndi.map_coordinates(ZBm, co, order=1, mode="nearest")
    front = (zf - zm) < (zm - zb)
    smp = lambda arr: ndi.map_coordinates(arr, co, order=1, mode="nearest")
    nx = np.where(front, -smp(GFX), smp(GBX))
    ny = -np.where(front, -smp(GFY), smp(GBY))                 # image y points down -> flip to y-up
    nz = np.where(front, 1.0, -1.0)
    nrm = np.sqrt(nx * nx + ny * ny + nz * nz)
    nx, ny, nz = nx / nrm, ny / nrm, nz / nrm
    nxw, nzw = nx * c + nz * s, -nx * s + nz * c               # rotate normal into view space
    lam1 = np.clip(nxw * L1[0] + ny * L1[1] + nzw * L1[2], 0, 1)
    lam2 = np.clip(nxw * L2[0] + ny * L2[1] + nzw * L2[2], 0, 1)
    rim = (1 - np.clip(nzw, 0, 1)) ** 3
    alb = np.where(front, smp(ALB_F), smp(ALB_B))
    val = np.clip(alb * (.14 + .90 * lam1 + .30 * lam2 + .20 * rim), 0, 1)
    full_v = np.zeros(NRAY, np.float32)
    full_h = np.zeros(NRAY, np.float32)
    full_v[idx], full_h[idx] = val, 1
    cell = lambda arr: arr.reshape(ROWS, SS, COLS, SS).sum(axis=(1, 3))
    hits = cell(full_h)
    return hits / (SS * SS), cell(full_v) / np.maximum(hits, 1)


def render_k(k):
    return render(2 * math.pi * k / FRAMES)


if __name__ == "__main__":
    print(f"head model {BWR}x{BHR}px, grid {COLS}x{ROWS} chars; ray-marching {FRAMES} frames ...")
    with Pool(max(1, min(os.cpu_count() or 1, 4))) as pool:
        frames = pool.map(render_k, range(FRAMES))

    pw, ph = COLS * CW, ROWS * CH
    PX, PY = (W - pw) / 2, (H - ph) / 2

    def glyph(i, j, d):
        d = min(.999, max(0.0, d ** .85 + (NOISE[i, j] - .5) * .08))
        grp = GROUPS[int(d * len(GROUPS))]
        return grp[VARI[i, j] % len(grp)]

    def build_rows(cov, val):
        """-> list of (backing path piece, <text> markup) per character row"""
        inc = cov > .5
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
                s = "".join(glyph(i, jj, val[i, jj]) for jj in range(j, k))
                txt += f'<text x="{x:.1f}" y="{y + CH - 1.6:.1f}" textLength="{n * CW:.1f}">{esc(s)}</text>'
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
    for (si, sj) in random.sample(stable, min(40, len(stable))):
        ch = random.choice("01@$%!#")
        col = random.choice(["#ffffff", "#ffffff", "#00e5ff", "#ff8c00", "#ffe600"])
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
      .head text{{font-size:{CH}px;font-weight:700;fill:{PORTRAIT_COLOR}}}
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
