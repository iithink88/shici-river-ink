#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inkflow.py —— 诗句沿山河流动 · 水墨字景合成引擎

原理：
  1. 从水墨底图计算结构张量方向场（沿等照度线 = 顺着景物笔触/水流/山脊的走势）
  2. 在画面右侧纹理密集区撒种子点，双向追踪生成流线
  3. 沿每条流线按弧长逐字排布，字号渐变、墨色分层、占位防重叠
  4. 叠加左侧竖排版式（标题 / 作者 / 印章 / 诗句原文）
  5. 纸纹 + 暗角 + 柔化，输出 3:4 成品

用法：
  python inkflow.py --base base.png --title "春残" --author "翁宏" \
      --poem "落花人独立，微雨燕双飞。" --out out.png
"""

import argparse
import math
import os
import random
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# ---------------------------------------------------------------- 常量

INK = (44, 42, 39)
PAPER = (246, 243, 233)
SEAL_RED = (176, 58, 46)

# ---- 字体：跨平台查找，可用环境变量覆盖
#   SHICI_FONT_SONG  指定宋体（标题/大字）
#   SHICI_FONT_KAI   指定楷体（诗句）
#   SHICI_FONT_DIR   指定字体目录，按文件名关键字搜索
_FONT_CANDIDATES = {
    "song": [
        r"C:\Windows\Fonts\simsun.ttc",
        r"C:\Windows\Fonts\Songti.ttc",
        "/System/Library/Fonts/Supplemental/Songti.ttc",
        "/System/Library/Fonts/Songti.ttc",
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSerifCJKsc-Regular.otf",
        "/usr/share/fonts/truetype/arphic/uming.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    ],
    "kai": [
        r"C:\Windows\Fonts\simkai.ttf",
        r"C:\Windows\Fonts\STKAITI.TTF",
        "/System/Library/Fonts/Supplemental/Kaiti.ttc",
        "/System/Library/Fonts/Supplemental/STKaiti.ttf",
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
        "/usr/share/fonts/truetype/arphic/ukai.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    ],
    "fang": [
        r"C:\Windows\Fonts\simfang.ttf",
        "/System/Library/Fonts/Supplemental/Songti.ttc",
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    ],
}

_FONT_KEYWORD = {
    "song": ["simsun", "songti", "notoserifcjk", "uming", "sourcehanserif"],
    "kai": ["simkai", "kaiti", "stkaiti", "ukai", "notoserifcjk", "sourcehanserif"],
    "fang": ["simfang", "fangsong", "songti", "notoserifcjk"],
}


def _resolve_font(kind):
    env = os.environ.get("SHICI_FONT_" + kind.upper())
    if env and os.path.exists(env):
        return env
    for p in _FONT_CANDIDATES[kind]:
        if os.path.exists(p):
            return p
    fdir = os.environ.get("SHICI_FONT_DIR")
    dirs = [fdir] if fdir else []
    dirs += [
        r"C:\Windows\Fonts",
        "/System/Library/Fonts",
        "/System/Library/Fonts/Supplemental",
        "/Library/Fonts",
        os.path.expanduser("~/.fonts"),
        os.path.expanduser("~/.local/share/fonts"),
        "/usr/share/fonts",
    ]
    kws = _FONT_KEYWORD[kind]
    for d in dirs:
        if not d or not os.path.isdir(d):
            continue
        for root, _dirs, files in os.walk(d):
            for f in files:
                low = f.lower()
                if not low.endswith((".ttf", ".ttc", ".otf")):
                    continue
                if any(k in low.replace(" ", "") for k in kws):
                    return os.path.join(root, f)
    return None


FONT_SONG = _resolve_font("song")
FONT_KAI = _resolve_font("kai") or FONT_SONG
FONT_FANG = _resolve_font("fang") or FONT_SONG

if FONT_SONG is None:
    print("警告：未找到中文字体，请设置环境变量 SHICI_FONT_SONG 指向任意中文 TTF/TTC 字体。",
          file=sys.stderr)

_FONT_CACHE = {}


def get_font(path, size):
    size = max(6, int(round(size)))
    key = (path, size)
    f = _FONT_CACHE.get(key)
    if f is None:
        try:
            f = ImageFont.truetype(path, size)
        except Exception:
            try:
                f = ImageFont.truetype(FONT_SONG, size)
            except Exception:
                f = ImageFont.load_default(size)
        _FONT_CACHE[key] = f
    return f


# ---------------------------------------------------------------- 高斯 / 方向场


def _gauss(a, sigma):
    """不依赖 scipy/cv2 的高斯模糊（三次箱式近似）。"""
    if sigma <= 0.5:
        return a.astype(np.float32)
    a = a.astype(np.float32)
    r = max(1, int(round(sigma * 1.6)))
    for _ in range(3):
        k = 2 * r + 1
        pad = np.pad(a, ((0, 0), (r, r)), mode="edge")
        cs = np.cumsum(pad, axis=1)
        cs = np.concatenate([np.zeros((cs.shape[0], 1), np.float32), cs], axis=1)
        a = (cs[:, k:] - cs[:, :-k]) / k
        pad = np.pad(a, ((r, r), (0, 0)), mode="edge")
        cs = np.cumsum(pad, axis=0)
        cs = np.concatenate([np.zeros((1, cs.shape[1]), np.float32), cs], axis=0)
        a = (cs[k:, :] - cs[:-k, :]) / k
    return a.astype(np.float32)


def build_field(gray, sigma=9.0):
    """结构张量 → 主梯度方向 → 切向单位向量场 + 纹理强度。"""
    g = _gauss(gray.astype(np.float32), 1.2)
    gx = np.zeros_like(g)
    gy = np.zeros_like(g)
    gx[:, 1:-1] = (g[:, 2:] - g[:, :-2]) * 0.5
    gy[1:-1, :] = (g[2:, :] - g[:-2, :]) * 0.5

    Jxx = _gauss(gx * gx, sigma)
    Jyy = _gauss(gy * gy, sigma)
    Jxy = _gauss(gx * gy, sigma)

    theta = 0.5 * np.arctan2(2.0 * Jxy, Jxx - Jyy)   # 主梯度方向
    tang = theta + math.pi / 2.0                       # 沿笔触的切向

    tx = np.cos(tang).astype(np.float32)
    ty = np.sin(tang).astype(np.float32)
    mag = np.sqrt(np.maximum(Jxx + Jyy, 0)).astype(np.float32)
    if mag.max() > 1e-6:
        mag = mag / (np.percentile(mag, 99.0) + 1e-6)
    return tx, ty, np.clip(mag, 0, 1)


def density_field(W, H, rng, cells=7, lo=0.35, hi=1.0):
    """低频斑驳场：制造大块的浓淡呼吸，像水墨晕染。"""
    small = rng.random((cells, max(2, int(cells * H / W)))).astype(np.float32)
    img = Image.fromarray((small * 255).astype(np.uint8)).resize((W, H), Image.BICUBIC)
    a = np.asarray(img, np.float32) / 255.0
    a = np.clip((a - a.min()) / (a.max() - a.min() + 1e-6), 0, 1)
    a = a ** 1.4
    return (lo + (hi - lo) * a).astype(np.float32)


def blend_flow(tx, ty, gx, gy, w=0.62):
    """局部切向与全局流向混合：既顺景物，又整体往下流淌。"""
    nx = tx * w + gx * (1.0 - w)
    ny = ty * w + gy * (1.0 - w)
    n = np.sqrt(nx * nx + ny * ny) + 1e-6
    return (nx / n).astype(np.float32), (ny / n).astype(np.float32)


def sample_vec(tx, ty, x, y):
    h, w = tx.shape
    if x < 1.0 or y < 1.0 or x > w - 2.0 or y > h - 2.0:
        return None
    x0 = int(x)
    y0 = int(y)
    fx = x - x0
    fy = y - y0
    w00 = (1 - fx) * (1 - fy)
    w10 = fx * (1 - fy)
    w01 = (1 - fx) * fy
    w11 = fx * fy
    a = tx[y0, x0] * w00 + tx[y0, x0 + 1] * w10 + tx[y0 + 1, x0] * w01 + tx[y0 + 1, x0 + 1] * w11
    b = ty[y0, x0] * w00 + ty[y0, x0 + 1] * w10 + ty[y0 + 1, x0] * w01 + ty[y0 + 1, x0 + 1] * w11
    n = math.hypot(a, b)
    if n < 1e-6:
        return None
    return (a / n, b / n)


def trace(seed, tx, ty, mag, step, max_steps, mag_min, W, H, max_turn=3.4):
    """从种子点双向追踪，返回点序列（正向 + 反向拼接）。"""
    back, fwd = [], []
    for sgn, sink in ((1, fwd), (-1, back)):
        p = [float(seed[0]), float(seed[1])]
        prev = None
        turn = 0.0
        for _ in range(max_steps):
            a = sample_vec(tx, ty, p[0], p[1])
            if a is None:
                break
            if prev is None:
                a = (a[0] * sgn, a[1] * sgn)
            elif a[0] * prev[0] + a[1] * prev[1] < 0:
                a = (-a[0], -a[1])
            if prev is not None:
                cross = prev[0] * a[1] - prev[1] * a[0]
                dot = max(-1.0, min(1.0, prev[0] * a[0] + prev[1] * a[1]))
                turn += math.atan2(cross, dot)
                if abs(turn) > max_turn:
                    break
            nx = p[0] + a[0] * step
            ny = p[1] + a[1] * step
            if nx < 2 or ny < 2 or nx > W - 3 or ny > H - 3:
                break
            mx = int(nx)
            my = int(ny)
            if mag[my, mx] < mag_min:
                break
            p = [nx, ny]
            prev = a
            sink.append(p)
    return list(reversed(back)) + fwd


def smooth_path(pts, k=3):
    if len(pts) < k * 2 + 1:
        return pts
    P = np.asarray(pts, np.float32)
    ker = np.ones(k * 2 + 1, np.float32) / (k * 2 + 1)
    out = P.copy()
    out[:, 0] = np.convolve(P[:, 0], ker, mode="same")
    out[:, 1] = np.convolve(P[:, 1], ker, mode="same")
    out[:k] = P[:k]
    out[-k:] = P[-k:]
    return [tuple(v) for v in out]


def path_length(pts):
    if len(pts) < 2:
        return 0.0
    P = np.asarray(pts, np.float32)
    return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())


def resample(pts, ds=1.0):
    """按弧长等距重采样，返回 (Nx2 数组, 总长)。"""
    if len(pts) < 2:
        return np.asarray(pts, np.float32), 0.0
    P = np.asarray(pts, np.float32)
    d = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    L = float(s[-1])
    if L <= 0:
        return P, 0.0
    n = max(2, int(L / ds) + 1)
    t = np.linspace(0, L, n)
    xs = np.interp(t, s, P[:, 0])
    ys = np.interp(t, s, P[:, 1])
    return np.stack([xs, ys], 1), L


# ---------------------------------------------------------------- 种子点


def make_seeds(mag, W, H, x_min, density, rng):
    """在右侧纹理密集区按网格撒点，返回 [(x, y)]，已按纹理强度加权。"""
    cell = 34
    cands = []
    for gy in range(0, H, cell):
        for gx in range(0, W, cell):
            if gx < x_min:
                continue
            x = gx + rng.randint(4, cell - 4)
            y = gy + rng.randint(4, cell - 4)
            if not (6 < x < W - 6 and 6 < y < H - 6):
                continue
            m = float(mag[y, x])
            cands.append((x, y, m))
    if not cands:
        return []
    ms = np.array([c[2] for c in cands])
    thr = np.percentile(ms, 30)
    cands = [c for c in cands if c[2] >= thr]
    rng.shuffle(cands)
    n = max(6, int(len(cands) * density))
    return [(c[0], c[1]) for c in cands[:n]]


def build_spines(mag, tx, ty, W, H, x_min, count, rng):
    """在强纹理处挑选彼此远离的种子，追踪出若干条长骨架线。"""
    # 分区取样：把文字区切成网格，每格取纹理最强点，保证骨架铺得开
    nx, ny = 5, 6
    x0, x1 = W * 0.26, W * 0.90
    y0, y1 = H * 0.06, H * 0.92
    cands = []
    for j in range(ny):
        for i in range(nx):
            bx0 = x0 + (x1 - x0) * i / nx
            bx1 = x0 + (x1 - x0) * (i + 1) / nx
            by0 = y0 + (y1 - y0) * j / ny
            by1 = y0 + (y1 - y0) * (j + 1) / ny
            best = None
            for _ in range(26):
                xx = rng.uniform(bx0, bx1)
                yy = rng.uniform(by0, by1)
                xi, yi = int(xx), int(yy)
                if not (0 <= xi < W and 0 <= yi < H):
                    continue
                v = float(mag[yi, xi]) + rng.uniform(0, 0.02)
                if best is None or v > best[2]:
                    best = (xx, yy, v)
            if best:
                cands.append(best)
    if not cands:
        return []
    cands.sort(key=lambda c: -c[2])

    min_d = W * 0.050
    chosen = []
    for (x, y, m) in cands:
        if all((x - a) ** 2 + (y - b) ** 2 > min_d * min_d for a, b in chosen):
            chosen.append((x, y))
        if len(chosen) >= count:
            break

    spines = []
    for (x, y) in chosen:
        pts = trace((x, y), tx, ty, mag, step=6.0, max_steps=560, mag_min=0.0004, W=W, H=H)
        if len(pts) < 24:
            continue
        pts = smooth_path(pts, 5)
        arr, Ln = resample(pts, 1.0)
        if Ln < 260:
            continue
        _i = np.linspace(0, len(arr) - 1, 16).astype(int)
        _m = float(np.mean([float(mag[int(arr[k][1]), int(arr[k][0])]) for k in _i]))
        spines.append((arr, Ln, _m))
    return spines


def offset_path(arr, off, taper_mix=0.62):
    """沿法向平行偏移生成子线，两端自动收窄（河流的锥度）。"""
    if abs(off) < 1e-6:
        return arr
    P = np.asarray(arr, np.float32)
    d = np.gradient(P, axis=0)
    # 平滑切向，避免局部抖动放大到法向
    k = 15
    ker = np.ones(k, np.float32) / k
    dx = np.convolve(d[:, 0], ker, mode="same")
    dy = np.convolve(d[:, 1], ker, mode="same")
    nrm = np.sqrt(dx * dx + dy * dy) + 1e-6
    nx = -dy / nrm
    ny = dx / nrm
    # 法向再平滑一次
    nx = np.convolve(nx, ker, mode="same")
    ny = np.convolve(ny, ker, mode="same")
    nrm = np.sqrt(nx * nx + ny * ny) + 1e-6
    nx /= nrm
    ny /= nrm

    u = np.linspace(0.0, 1.0, len(P))
    taper = (0.42 + 0.58 * np.sin(np.pi * np.clip(0.10 + 0.84 * u, 0, 1)) ** 0.8)
    taper = taper * (1.0 - taper_mix) + taper_mix * taper   # 保持形状，仅做强度混合
    scale = taper_mix + (1.0 - taper_mix) * taper
    s = off * scale
    out = np.stack([P[:, 0] + nx * s, P[:, 1] + ny * s], 1)
    out[:6] = P[:6] + np.stack([nx[:6] * off, ny[:6] * off], 1)
    out[-6:] = P[-6:] + np.stack([nx[-6:] * off, ny[-6:] * off], 1)
    return out


def longest_inside(pts, W, H, margin=4):
    """取落在画面内的最长连续段（避免越界被裁剪后堆在边缘）。"""
    P = np.asarray(pts, np.float32)
    ok = ((P[:, 0] > margin) & (P[:, 0] < W - margin) &
          (P[:, 1] > margin) & (P[:, 1] < H - margin))
    if not ok.any():
        return None
    # 找最长连续 True
    best_s = best_e = -1
    s0 = None
    for i, v in enumerate(ok):
        if v and s0 is None:
            s0 = i
        elif not v and s0 is not None:
            if i - s0 > best_e - best_s:
                best_s, best_e = s0, i
            s0 = None
    if s0 is not None and len(ok) - s0 > best_e - best_s:
        best_s, best_e = s0, len(ok)
    if best_e - best_s < 4:
        return None
    seg = P[best_s:best_e]
    d = float(np.linalg.norm(np.diff(seg, axis=0), axis=1).sum()) if len(seg) > 1 else 0.0
    return seg, d


# ---------------------------------------------------------------- 占位网格


class Occupancy:
    def __init__(self, W, H, cell=5):
        self.cell = cell
        self.gw = W // cell + 2
        self.gh = H // cell + 2
        self.g = np.zeros((self.gh, self.gw), bool)

    def free(self, cx, cy, r):
        c = self.cell
        x0 = max(0, int((cx - r) / c))
        x1 = min(self.gw, int((cx + r) / c) + 1)
        y0 = max(0, int((cy - r) / c))
        y1 = min(self.gh, int((cy + r) / c) + 1)
        if x0 >= x1 or y0 >= y1:
            return True
        return not self.g[y0:y1, x0:x1].any()

    def mark(self, cx, cy, r):
        c = self.cell
        x0 = max(0, int((cx - r) / c))
        x1 = min(self.gw, int((cx + r) / c) + 1)
        y0 = max(0, int((cy - r) / c))
        y1 = min(self.gh, int((cy + r) / c) + 1)
        if x0 < x1 and y0 < y1:
            self.g[y0:y1, x0:x1] = True


# ---------------------------------------------------------------- 生成字形指令


def glyphs_on_path(pts, L, cycle, start_idx, s_fn, a_fn, spacing):
    """沿等距路径按字宽推进，产出 (x, y, size, alpha, ch)。"""
    if L < 8:
        return
    out = []
    pos = 0.0
    i = start_idx
    n = len(cycle)
    guard = 0
    while pos < L and guard < 4000:
        guard += 1
        u = pos / L
        s = s_fn(u)
        s = max(4.0, s)
        idx = min(len(pts) - 1, int(pos))
        x, y = pts[idx]
        out.append((float(x), float(y), s, a_fn(u), cycle[i % n]))
        i += 1
        pos += s * spacing
    return out


# ---------------------------------------------------------------- 版式


def draw_vertical(dr, text, x, y, font, gap, fill):
    """竖排文字（含标点微调），返回结束 y。"""
    s = font.size
    for ch in text:
        if ch in "，。、？！；：":
            dr.text((x, y + s * 0.04), ch, font=font, fill=fill, anchor="mm")
        else:
            dr.text((x, y), ch, font=font, fill=fill, anchor="mm")
        y += s * gap
    return y


def draw_seal(img, x, y, size, text, rng):
    """朱文方印：红底白字。"""
    size = max(12, int(round(size)))
    x = int(round(x))
    y = int(round(y))
    pad = max(2, size // 8)
    layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.rectangle([0, 0, size - 1, size - 1], fill=SEAL_RED + (232,))
    d.rectangle([0, 0, size - 1, size - 1], outline=(255, 255, 255, 40), width=1)
    inner = size - pad * 2
    if text:
        n = len(text)
        fs = int(inner / max(1, n) * 0.96)
        f = get_font(FONT_SONG, fs)
        # 竖排双字
        step = inner / max(1, n)
        for i, ch in enumerate(text):
            cy = pad + step * (i + 0.5)
            d.text((size / 2, cy), ch, font=f, fill=(252, 249, 242, 235), anchor="mm")
    # 做旧：边缘轻微残缺
    px = layer.load()
    for _ in range(int(size * size * 0.012)):
        xx = rng.randint(0, size - 1)
        yy = rng.randint(0, size - 1)
        r, g, b, a = px[xx, yy]
        px[xx, yy] = (r, g, b, int(a * rng.uniform(0.2, 0.7)))
    layer = layer.filter(ImageFilter.GaussianBlur(0.4))
    img.alpha_composite(layer, (int(x), int(y)))


def wrap_columns(text, max_per_col):
    """把长句切分成若干竖列（按字数和标点）。"""
    if len(text) <= max_per_col:
        return [text]
    cols = []
    cur = ""
    for ch in text:
        cur += ch
        if len(cur) >= max_per_col and (ch in "，。、？！；：" or len(cur) >= max_per_col + 2):
            cols.append(cur)
            cur = ""
    if cur:
        cols.append(cur)
    return cols


def draw_layout(img, title, author, poem_lines, rng, opts):
    W, H = img.size
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    dr = ImageDraw.Draw(layer)

    title_size = opts["title_size"]
    author_size = opts["author_size"]
    poem_size = opts["poem_size"]

    title_x = W * opts["title_x"]
    col_gap = W * opts["col_gap"]

    f_title = get_font(FONT_SONG, title_size)
    f_author = get_font(FONT_SONG, author_size)
    f_poem = get_font(FONT_KAI, poem_size)

    ink = INK + (238,)

    # 标题
    y0 = H * opts["top_y"]
    draw_vertical(dr, title, title_x, y0 + title_size * 0.5, f_title, 1.12, ink)

    # 作者 + 印章
    ax = title_x + col_gap * 0.95
    ay = y0 + author_size * 0.55
    end_y = draw_vertical(dr, author, ax, ay, f_author, 1.10, INK + (215,))
    draw_seal(layer, ax - author_size * 0.62, end_y + author_size * 0.18,
              author_size * 1.45, (author or "")[:2] if author else "诗", rng)
    dr = ImageDraw.Draw(layer)

    # 诗句：从右往左排列（第 1 句在最右）
    max_per_col = int((H - y0 - H * 0.06) / (poem_size * 1.16))
    cols = []
    for line in poem_lines:
        cols.extend(wrap_columns(line, max_per_col))
    base_x = title_x + col_gap * 2.0
    for i, col in enumerate(cols):
        cx = base_x + (len(cols) - 1 - i) * col_gap
        draw_vertical(dr, col, cx, y0 + poem_size * 0.52, f_poem, 1.16, INK + (232,))

    img.alpha_composite(layer)


# ---------------------------------------------------------------- 纸纹


def paper_grain(W, H, rng, strength=1.0):
    """低频斑驳 + 高频颗粒，返回 L 模式的叠加层（255 中性）。"""
    small = rng.normal(128, 26, (max(8, H // 40), max(8, W // 40))).astype(np.float32)
    lo = Image.fromarray(np.clip(small, 0, 255).astype(np.uint8)).resize((W, H), Image.BICUBIC)
    lo = lo.filter(ImageFilter.GaussianBlur(W / 90.0))
    hi = rng.normal(128, 9, (H, W)).astype(np.float32)
    a = np.asarray(lo, np.float32) * 0.75 + hi * 0.25
    a = 128 + (a - 128) * 0.55 * strength
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def vignette(W, H):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cx, cy = W / 2.0, H / 2.0
    r = np.sqrt(((xx - cx) / (W * 0.72)) ** 2 + ((yy - cy) / (H * 0.72)) ** 2)
    v = np.clip(1.0 - 0.16 * np.clip(r - 0.55, 0, 3) ** 1.4, 0, 1)
    return np.dstack([v, v, v])


# ---------------------------------------------------------------- 主流程


def compose(args):
    rng = random.Random(args.seed)
    nprng = np.random.default_rng(args.seed)

    base = Image.open(args.base).convert("RGB")

    # 目标画幅（3:4），ss 为超采样倍数：渲染尺寸 = 逻辑宽度 x ss
    W = int(round(args.width * args.ss))
    H = int(round(W * 4 / 3))

    # 等比铺满后居中裁切
    br = base.width / base.height
    tr = W / H
    if br > tr:
        nh = H
        nw = int(round(H * br))
    else:
        nw = W
        nh = int(round(W / br))
    base = base.resize((nw, nh), Image.LANCZOS)
    left = (nw - W) // 2
    top = (nh - H) // 2
    base = base.crop((left, top, left + W, top + H))

    # 擦掉右下角水印
    if args.clean_corner:
        cw = int(W * 0.20)
        ch = int(H * 0.075)
        patch = base.crop((W - cw - 10, H - ch - 10, W - 10, H - 10))
        patch = patch.resize((cw + 40, ch + 40), Image.LANCZOS).filter(ImageFilter.GaussianBlur(6))
        mask = Image.new("L", (cw + 40, ch + 40), 0)
        ImageDraw.Draw(mask).ellipse([6, 6, cw + 34, ch + 34], fill=255)
        mask = mask.filter(ImageFilter.GaussianBlur(8))
        tint = Image.new("RGB", patch.size, PAPER)
        base.paste(Image.blend(patch, tint, 0.45), (W - cw - 10, H - ch - 10), mask)

    # 提亮一点，给墨字让位
    base = Image.blend(base, Image.new("RGB", base.size, PAPER), 0.22)

    gray = np.asarray(base.convert("L"), np.float32) / 255.0
    lum = gray

    dfld = density_field(W, H, nprng, cells=args.blotch)
    tx, ty, mag = build_field(gray, sigma=args.field_sigma)
    gv = np.array([args.flow_x, 1.0], np.float32)
    gv /= np.linalg.norm(gv)
    # 切向恒在下半平面（sin>=0），与全局流向同向，混合不会打架
    tx, ty = blend_flow(tx, ty, gv[0], gv[1], args.flow_bias)

    # ---- 骨架 + 平行字带
    layers = args.layers
    x_min = int(W * args.text_x_min)
    spines = build_spines(mag, tx, ty, W, H, x_min, args.spine_count, rng)
    spines.sort(key=lambda t: -t[2])
    _ls = sorted([int(l) for _, l, _m in spines], reverse=True)
    print(f"  骨架流线: {len(spines)} 条, 长度={[int(v) for v in _ls]}")

    all_glyphs = []

    for li, L in enumerate(layers):
        _before = len(all_glyphs)
        occ = Occupancy(W, H, cell=5)      # 每层独立，保证同层不叠、跨层可穿插
        band = L["band"] * W
        step = L["step"] * W / max(0.25, args.density)
        offs = []
        _p = -band
        while _p <= band:
            offs.append(_p)
            _p += step * rng.uniform(0.62, 1.45)
        use = spines[:3] if li == 0 else spines      # 大字只落在纹理最强的主字流上
        for (arr, Ln, _mm) in use:
            for off in offs:
                if rng.random() < L.get("skip", 0.0):
                    continue                       # 随机留出透气通道
                jitter = rng.uniform(-0.16, 0.16) * step
                path = offset_path(arr, off + jitter)
                parr, _ = resample([tuple(p) for p in path], 1.0)
                seg = longest_inside(parr, W, H)
                if seg is None:
                    continue
                parr, Lp = seg
                if Lp < L["min_len"]:
                    continue

                edge = abs(off) / max(1.0, band)
                # 整列共用一个纹理强度 → 列内字号稳定，列间形成主次
                _idx = np.linspace(0, len(parr) - 1, 14).astype(int)
                _m = [float(mag[int(parr[i][1]), int(parr[i][0])]) for i in _idx]
                mean_tex = min(1.0, float(np.mean(_m)) * 1.9)

                s_hi = (L["size"][0] * W * (1.0 - 0.22 * edge)
                        * (0.80 + 0.34 * mean_tex) * rng.uniform(0.86, 1.18))
                s_lo = s_hi * rng.uniform(0.90, 1.00)

                a_hi = (L["alpha"][0] * (1.0 - 0.20 * edge)
                        * (0.46 + 0.95 * mean_tex) * rng.uniform(0.92, 1.04))
                a_lo = a_hi * rng.uniform(0.82, 0.98)
                phase = rng.uniform(0, math.tau)
                wave = rng.uniform(0.04, 0.12)

                def s_fn(u, s_hi=s_hi, s_lo=s_lo):
                    return s_hi + (s_lo - s_hi) * (u ** 0.7)

                def a_fn(u, a_hi=a_hi, a_lo=a_lo, phase=phase, wave=wave):
                    b = a_hi + (a_lo - a_hi) * (u ** 0.8)
                    return b * (1.0 + wave * math.sin(phase + u * 5.0))

                gl = glyphs_on_path(parr, Lp, args.cycle, rng.randrange(len(args.cycle)),
                                    s_fn, a_fn, L["spacing"] * rng.uniform(1.0, 1.14))
                if args.debug:
                    print(f"      L{li} off={off:+.0f} Lp={Lp:.0f} glyphs={len(gl)}")
                for (x, y, s, a, ch) in gl:
                    if x < x_min or y < 4 or y > H - 4 or x > W - 4:
                        continue
                    if ch.strip() == "":
                        continue
                    iy = int(y)
                    ix = int(x)
                    magv = float(mag[iy, ix])
                    lv = float(lum[iy, ix])
                    # 纹理越强，字越大越浓；越靠画面下方（近景）越大
                    if L.get("bg_only") and magv > 0.35:
                        continue
                    if li == 2 and magv > 0.42 and rng.random() < 0.75:
                        continue
                    tex = min(1.0, magv * 1.9)
                    dv = float(dfld[iy, ix])
                    s2 = s * (0.97 + 0.07 * tex) * (0.95 + 0.10 * (y / H)) * (0.88 + 0.16 * dv)
                    a2 = a * (0.55 + 0.90 * tex) * (0.62 + 0.70 * lv) * dv
                    a2 = max(0.0, min(1.0, a2))
                    if a2 < 0.025:
                        continue
                    r = s2 * (0.40 if li == 0 else (0.37 if li == 1 else 0.34))
                    if not occ.free(x, y, r):
                        continue
                    occ.mark(x, y, r)
                    all_glyphs.append((x, y, s2, a2, ch))
        print(f"    layer{li}: +{len(all_glyphs) - _before}")

    print(f"  字形数量: {len(all_glyphs)}")

    # ---- 绘制：按墨色分桶批量合成
    NB = 14
    masks = [Image.new("L", (W, H), 0) for _ in range(NB)]
    draws = [ImageDraw.Draw(m) for m in masks]
    for (x, y, s, a, ch) in sorted(all_glyphs, key=lambda g: g[3]):
        b = min(NB - 1, max(0, int(a * NB)))
        f = get_font(args.body_font, s)
        try:
            draws[b].text((x, y), ch, font=f, fill=255, anchor="mm")
        except Exception:
            pass

    out = base.convert("RGBA")
    for b in range(NB):
        a = (b + 0.6) / NB
        m = masks[b].point(lambda v, a=a: int(v * a))
        layer = Image.new("RGBA", (W, H), INK + (0,))
        layer.putalpha(m)
        out.alpha_composite(layer)

    # ---- 版式
    draw_layout(out, args.title, args.author, args.lines, rng, {
        "title_size": W * args.title_scale,
        "author_size": W * args.author_scale,
        "poem_size": W * args.poem_scale,
        "title_x": args.title_x,
        "col_gap": args.col_gap,
        "top_y": args.top_y,
    })

    # ---- 纸纹 + 暗角
    grain = paper_grain(W, H, nprng, strength=0.5)
    grgb = Image.merge("RGB", (grain, grain, grain))
    out = Image.blend(out.convert("RGB"), grgb, 0.055).convert("RGBA")
    out = Image.blend(out, Image.new("RGBA", (W, H), PAPER + (255,)), 0.055)

    va = vignette(W, H)
    arr = np.asarray(out.convert("RGB"), np.float32) * va
    out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    out.save(args.out, quality=95)
    print(f"  已输出: {args.out}  ({W}x{H})")
    return args.out


# ---------------------------------------------------------------- CLI


LAYER_PRESETS = {
    # 大字层：贴着骨架的主字链，字大墨浓，是视觉焦点
    "big": dict(band=0.069, step=0.066, size=(0.0575, 0.0), alpha=(0.96, 0.72),
                spacing=1.05, min_len=240, skip=0.04),
    # 中字层：主体编织面
    "mid": dict(band=0.135, step=0.0440, size=(0.0245, 0.0), alpha=(0.70, 0.44),
                spacing=1.08, min_len=110, skip=0.10),
    # 远景雾：铺满留白处的极淡小字，制造空气感
    "veil": dict(band=0.200, step=0.0265, size=(0.0098, 0.0), alpha=(0.15, 0.06),
                 spacing=1.24, min_len=40, skip=0.30, bg_only=True),
    # 雾字层：外围碎字，极淡，填补缝隙
    "mist": dict(band=0.170, step=0.0285, size=(0.0138, 0.0), alpha=(0.38, 0.17),
                 spacing=1.14, min_len=50, skip=0.16),
}


PRESETS = {
    # 瀑布、山涧：字随水垂落，略偏右下
    "waterfall": dict(flow_x=0.22, flow_bias=0.42, field_sigma=16, blotch=7),
    # 湖面、江河：涟漪横向铺开，弱化重力
    "water":     dict(flow_x=0.80, flow_bias=0.20, field_sigma=13, blotch=9),
    # 星河、夜空：斜向倾泻
    "star":      dict(flow_x=-0.38, flow_bias=0.24, field_sigma=18, blotch=6),
    # 庭院、花木、雨檐：多方向垂落
    "garden":    dict(flow_x=0.15, flow_bias=0.38, field_sigma=14, blotch=8),
    # 山路、峰峦：顺山脊盘绕
    "mountain":  dict(flow_x=0.48, flow_bias=0.46, field_sigma=18, blotch=6),
}


def split_poem(text):
    """把整首诗拆成竖列显示用的行。"""
    text = text.replace("\n", " ").strip()
    if " " in text:
        parts = [p for p in text.split(" ") if p]
    else:
        parts = [text]
    return parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="水墨底图路径")
    ap.add_argument("--out", required=True, help="输出 PNG 路径")
    ap.add_argument("--title", default="", help="诗题（竖排大字）")
    ap.add_argument("--author", default="", help="作者（小字 + 印章）")
    ap.add_argument("--poem", default="", help="诗句全文，多句用全角/半角标点或空格分隔")
    ap.add_argument("--lines", default=None, help="显式指定竖排行，用 | 分隔")
    ap.add_argument("--width", type=int, default=1152, help="逻辑宽度")
    ap.add_argument("--preset", default=None, choices=list(PRESETS.keys()),
                    help="题材预设，会覆盖流向/平滑/斑驳参数")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--ss", type=float, default=2.0, help="超采样倍数，2=高清，1=快速")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--density", type=float, default=1.0, help="字列疏密倍数（越大越密）")
    ap.add_argument("--spine-count", type=int, default=22, help="骨架流线条数")
    ap.add_argument("--blotch", type=int, default=7, help="墨色斑驳尺度，越大块越大")
    ap.add_argument("--flow-bias", type=float, default=0.40, help="全局流向权重：1=完全跟随景物，0=完全顺流")
    ap.add_argument("--flow-x", type=float, default=0.22, help="全局流向的横向分量（负=向左下流）")
    ap.add_argument("--field-sigma", type=float, default=16.0, help="方向场平滑尺度，大=更整体的走势")
    ap.add_argument("--text-x-min", type=float, default=0.225, help="文字流左边界比例")
    ap.add_argument("--title-scale", type=float, default=0.040)
    ap.add_argument("--author-scale", type=float, default=0.020)
    ap.add_argument("--poem-scale", type=float, default=0.0255)
    ap.add_argument("--title-x", type=float, default=0.108)
    ap.add_argument("--col-gap", type=float, default=0.058)
    ap.add_argument("--top-y", type=float, default=0.055)
    ap.add_argument("--clean-corner", action="store_true", default=True)
    ap.add_argument("--no-clean-corner", dest="clean_corner", action="store_false")
    ap.add_argument("--body-font", default=FONT_SONG)
    ap.add_argument("--layers", default="big,mid,mist,veil")
    args = ap.parse_args()

    if args.preset:
        for k, v in PRESETS[args.preset].items():
            setattr(args, k, v)

    args.layers = [LAYER_PRESETS[k.strip()] for k in args.layers.split(",") if k.strip()]

    if args.lines:
        args.lines = [x for x in args.lines.split("|") if x]
        args.cycle = " ".join(args.lines).replace(" ", "")
    else:
        raw = args.poem.strip()
        args.lines = split_poem(raw) if raw else [args.title]
        args.cycle = " ".join(args.lines).replace(" ", "")

    if not args.cycle:
        print("错误：需要 --poem 或 --lines", file=sys.stderr)
        sys.exit(1)

    compose(args)


if __name__ == "__main__":
    main()
