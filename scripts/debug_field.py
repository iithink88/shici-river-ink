#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""方向场可视化调试：输出「底图 | 切向短线 + 纹理强度」，用来判断字流会往哪走。

用法：
    python debug_field.py 底图.png 输出.png [方向场平滑尺度]

图右侧每个小线段代表该处文字的走向，线段越长越深代表该处纹理越强——
字流会沿着这些线段走。调 --field-sigma 时可反复对照。
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inkflow import build_field  # noqa: E402

if len(sys.argv) < 3:
    print(__doc__)
    sys.exit(1)

src = sys.argv[1]
out = sys.argv[2]
sigma = float(sys.argv[3]) if len(sys.argv) > 3 else 16.0

im = Image.open(src).convert("RGB")
W, H = im.size
gray = np.asarray(im.convert("L"), np.float32) / 255.0
tx, ty, mag = build_field(gray, sigma=sigma)

pad = 24
canvas = Image.new("RGB", (W * 2 + pad, H), (250, 248, 242))
canvas.paste(im, (0, 0))
right = im.copy().convert("RGB")
right = Image.blend(right, Image.new("RGB", right.size, (255, 255, 255)), 0.55)
dr = ImageDraw.Draw(right)

step = max(10, W // 64)
for y in range(step, H - step, step):
    for x in range(step, W - step, step):
        a = (tx[y, x], ty[y, x])
        m = mag[y, x]
        ln = 6 + 10 * min(1.0, m * 3)
        x2 = x + a[0] * ln
        y2 = y + a[1] * ln
        v = int(150 - 110 * min(1.0, m * 2))
        dr.line([(x, y), (x2, y2)], fill=(v, v, v), width=1)
        if m > 0.02:
            dr.ellipse([x2 - 1, y2 - 1, x2 + 1, y2 + 1], fill=(180, 60, 40))

canvas.paste(right, (W + pad, 0))
canvas.save(out)
print("saved", out, canvas.size)
