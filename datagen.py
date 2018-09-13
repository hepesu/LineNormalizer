'''
Modified from https://github.com/byungsook/vectornet/blob/master/data_line.py

Semantic Segmentation for Line Drawing Vectorization Using Neural Networks
Tensorflow implementation of Semantic Segmentation for Line Drawing Vectorization Using Neural Networks.

Byungsoo Kim1, Oliver Wang2, Cengiz ?ztireli1, Markus Gross1

1ETH Zurich, 2Adobe Research

Computer Graphics Forum (Proceedings of Eurographics 2018)

'''

import numpy as np
import cv2
from PIL import Image
import cairosvg
import io

SEED = 1
WIDTH, HEIGHT = 128, 128

MIN_STROKE_WIDTH, MAX_STROKE_WIDTH = 0.2, 2
MAX_STROKE_COLOR = 50

NORM_STROKE_WIDTH = 0.5

MAX_NUM_STROKES = 10

SVG_START_TEMPLATE = """<?xml version="1.0" encoding="utf-8"?><!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
<svg width="{w}" height="{h}" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg" version="1.1">
<rect x="0" y="0" width="{w}" height="{h}" fill="rgb(255,255,255)"/>
<g fill="none" transform="rotate({rot})">\n"""

SVG_RECT_TEMPLATE = """<rect x="{x}" y="{y}" width="{w}" height="{h}" stroke="rgb({r},{g},{b})" stroke-width="{sw}" fill="rgba(255,255,255,0)"/>"""

SVG_ELLIPSE_TEMPLATE = """<ellipse cx="{x}" cy="{y}" rx="{rx}" ry="{ry}" stroke="rgb({r},{g},{b})" stroke-width="{sw}" fill="rgba(255,255,255,0)"/>"""

SVG_LINE_TEMPLATE = """<path id="{id}" d="M {x1} {y1} L{x2} {y2}" stroke="rgb({r},{g},{b})" stroke-width="{sw}"/>"""

SVG_CUBIC_BEZIER_TEMPLATE = """<path id="{id}" d="M {sx} {sy} C {cx1} {cy1} {cx2} {cy2} {tx} {ty}" stroke="rgb({r},{g},{b})" stroke-width="{sw}"/>"""

SVG_END_TEMPLATE = """</g>\n</svg>"""


def draw_line(id, w, h, rng):
    stroke_color = rng.randint(MAX_STROKE_COLOR)
    stroke_width = rng.rand() * (MAX_STROKE_WIDTH - MIN_STROKE_WIDTH) + MIN_STROKE_WIDTH
    x = rng.randint(w, size=2)
    y = rng.randint(h, size=2)

    return SVG_LINE_TEMPLATE.format(
        id=id,
        x1=x[0], y1=y[0],
        x2=x[1], y2=y[1],
        r=stroke_color, g=stroke_color, b=stroke_color,
        sw=stroke_width
    )


def draw_cubic_bezier_curve(id, w, h, rng):
    stroke_color = rng.randint(MAX_STROKE_COLOR)
    stroke_width = rng.rand() * (MAX_STROKE_WIDTH - MIN_STROKE_WIDTH) + MIN_STROKE_WIDTH
    x = rng.randint(w, size=4)
    y = rng.randint(h, size=4)

    return SVG_CUBIC_BEZIER_TEMPLATE.format(
        id=id,
        sx=x[0], sy=y[0],
        cx1=x[1], cy1=y[1],
        cx2=x[2], cy2=y[2],
        tx=x[3], ty=y[3],
        r=stroke_color, g=stroke_color, b=stroke_color,
        sw=stroke_width
    )


def draw_rect(id, w, h, rng):
    stroke_color = rng.randint(MAX_STROKE_COLOR)
    stroke_width = rng.rand() * (MAX_STROKE_WIDTH - MIN_STROKE_WIDTH) + MIN_STROKE_WIDTH
    x = rng.randint(w)
    y = rng.randint(h)
    w = rng.randint(low=w // 4, high=w // 2)
    h = rng.randint(low=h // 4, high=h // 2)

    return SVG_RECT_TEMPLATE.format(
        id=id,
        x=x, y=y,
        w=w, h=h,
        r=stroke_color, g=stroke_color, b=stroke_color,
        sw=stroke_width
    )


def draw_ellipse(id, w, h, rng):
    stroke_color = rng.randint(MAX_STROKE_COLOR)
    stroke_width = rng.rand() * (MAX_STROKE_WIDTH - MIN_STROKE_WIDTH) + MIN_STROKE_WIDTH
    x = rng.randint(w)
    y = rng.randint(h)
    rx = rng.randint(low=w // 4, high=w // 2)
    ry = rng.randint(low=h // 4, high=h // 2)

    return SVG_ELLIPSE_TEMPLATE.format(
        id=id,
        x=x, y=y,
        rx=rx, ry=ry,
        r=stroke_color, g=stroke_color, b=stroke_color,
        sw=stroke_width
    )


def draw_path(id, w, h, rng):
    path_selector = {
        0: draw_line,
        1: draw_cubic_bezier_curve,
        2: draw_rect,
        3: draw_ellipse
    }

    stroke_type = rng.randint(len(path_selector))

    return path_selector[stroke_type](id, w, h, rng)


def gen_data(rng, batch_size):
    x = []
    y = []

    norm_stroke_width_txt = """stroke-width="{sw}" _stroke-width""".format(sw=NORM_STROKE_WIDTH)
    for file_id in range(batch_size):
        while True:
            svg = SVG_START_TEMPLATE.format(
                w=WIDTH,
                h=HEIGHT,
                rot=rng.randint(0, 180)
            )
            svgpre = SVG_START_TEMPLATE

            for i in range(rng.randint(MAX_NUM_STROKES) + 1):
                path = draw_path(
                    id=i,
                    w=WIDTH,
                    h=HEIGHT,
                    rng=rng
                )
                svg += path + '\n'
                svgpre += path + '\n'

            svg += SVG_END_TEMPLATE

            x_png = cairosvg.svg2png(bytestring=svg.encode('utf-8'))
            x_img = Image.open(io.BytesIO(x_png))
            x_arr = np.array(x_img, np.float)

            # with open('data/s.svg', 'w') as f:
            #     f.write(svg.replace('stroke-width', norm_stroke_width_txt))

            y_png = cairosvg.svg2png(bytestring=svg.replace('stroke-width', norm_stroke_width_txt).encode('utf-8'))
            y_img = Image.open(io.BytesIO(y_png))
            y_arr = np.array(y_img, np.float)

            if np.mean(x_arr) < 200 or np.mean(x_arr) > 245:
                continue
            else:
                x.append(np.reshape(x_arr[:, :, 0], (HEIGHT, WIDTH, 1)))
                y.append(np.reshape(y_arr[:, :, 0], (HEIGHT, WIDTH, 1)))
                break

    return np.array(x) / 255.0, np.array(y) / 255.0


def test():
    rnd = np.random.RandomState(SEED)

    for i in range(5):
        x_data, y_data = gen_data(rnd, 4)
        for j in range(4):
            cv2.imwrite('data/x_%d_%d.png' % (i, j), x_data[j] * 255)
            cv2.imwrite('data/y_%d_%d.png' % (i, j), y_data[j] * 255)


if __name__ == "__main__":
    test()
