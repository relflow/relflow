"""Helpers shared by the relflow scenes: text, schema-tree cards, token tiles, code, plots, data."""

from __future__ import annotations

import functools
import json
import math
from pathlib import Path

import numpy as np

from explainer import *  # noqa: F403

DATA_DIR = Path(__file__).resolve().parent / "data"
ASSETS = Path(__file__).resolve().parent / "assets"
MONO = "DejaVu Sans Mono"
SANS = "DejaVu Sans"
BRAND_FONT = "Libertinus Serif"  # relflow's wordmark face (SIL OFL, assets/OFL.txt)
BRAND_INK = "#dce5ce"  # relflow's "Forest" palette: lettering and branches
BRAND_COLUMNS = "#465f4b"

try:
    import manimpango

    manimpango.register_font(str(ASSETS / "LibertinusSerif-Semibold.otf"))
except Exception:  # pragma: no cover - falls back to the default serif
    pass


def logo(height=2.0) -> SVGMobject:
    """relflow's logo (docs/assets/branding/logos/logo.dark.svg in the relflow repository)."""
    return SVGMobject(str(ASSETS / "relflow_logo.svg")).set(height=height)


def wordmark(font_size=110, color=BRAND_INK) -> Text:
    return Text("relflow", font=BRAND_FONT, weight=SEMIBOLD, font_size=font_size, color=color)


@functools.cache
def D(name: str) -> dict:
    """A committed result of ``python -m videos.relflow.compute`` (real relflow runs)."""
    return json.loads((DATA_DIR / f"{name}.json").read_text())


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------
WIDE_TEX = TEX_TEMPLATE.copy()
WIDE_TEX.add_to_preamble(r"\setlength{\textwidth}{40cm}")
MAX_TEXT_WIDTH = 13.6


def label(text: str, color=WHITE, font_size=32, **kw) -> Tex:
    """A Tex label that never wraps on its own and shrinks to fit the frame."""
    kw.setdefault("tex_template", WIDE_TEX)
    t = Tex(text, color=color, font_size=font_size, **kw)
    if t.width > MAX_TEXT_WIDTH:
        t.width = MAX_TEXT_WIDTH
    return t


def mtex(tex: str, color=WHITE, font_size=36, **kw) -> MathTex:
    return MathTex(tex, color=color, font_size=font_size, **kw)


def note(text: str, font_size=22, color=GREY_B) -> Tex:
    return label(text, font_size=font_size, color=color)


def tagged(text: str, color=WHITE, font_size=28, opacity=0.85) -> Tex:
    t = label(text, color=color, font_size=font_size)
    t.add_background_rectangle(color=BACKGROUND, opacity=opacity, buff=0.06)
    return t


def title(text: str, font_size=40, color=WHITE) -> Tex:
    return label(text, font_size=font_size, color=color).to_edge(UP, buff=0.4)


def real_tag(text: str = r"real relflow 0.2.3, run on the computer that rendered this video", corner=DR) -> Tex:
    return note(text, color=GREY_A, font_size=20).to_corner(corner, buff=0.25)


def schematic_tag(corner=DR) -> Tex:
    return note(r"schematic", font_size=20).to_corner(corner, buff=0.25)


def proof_tag(text: str, corner=DR) -> Tex:
    return note(r"recorded in relflow's proof catalog: " + text, color=GREY_A, font_size=20).to_corner(corner, buff=0.25)


class Mono(VGroup):
    """Monospace text whose bounding box always includes ascenders and descenders, so rows line up."""

    def __init__(self, s: str, font_size=24, color=WHITE, weight=NORMAL, **kw):
        strut = "Ég"
        t = Text(strut + s, font=MONO, font_size=font_size, color=color, weight=weight, **kw)
        struts, glyphs = t[:2], t[2:]
        if len(glyphs):
            x0, x1 = glyphs.get_left()[0], glyphs.get_right()[0]
        else:
            x0 = struts.get_right()[0]
            x1 = x0 + 0.2
        frame = VGroup(VectorizedPoint([x0, struts.get_top()[1], 0]), VectorizedPoint([x1, struts.get_bottom()[1], 0]))
        super().__init__(frame, *glyphs)
        self.frame = frame
        self.glyphs = VGroup(*glyphs)


def chip(text: str, color=GREY_B, font_size=22, fill=0.18, mono=False, text_color=WHITE) -> VGroup:
    t = Mono(text, font_size=font_size, color=text_color) if mono else label(text, font_size=font_size, color=text_color)
    box = RoundedRectangle(width=t.width + 0.28, height=t.height + 0.18, corner_radius=0.08, stroke_color=color,
                           stroke_width=1.5, fill_color=color, fill_opacity=fill)
    t.move_to(box)
    g = VGroup(box, t)
    g.box, g.text = box, t
    return g


def box(text: str, color=GREY_B, width=None, height=0.8, font_size=26, fill=0.15, corner=0.12) -> VGroup:
    t = label(text, font_size=font_size)
    w = width or t.width + 0.5
    r = RoundedRectangle(width=w, height=max(height, t.height + 0.3), corner_radius=corner, stroke_color=color,
                         stroke_width=2.5, fill_color=color, fill_opacity=fill)
    t.move_to(r)
    if t.width > w - 0.2:
        t.width = w - 0.2
    g = VGroup(r, t)
    g.box, g.text = r, t
    return g


def fmt_int(n) -> str:
    return f"{int(round(n)):,}".replace(",", "{,}")


def episode_card(number: int, text: str) -> VGroup:
    series = VGroup(wordmark(font_size=40, color=GREY_A), label(r"visualized", font_size=30, color=C.DIM))
    series.arrange(RIGHT, buff=0.18, aligned_edge=DOWN)
    num = label(rf"Episode {number}", font_size=30, color=C.DIM)
    t = label(text, font_size=56)
    return VGroup(series, num, t).arrange(DOWN, buff=0.28)


def show_episode_card(scene: Scene, hold: float = 1.4):
    """The scene's episode number and title, from the manifest (each scene is one episode of the series)."""
    from videos.relflow.manifest import episode

    e = episode(type(scene).__name__)
    card = episode_card(e["number"], e["title"])
    scene.play(FadeIn(card, shift=UP * 0.2), run_time=0.8)
    scene.wait(hold)
    scene.play(FadeOut(card, shift=UP * 0.2), run_time=0.6)


# ---------------------------------------------------------------------------
# Schema-tree cards (after relflow's own Typst diagrams: root, branch, field, target)
# ---------------------------------------------------------------------------
KIND_COLOR = {"root": C.RF_ROOT, "branch": C.RF_BRANCH, "field": C.RF_FIELD, "target": C.RF_TARGET}
TYPE_COLOR = {"Number": C.NUMBER, "Category": C.CATEGORY, "Hash": C.HASH, "Boolean": C.BOOLEAN, "Branch": C.RF_BRANCH,
              "Root": C.RF_ROOT}


def tint(color, amount=0.82):
    return interpolate_color(ManimColor(color), ManimColor(BACKGROUND), amount)


class Card(VGroup):
    """A schema node: small upper-case type over a monospace name, colored by its role."""

    def __init__(self, name: str, kind: str = "field", type: str | None = None, detail: str | None = None,
                 width: float | None = None, font_size=22, repeated: bool = False):
        super().__init__()
        color = KIND_COLOR[kind]
        type = type or ("Branch" if kind in ("root", "branch") else "Field")
        cap = Text(type.upper(), font=SANS, weight=BOLD, font_size=font_size * 0.55, color=color)
        nm = Mono(name, font_size=font_size, color=color, weight=BOLD)
        rows = [cap, nm]
        if detail:
            rows.append(Text(detail, font=SANS, font_size=font_size * 0.6, color=interpolate_color(ManimColor(color), WHITE, 0.3)))
        body = VGroup(*rows).arrange(DOWN, aligned_edge=LEFT, buff=0.07)
        w = max(width or 0, body.width + 0.34)
        rect = RoundedRectangle(width=w, height=body.height + 0.26, corner_radius=0.09, stroke_color=color,
                                stroke_width=1.6, stroke_opacity=0.8, fill_color=tint(color), fill_opacity=1)
        body.move_to(rect).align_to(rect, LEFT).shift(RIGHT * 0.17)
        if repeated:  # a second card peeking out behind: a repeated collection
            back = rect.copy().set_stroke(opacity=0.45).set_fill(opacity=1).shift(RIGHT * 0.07 + DOWN * 0.07)
            self.add(back)
        self.add(rect, body)
        # not self.color: VGroup.color is a property that would recolor every submobject
        self.rect, self.cap, self.name_text, self.role_color, self.kind = rect, cap, nm, color, kind


class Tree(VGroup):
    """Lay out a schema left to right, like relflow's diagrams.  ``spec`` is
    (name, kind, type, children) with optional detail as a fifth item.
    ``self.cards`` maps the slash address of each node to its card."""

    def __init__(self, spec, x_gap=0.55, y_gap=0.16, font_size=22, root_address="/"):
        super().__init__()
        self.cards: dict[str, Card] = {}
        self.edges = VGroup()
        self.edge_to: dict[str, VMobject] = {}
        self.font_size = font_size
        self.x_gap, self.y_gap = x_gap, y_gap
        root = self._build(spec, root_address)
        self._place(root, 0.0, 0.0)
        self._connect(root)
        self.add(self.edges, *[n["card"] for n in self._all(root)])
        self.root = root
        self.center()

    def _build(self, spec, address):
        name, kind, type_, children = spec[:4]
        detail = spec[4] if len(spec) > 4 else None
        card = Card(name, kind, type_, detail=detail, font_size=self.font_size, repeated=(kind == "branch"))
        self.cards[address] = card
        kids = [self._build(c, (address.rstrip("/") + "/" + c[0])) for c in children]
        h = sum(k["h"] for k in kids) + self.y_gap * max(0, len(kids) - 1)
        return {"card": card, "kids": kids, "h": max(card.height, h), "address": address}

    def _place(self, node, x, top):
        card = node["card"]
        if node["kids"]:
            y = top
            for k in node["kids"]:
                self._place(k, x + card.width + self.x_gap, y)
                y -= k["h"] + self.y_gap
            ys = [k["card"].get_center()[1] for k in node["kids"]]
            cy = (max(ys) + min(ys)) / 2
        else:
            cy = top - node["h"] / 2
        card.move_to([x + card.width / 2, cy, 0])

    def _connect(self, node):
        if not node["kids"]:
            return
        p = node["card"].rect.get_right()
        xb = p[0] + self.x_gap / 2
        for k in node["kids"]:
            q = k["card"].rect.get_left()
            e = VMobject().set_points_as_corners([p, [xb, p[1], 0], [xb, q[1], 0], q])
            e.set_stroke(GREY_C, 1.6).set_fill(opacity=0)
            self.edges.add(e)
            self.edge_to[k["address"]] = e
            self._connect(k)

    def _all(self, node):
        yield node
        for k in node["kids"]:
            yield from self._all(k)


# ---------------------------------------------------------------------------
# Tokens and value states
# ---------------------------------------------------------------------------
STATES = ("valued", "null", "padded", "masked", "other")
STATE_COLOR = {0: C.VALUED, 1: C.NULL_STATE, 2: C.PADDED, 3: C.MASKED, 4: TEAL}
STATE_LETTER = {0: "V", 1: "N", 2: "P", 3: "M", 4: "O"}


def tile(color, size=0.36, letter: str | None = None, fill=0.85, font_size=None, stroke=None, text_color=None) -> VGroup:
    sq = RoundedRectangle(width=size, height=size, corner_radius=size * 0.16, stroke_width=1.2 if stroke else 0,
                          stroke_color=stroke or color, fill_color=color, fill_opacity=fill)
    g = VGroup(sq)
    g.sq = sq
    if letter:
        t = Text(letter, font=SANS, weight=BOLD, font_size=font_size or size * 46, color=text_color or BACKGROUND)
        t.move_to(sq)
        g.add(t)
        g.letter = t
    return g


def state_tile(state: int, size=0.36, letters=True) -> VGroup:
    col = STATE_COLOR[state]
    fill = 0.35 if state == 2 else 0.9
    txt = GREY_B if state == 2 else BACKGROUND
    return tile(col, size, STATE_LETTER[state] if letters else None, fill=fill, text_color=txt)


def token_row(colors, size=0.3, gap=0.05, fill=0.85) -> VGroup:
    return VGroup(*[tile(c, size, fill=fill) for c in colors]).arrange(RIGHT, buff=gap)


def vector_strip(values, color=C.EMBED, cell=0.18, gap=0.03, vmax=None, direction=DOWN, outline=True) -> VGroup:
    """A column (or row) of cells whose brightness follows |value|; negative values are dimmer."""
    values = np.asarray(values, dtype=float)
    vmax = vmax or (np.abs(values).max() + 1e-9)
    cells = VGroup()
    for v in values:
        a = min(1.0, abs(v) / vmax)
        col = color if v >= 0 else interpolate_color(ManimColor(color), ManimColor(BACKGROUND), 0.45)
        cells.add(Square(cell, stroke_width=0, fill_color=col, fill_opacity=0.1 + 0.9 * a))
    cells.arrange(direction, buff=gap)
    g = VGroup(cells)
    if outline:
        g.add(SurroundingRectangle(cells, buff=0.04, color=color, stroke_width=1.4, corner_radius=0.04))
    g.cells = cells
    return g


def heat_grid(values, color, cell=0.3, gap=0.03, vmax=None, vmin=0.0, low=None) -> VGroup:
    """2-D array -> grid of squares (rows top to bottom), brightness = value."""
    values = np.asarray(values, dtype=float)
    vmax = values.max() if vmax is None else vmax
    rows = VGroup()
    for r in values:
        row = VGroup()
        for v in r:
            a = 0.0 if vmax <= vmin else min(1.0, max(0.0, (v - vmin) / (vmax - vmin)))
            row.add(Square(cell, stroke_width=0, fill_color=color if low is None else interpolate_color(ManimColor(low), ManimColor(color), a),
                           fill_opacity=(0.06 + 0.94 * a) if low is None else 1))
        row.arrange(RIGHT, buff=gap)
        rows.add(row)
    rows.arrange(DOWN, buff=gap)
    return rows


# ---------------------------------------------------------------------------
# Code
# ---------------------------------------------------------------------------
def code(src: str, font_size=22, style="one-dark", background=True) -> Code:
    c = Code(code_string=src.strip("\n"), language="python", formatter_style=style,
             background="rectangle" if background else "rectangle", add_line_numbers=False,
             paragraph_config={"font": MONO, "font_size": font_size, "line_spacing": 0.75},
             background_config={"stroke_color": GREY_D, "stroke_width": 1.2, "fill_color": "#161B24",
                                "fill_opacity": 1, "corner_radius": 0.12, "buff": 0.25})
    if c.width > MAX_TEXT_WIDTH - 0.2:  # long lines: shrink to the frame rather than run off it
        c.width = MAX_TEXT_WIDTH - 0.2
    return c


def code_line(c: Code, i: int) -> VMobject:
    return c.code_lines[i]


def find_text(c: Code, line: int, start: int, end: int) -> VGroup:
    """Glyphs [start, end) of a code line, counting non-space characters."""
    return VGroup(*c.code_lines[line][start:end])


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
class Plot(VGroup):
    """Axes along the bottom and left of a box, linear or log in each direction, light gridlines.
    Built centered on the origin; move it freely, c2p() follows."""

    def __init__(self, x_range, y_range, width=10.0, height=5.0, log_x=False, log_y=False, x_ticks=(), y_ticks=(),
                 x_fmt=None, y_fmt=None, font_size=24, grid=True, x_grid=False, color=GREY_B, tick_color=GREY_A):
        super().__init__()
        self.x_range, self.y_range, self.w, self.h = x_range, y_range, width, height
        self.log_x, self.log_y = log_x, log_y
        x_fmt = x_fmt or (lambda v: f"{v:g}")
        y_fmt = y_fmt or (lambda v: f"{v:g}")
        self.x_axis = Line(ORIGIN, RIGHT * width, color=color, stroke_width=2)
        self.y_axis = Line(ORIGIN, UP * height, color=color, stroke_width=2)
        self.grid = VGroup()
        if grid:
            for t in y_ticks:
                if t != y_range[0]:
                    self.grid.add(Line(self.c2p(x_range[0], t), self.c2p(x_range[1], t), color=GREY_D, stroke_width=1,
                                       stroke_opacity=0.7))
        if x_grid:
            for t in x_ticks:
                if t != x_range[0]:
                    self.grid.add(Line(self.c2p(t, y_range[0]), self.c2p(t, y_range[1]), color=GREY_D, stroke_width=1,
                                       stroke_opacity=0.5))
        self.x_labels, self.y_labels = VGroup(), VGroup()
        for t in x_ticks:
            p = self.c2p(t, y_range[0])
            self.x_labels.add(Line(p, p + DOWN * 0.08, color=color, stroke_width=2))
            lab = x_fmt(t)
            m = lab if isinstance(lab, Mobject) else MathTex(lab, font_size=font_size, color=tick_color)
            self.x_labels.add(m.next_to(p, DOWN, buff=0.14))
        for t in y_ticks:
            p = self.c2p(x_range[0], t)
            lab = y_fmt(t)
            m = lab if isinstance(lab, Mobject) else MathTex(lab, font_size=font_size, color=tick_color)
            self.y_labels.add(m.next_to(p, LEFT, buff=0.14))
        self.add(self.grid, self.x_axis, self.y_axis, self.x_labels, self.y_labels)
        self.shift(-(self.x_axis.get_start() + np.array([width / 2, height / 2, 0])))

    def _f(self, v, rng, log):
        lo, hi = rng
        if log:
            v = np.maximum(v, lo * 1e-6) if isinstance(v, np.ndarray) else max(v, lo * 1e-6)
            return (np.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))
        return (v - lo) / (hi - lo)

    def c2p(self, x, y):
        o = self.x_axis.get_start()
        return o + RIGHT * self.w * self._f(x, self.x_range, self.log_x) + UP * self.h * self._f(y, self.y_range, self.log_y)

    def line(self, xs, ys, color, stroke_width=4, clip=True) -> VMobject:
        xs, ys = np.asarray(xs, float), np.asarray(ys, float)
        m = np.isfinite(xs) & np.isfinite(ys)
        if clip:
            lo, hi = self.y_range
            ys = np.clip(ys, lo, hi)
        pts = [self.c2p(a, b) for a, b in zip(xs[m], ys[m])]
        return VMobject().set_points_smoothly(pts).set_stroke(color, stroke_width).set_fill(opacity=0) if len(pts) < 400 \
            else VMobject().set_points_as_corners(pts).set_stroke(color, stroke_width).set_fill(opacity=0)

    def bar(self, x, y, width, color, opacity=0.85, base=None) -> Rectangle:
        base = self.y_range[0] if base is None else base
        p0, p1 = self.c2p(x, base), self.c2p(x, y)
        w = abs(self.c2p(x + width / 2, base)[0] - self.c2p(x - width / 2, base)[0])
        r = Rectangle(width=w, height=max(abs(p1[1] - p0[1]), 1e-3), stroke_width=0, fill_color=color,
                      fill_opacity=opacity)
        return r.move_to((p0 + p1) / 2)

    def hline(self, y, color=GREY_B, stroke_width=2, dashed=True) -> VMobject:
        a, b = self.c2p(self.x_range[0], y), self.c2p(self.x_range[1], y)
        return DashedLine(a, b, color=color, stroke_width=stroke_width, dash_length=0.08) if dashed else \
            Line(a, b, color=color, stroke_width=stroke_width)

    def vline(self, x, color=GREY_B, stroke_width=2, dashed=True) -> VMobject:
        a, b = self.c2p(x, self.y_range[0]), self.c2p(x, self.y_range[1])
        return DashedLine(a, b, color=color, stroke_width=stroke_width, dash_length=0.08) if dashed else \
            Line(a, b, color=color, stroke_width=stroke_width)

    def dot(self, x, y, color=WHITE, radius=0.05, opacity=1.0) -> Dot:
        return Dot(self.c2p(x, y), radius=radius, color=color, fill_opacity=opacity)

    def x_title(self, text, color=GREY_A, font_size=26) -> Tex:
        return label(text, color=color, font_size=font_size).next_to(self.x_labels, DOWN, buff=0.15)

    def y_title(self, text, color=GREY_A, font_size=26) -> Tex:
        return label(text, color=color, font_size=font_size).next_to(self.y_axis, UP, buff=0.2).align_to(self.y_axis, LEFT)


def flow_arrow(a, b, color=GREY_B, stroke_width=3, buff=0.1, tip=0.18) -> Arrow:
    return Arrow(a, b, buff=buff, color=color, stroke_width=stroke_width, max_tip_length_to_length_ratio=0.5,
                 tip_length=tip)
