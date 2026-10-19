"""Animated instruments for the current API: shared colors, persistent object identity."""

from __future__ import annotations

from explainer import *  # noqa: F403
from videos.ml.relflow.common import C, D, Mono, box, label, note


def heading(text):
    return label(text, font_size=40).to_edge(UP, buff=0.45)


def trace_tag(text="untrained API trace; relflow 0.3.2"):
    return note(text, font_size=19).to_edge(DOWN, buff=0.2)


def grid(lengths, width=None, cell=0.5):
    width = max(lengths) if width is None else width
    cells = VGroup()
    for row, n in enumerate(lengths):
        cells.add(
            VGroup(
                *[
                    RoundedRectangle(
                        width=cell,
                        height=cell,
                        corner_radius=0.05,
                        stroke_width=1,
                        stroke_color=C.RF_BRANCH if j < n else C.PADDED,
                        fill_color=C.RF_BRANCH if j < n else C.PADDED,
                        fill_opacity=0.85 if j < n else 0.2,
                    )
                    for j in range(width)
                ]
            ).arrange(RIGHT, buff=0.07)
        )
    cells.arrange(DOWN, aligned_edge=LEFT, buff=0.15)
    names = VGroup(
        *[
            Mono(f"row {i}: {n}", font_size=20, color=C.RF_BRANCH).next_to(r, LEFT, buff=0.25)
            for i, (r, n) in enumerate(zip(cells, lengths))
        ]
    )
    result = VGroup(cells, names)
    result.cells = cells
    return result


def dynamic_widths(scene):
    """Morph three real encoded batches around one unchanged parameter block."""
    d = D("current")
    assert d["parameters_unchanged"]
    head = heading("The schema stays. The batch changes.")
    model = box("same parameters", color=C.RF_ROOT, width=3.4, height=1.7, font_size=25).move_to(RIGHT * 3.6)
    code = Mono("trxns = rf.Branch(...)", font_size=25, color=C.RF_BRANCH).move_to(UP * 2.1)
    g = grid(d["geometry"][0]["lengths"]).move_to(LEFT * 2.7 + DOWN * 0.25)
    shape = Mono(str(d["geometry"][0]["shape"]), font_size=27).next_to(g, DOWN, buff=0.4)
    cap = label(r"width = largest retained collection in this batch", font_size=26, color=C.RF_BRANCH).move_to(
        DOWN * 2.3
    )
    tag = trace_tag("real model.encode traces; relflow 0.3.2; no parameter resize")
    arrow = Arrow(g.get_right(), model.get_left(), buff=0.2, color=C.RF_BRANCH)
    with scene.voiceover(
        "A schema fixes the kinds of fields and the model parameters. It no longer fixes every execution width. "
        "<bookmark mark='a'/> Leave branch length unspecified, and this batch uses two transaction slots. "
        "<bookmark mark='b'/> The next batch needs five. <bookmark mark='c'/> The next needs only one. "
        "These are shapes recorded from the real encoder. The parameter objects remain identical. "
        "<bookmark mark='d'/> A finite length is a cap: even a capped branch executes at the largest retained width, "
        "up to that limit. Shorter records still need padding within a batch. Automatic widths do not make attention free. Cap histories and request sizes before exposing a public service."
    ) as vo:
        scene.play(FadeIn(head), FadeIn(code), FadeIn(model), FadeIn(tag))
        vo.wait_until("a")
        scene.play(FadeIn(g), FadeIn(shape), GrowArrow(arrow), FadeIn(cap))
        for mark, batch in zip("bc", d["geometry"][1:]):
            vo.wait_until(mark)
            target = grid(batch["lengths"]).move_to(LEFT * 2.7 + DOWN * 0.25)
            s = Mono(str(batch["shape"]), font_size=27).next_to(target, DOWN, buff=0.4)
            scene.play(Transform(g, target), Transform(shape, s), run_time=1.5)
            scene.play(Indicate(model, color=C.RF_ROOT, scale_factor=1.04))
        vo.wait_until("d")
        finite = Mono('length=32, overflow="tail"', font_size=25, color=C.RF_TARGET).move_to(code)
        scene.play(Transform(code, finite))
    scene.clear_scene()
