"""Native vector instruments: direction, token identity, and measured attention."""

# ruff: noqa: F405
from __future__ import annotations

import numpy as np
from explainer import *  # noqa: F403
from videos.ml.relflow.common import C, D, Mono, label, vector_strip
from videos.ml.relflow.current_visuals import heading, trace_tag


def transmit(scene, sources, target, color=C.RF_BRANCH, weights=None, run_time=1.0):
    """Copy information toward a receiver; optional weights control the packet areas."""
    sources = list(sources)
    weights = np.ones(len(sources)) if weights is None else np.asarray(weights)
    packets, animations = [], []
    for source, weight in zip(sources, weights):
        if weight < 1e-5:
            continue
        radius = 0.035 + 0.13 * np.sqrt(float(weight))
        packet = Dot(source.get_center(), radius=radius, color=color)
        path = Line(source.get_center(), target.get_center())
        packets.append(packet)
        animations.append(MoveAlongPath(packet, path, rate_func=linear))
    if not animations:
        return
    scene.add(*packets)
    scene.play(LaggedStart(*animations, lag_ratio=0.045), run_time=run_time)
    scene.remove(*packets)


def stream(scene, paths, color=C.RF_ROOT, reverse=False, run_time=1.0):
    """Travel on the actual drawn route, including the bends of a schema tree."""
    paths = list(paths)
    packets, animations = [], []
    for path in paths:
        route = path.copy()
        if reverse:
            route.reverse_points()
        packet = Dot(route.get_start(), radius=0.065, color=color)
        packets.append(packet)
        animations.append(
            AnimationGroup(
                ShowPassingFlash(route.copy().set_stroke(color, 4), time_width=0.35),
                MoveAlongPath(packet, route, rate_func=linear),
            )
        )
    scene.add(*packets)
    scene.play(LaggedStart(*animations, lag_ratio=0.15), run_time=run_time)
    scene.remove(*packets)


def tree_up(scene, tree, run_time=1.8):
    """Messages aggregate by depth; targets do not enter the encoder."""
    paths = [(a, p) for a, p in tree.edge_to.items() if tree.cards[a].kind != "target"]
    depths = sorted({a.count("/") for a, p in paths}, reverse=True)
    for depth in depths:
        stream(
            scene,
            [p for a, p in paths if a.count("/") == depth],
            reverse=True,
            color=C.RF_BRANCH if depth > 1 else C.RF_ROOT,
            run_time=run_time / len(depths),
        )


def attention_read(scene):
    """Actual Q/K → normalized weights → weighted V for one trained reducer head."""
    d = D("flow_current")
    t = d["traces"][0]["reduction"]
    weights = np.asarray(t["weights"])[0, 0, 0]
    logits = np.asarray(t["q"])[0, 0, 0] @ np.asarray(t["k"])[0, 0].T / 4
    values = np.asarray(t["v"])[0, 0]
    context = np.asarray(t["context"])[0, 0, 0]
    assert np.allclose(weights @ values, context, atol=1e-5)
    head = heading("Attention chooses a mixture of value vectors.")
    tokens = (
        VGroup(
            *[
                Mono(f"{it['sku']} ×{it['quantity']}", font_size=18, color=C.CATEGORY)
                for it in d["probes"][0]["line_items"]
            ]
        )
        .arrange(RIGHT, buff=0.32)
        .move_to(UP * 2.15)
    )
    grid = (
        VGroup(
            *[
                Square(
                    0.29,
                    stroke_width=0.5,
                    stroke_color=GREY_D,
                    fill_color=C.RF_TARGET,
                    fill_opacity=0.04 + 0.96 * float(w),
                )
                for w in weights
            ]
        )
        .arrange(RIGHT, buff=0.04)
        .move_to(UP * 0.9)
    )
    names = VGroup(
        *[
            Mono(x, font_size=19, color=c).next_to(VGroup(*grid[6 * i : 6 * i + 6]), UP, buff=0.17)
            for i, (x, c) in enumerate([("SKU", C.CATEGORY), ("quantity", C.NUMBER), ("price", GREY_B)])
        ]
    )
    scores = vector_strip(
        logits, C.RF_BRANCH, cell=0.29, gap=0.04, direction=RIGHT, vmax=float(abs(logits).max()), outline=False
    ).move_to(DOWN * 0.05)
    wl = Mono("weights", font_size=18, color=C.RF_TARGET).next_to(grid, LEFT, buff=0.3)
    sl = Mono("q·k / √16", font_size=18, color=C.RF_BRANCH).next_to(scores, LEFT, buff=0.3)
    softmax = Arrow(scores.get_right() + RIGHT * 0.35, grid.get_right() + RIGHT * 0.35, buff=0.05, color=C.RF_TARGET)
    sf = Mono("softmax", font_size=17, color=C.RF_TARGET).next_to(softmax, RIGHT, buff=0.1)
    selected = int(weights.argmax())
    mark = SurroundingRectangle(grid[selected], buff=0.04, color=C.NUMBER, stroke_width=2)
    bars = VGroup(
        *[
            vector_strip(v[:8], C.NUMBER, cell=0.17, gap=0.015, direction=RIGHT, vmax=float(abs(values[:, :8]).max()))
            for v in values
        ]
    ).arrange(DOWN, buff=0.12)
    # Show only the dominant token and three alternatives, keeping their original indices.
    picks = [selected, *[i for i in np.argsort(weights)[::-1] if i != selected][:3]]
    shown = (
        VGroup(
            *[
                VGroup(Mono(f"token {i + 1:02d} | {weights[i]:.4f}", font_size=18, color=C.RF_TARGET), bars[i]).arrange(
                    RIGHT, buff=0.25
                )
                for i in picks
            ]
        )
        .arrange(DOWN, aligned_edge=LEFT, buff=0.28)
        .move_to(LEFT * 2.5 + DOWN * 1.35)
    )
    result = vector_strip(
        context[:8], C.MASS, cell=0.23, gap=0.025, direction=RIGHT, vmax=float(abs(values[:, :8]).max())
    ).move_to(RIGHT * 3.65 + DOWN * 1.35)
    result_label = Mono("Σ weight × V", font_size=24, color=C.MASS).next_to(result, UP, buff=0.3)
    detail = Mono("first 8 of 16 head coordinates", font_size=17, color=GREY_B).next_to(result, DOWN, buff=0.25)
    footer = trace_tag(
        "actual trained reducer, first query / head 0; 18 tokens; all 18 values enter the sum; relflow 0.3.2"
    )
    with scene.voiceover(
        "Let's open an actual trained attention head. <bookmark mark='w'/> The query and keys produce eighteen "
        "scores: six product tokens, six quantities, six prices. Softmax turns the signed scores into "
        "positive weights that add to one. The first head puts almost all its "
        "weight on S eleven, bought four at a time. <bookmark mark='v'/> But attention does not send the weight "
        "itself onward. Each token also supplies a value vector. Here are the leading contributions, with eight "
        "coordinates visible. <bookmark mark='s'/> Multiply each value vector by its weight, and add all eighteen. "
        "The result is this head's context vector. Other heads read separately, then their outputs are projected "
        "and combined. This is one measured read, not an explanation of the entire prediction."
    ) as vo:
        scene.play(FadeIn(head), FadeIn(tokens), FadeIn(footer))
        vo.wait_until("w")
        scene.play(FadeIn(scores), FadeIn(sl), FadeIn(names))
        scene.play(TransformFromCopy(scores.cells, grid), FadeIn(wl), GrowArrow(softmax), FadeIn(sf), Create(mark))
        transmit(scene, [tokens[2]], grid[selected], C.CATEGORY, run_time=0.7)
        vo.wait_until("v")
        scene.play(FadeIn(shown, lag_ratio=0.15))
        vo.wait_until("s")
        scene.play(FadeIn(result_label), FadeIn(detail))
        transmit(scene, [row[1] for row in shown], result, C.NUMBER, weights=weights[picks], run_time=1.4)
        scene.play(FadeIn(result))
    scene.clear_scene()


def contextual_embeddings(scene):
    """Same SKU lookup; different quantity; real vectors change after local mixing."""
    d = D("flow_current")
    a, b = d["traces"][:2]
    raw = np.asarray([a["sku_vectors"][2], b["sku_vectors"][2]])
    mixed = np.asarray([a["coordinate_vectors"][6], b["coordinate_vectors"][6]])
    assert np.allclose(raw[0], raw[1]) and not np.allclose(mixed[0], mixed[1])
    head = heading("The same SKU acquires different context.")
    left = (
        VGroup(Mono("S11 × 4", font_size=29, color=C.CATEGORY), Mono("S11 × 1", font_size=29, color=C.CATEGORY))
        .arrange(DOWN, buff=1.4)
        .move_to(LEFT * 5 + UP * 0.3)
    )
    scale = float(max(abs(raw).max(), abs(mixed).max()))
    lookup = (
        VGroup(*[vector_strip(v[:16], C.CATEGORY, cell=0.18, gap=0.02, direction=RIGHT, vmax=scale) for v in raw])
        .arrange(DOWN, buff=1.5)
        .move_to(LEFT * 1.7 + UP * 0.3)
    )
    output = (
        VGroup(*[vector_strip(v[:16], C.RF_BRANCH, cell=0.18, gap=0.02, direction=RIGHT, vmax=scale) for v in mixed])
        .arrange(DOWN, buff=1.5)
        .move_to(RIGHT * 3.5 + UP * 0.3)
    )
    titles = VGroup(
        Mono("lookup: identical", font_size=23, color=C.CATEGORY).next_to(lookup, UP, buff=0.45),
        Mono("after coordinate attention", font_size=22, color=C.RF_BRANCH).next_to(output, UP, buff=0.45),
    )
    delta = float(np.linalg.norm(mixed[0] - mixed[1]))
    cap = (
        VGroup(
            label("same identity; different item context", font_size=28),
            label(rf"64-d contextual vector difference: ${delta:.3f}$", font_size=25),
        )
        .arrange(DOWN, buff=0.18)
        .move_to(DOWN * 2)
    )
    footer = trace_tag("trained model, fixed weights; first 16 of 64 coordinates; shared scale; relflow 0.3.2")
    with scene.voiceover(
        "An embedding starts with identity, then acquires context. <bookmark mark='i'/> These two items have "
        "the same product code, S eleven. Before attention, their SKU lookup vectors are identical. "
        "<bookmark mark='c'/> One item has quantity four, the other quantity one. Local coordinate attention "
        "lets the SKU read its own item's fields. <bookmark mark='o'/> Now the contextualized SKU vectors differ. "
        "We show the same sixteen coordinates with one shared scale. This is how a product token can carry "
        "information about quantity before the branch reads across the whole order."
    ) as vo:
        scene.play(FadeIn(head), FadeIn(left), FadeIn(footer))
        vo.wait_until("i")
        scene.play(FadeIn(lookup), FadeIn(titles[0]))
        vo.wait_until("c")
        for i in range(2):
            transmit(scene, [lookup[i]], output[i], C.NUMBER, run_time=0.65)
        vo.wait_until("o")
        scene.play(FadeIn(output), FadeIn(titles[1]), FadeIn(cap))
    scene.clear_scene()


def embedding_lookup(scene):
    d = D("flow_current")
    rows = np.asarray(d["traces"][0]["sku_vectors"])
    items = d["probes"][0]["line_items"]
    scale = float(abs(rows).max())
    table = (
        VGroup(
            *[
                VGroup(
                    Mono(it["sku"], font_size=23, color=C.CATEGORY),
                    vector_strip(v[:16], C.CATEGORY, cell=0.19, gap=0.02, direction=RIGHT, vmax=scale),
                ).arrange(RIGHT, buff=0.3)
                for it, v in zip(items, rows)
            ]
        )
        .arrange(DOWN, aligned_edge=LEFT, buff=0.22)
        .move_to(LEFT * 2.6)
    )
    head = heading("A label selects a learned vector.")
    query = Mono("S11", font_size=35, color=C.CATEGORY).move_to(RIGHT * 3.4 + UP * 1.6)
    out = table[2][1].copy().move_to(RIGHT * 3.4 + DOWN * 0.1)
    cap = Mono("same label → same lookup", font_size=23, color=C.CATEGORY).move_to(RIGHT * 3.4 + DOWN * 1.5)
    selection = SurroundingRectangle(table[2], color=C.RF_TARGET, buff=0.08)
    footer = trace_tag("6 displayed trained SKU rows; first 16 of 64 coordinates; one shared scale; relflow 0.3.2")
    with scene.voiceover(
        "Let's make the embedding table tangible. <bookmark mark='t'/> These are six rows from an actual trained "
        "SKU field. Every label owns a vector, with sixteen of its sixty four coordinates visible here. "
        "<bookmark mark='r'/> S eleven selects its row, and that vector enters the model. "
        "<bookmark mark='s'/> A second S eleven reads the same row. The row is a learned parameter, not a new "
        "random vector for each observation. Attention will give that shared identity different context in different items."
    ) as vo:
        scene.play(FadeIn(head), FadeIn(footer))
        vo.wait_until("t")
        scene.play(FadeIn(table, lag_ratio=0.15))
        vo.wait_until("r")
        scene.play(FadeIn(query), Create(selection))
        scene.play(TransformFromCopy(table[2][1], out), run_time=1.2)
        vo.wait_until("s")
        scene.play(FadeIn(cap), Circumscribe(out, color=C.RF_TARGET))
    scene.clear_scene()


def pretraining_geometry(scene):
    d = D("pretraining")["embedding"]
    rows = np.asarray(d["rows"])
    unit = rows / np.linalg.norm(rows, axis=1, keepdims=True)
    labels = d["labels"]
    colors = ["#FFA94D", "#5CD0F0", "#B197FC", "#FF8FA3", "#A8D59A", "#FFD166", "#7FB7DA", "#E890B6"]
    similarity = unit @ unit[0]
    from videos.ml.relflow.common import Plot

    plot = Plot(
        (-0.45, 0.45),
        (-0.6, 7.6),
        width=8.5,
        height=4.7,
        x_ticks=[-0.4, 0, 0.4],
        y_ticks=[],
        font_size=19,
        grid=False,
    ).move_to(LEFT * 0.65 + DOWN * 0.15)

    def position(i):
        row = 7 - (ord(labels[i][0]) - 65)
        return plot.c2p(float(similarity[i]), row + ((i % 25) - 12) * 0.013)

    points = {
        i: Dot(position(i), radius=0.043, color=colors[ord(labels[i][0]) - 65], fill_opacity=0.8)
        for i in range(1, len(labels))
    }
    cloud = VGroup(*points.values())
    group_names = VGroup(
        *[
            Mono(chr(65 + i), font_size=23, color=c).next_to(plot.c2p(-0.45, 7 - i), LEFT, buff=0.2)
            for i, c in enumerate(colors)
        ]
    )
    axis_title = label("cosine similarity to the first SKU", font_size=23, color=GREY_A).next_to(plot, DOWN, buff=0.2)
    head = heading("Which vectors point in the same direction?")
    closest = np.argsort(similarity)[-5:-1]
    marks = VGroup(*[Circle(radius=0.11, color=WHITE, stroke_width=2).move_to(points[int(i)]) for i in closest])
    caption = (
        VGroup(
            Mono(f"anchor: {labels[0]}", font_size=23, color=colors[ord(labels[0][0]) - 65]),
            Mono("four nearest labels:", font_size=18, color=GREY_A),
            *[
                Mono(
                    f"{labels[int(i)]}: {similarity[int(i)]:.3f}",
                    font_size=20,
                    color=colors[ord(labels[int(i)][0]) - 65],
                )
                for i in closest
            ],
        )
        .arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        .move_to(RIGHT * 4.9 + UP * 0.2)
    )
    footer = trace_tag(
        "actual final 0.2.3 embeddings; horizontal positions = full-vector cosine; rows = known synthetic groups"
    )
    with scene.voiceover(
        "Let's inspect what the embedding directions mean. <bookmark mark='g'/> Each dot is one SKU at the end "
        "of the archived pretraining run. Horizontal position is its cosine similarity to the first SKU, shown "
        "as the anchor. Rows and colors organize the known synthetic groups for display; group labels were "
        "never training inputs. <bookmark mark='n'/> Highlight the four nearest other labels. These neighbors "
        "include several other groups. Individual neighborhoods overlap. This comparison uses all embedding coordinates, so it does not depend on "
        "a two-dimensional projection. Next we will repeat that comparison for every group."
    ) as vo:
        scene.play(FadeIn(head), FadeIn(footer), Create(plot), FadeIn(group_names), FadeIn(axis_title))
        vo.wait_until("g")
        scene.play(LaggedStart(*[FadeIn(p) for p in cloud], lag_ratio=0.003), run_time=2)
        vo.wait_until("n")
        scene.play(Create(marks), FadeIn(caption))
    scene.clear_scene()


def residual_block(scene):
    d = D("flow_current")["traces"][0]
    initial = np.asarray(d["coordinate_vectors"])[6]
    att = np.asarray(d["attention_delta"])[2]
    ffn = np.asarray(d["ffn_delta"])[2]
    final = np.asarray(d["sequence_vectors"])[2]
    middle = initial + att
    assert np.allclose(middle + ffn, final, atol=2e-5)
    scale = float(max(abs(initial[:8]).max(), abs(middle[:8]).max(), abs(final[:8]).max()))
    head = heading("A residual stream carries the old information forward.")
    rows = []
    for y, before, after, name, col in [
        (1.1, initial, middle, "attention", C.RF_BRANCH),
        (-1.1, middle, final, "FFN × 8", C.EMBED),
    ]:
        src = vector_strip(before[:8], C.NUMBER, cell=0.17, gap=0.02, direction=DOWN, vmax=scale).move_to(
            LEFT * 5.9 + UP * y
        )
        norm = RoundedRectangle(width=1.8, height=0.75, corner_radius=0.1, stroke_color=GREY_B).move_to(
            LEFT * 3.6 + UP * y
        )
        normlabel = Mono("RMSNorm", font_size=21).move_to(norm)
        block = RoundedRectangle(width=2.1, height=0.75, corner_radius=0.1, stroke_color=col).move_to(
            LEFT * 0.7 + UP * y
        )
        blocklabel = Mono(name, font_size=21, color=col).move_to(block)
        plus = VGroup(
            Circle(radius=0.25, color=C.RF_ROOT), Mono("+", font_size=27, color=C.RF_ROOT).move_to(ORIGIN)
        ).move_to(RIGHT * 2.2 + UP * y)
        out = vector_strip(after[:8], C.RF_ROOT, cell=0.17, gap=0.02, direction=DOWN, vmax=scale).move_to(
            RIGHT * 5.1 + UP * y
        )
        route = VGroup(
            *[
                Arrow(a.get_right(), b.get_left(), buff=0.12, color=col, stroke_width=2)
                for a, b in zip([src, norm, block, plus], [norm, block, plus, out])
            ]
        )
        bypass = (
            VMobject()
            .set_points_as_corners(
                [src.get_top(), src.get_top() + UP * 0.3, [plus.get_x(), src.get_top()[1] + 0.3, 0], plus.get_top()]
            )
            .set_stroke(C.NUMBER, 2)
        )
        rows.append((VGroup(src, norm, normlabel, block, blocklabel, plus, out, route, bypass), route, bypass))
    foot = trace_tag(
        "actual trained SKU token, first 8 of 64 coordinates, shared scale; RMSNorm/FFN routes; relflow 0.3.2"
    )
    cap = (
        Mono("new information is added; the old vector has a direct path", font_size=24, color=C.RF_ROOT)
        .scale_to_fit_width(12.4)
        .move_to(DOWN * 2.65)
    )
    with scene.voiceover(
        "Each transformer layer has a direct path for the existing vector. <bookmark mark='a'/> First, R M S "
        "normalization feeds attention. Attention contributes a change, which is added to the original vector. "
        "The blue shortcut carries that original information around the read operation. "
        "<bookmark mark='f'/> Then another R M S normalization feeds the feed forward network: expand by f f n x, "
        "apply tanh GELU, project back, and add. The default multiplier is eight, independently configurable at "
        "each node. These strips are real intermediate vectors from the trained SKU token. "
        "<bookmark mark='p'/> Padding is zeroed before and after a layer and excluded from attention. "
        "<bookmark mark='m'/> A branch can keep one key and value head per query head, share them in groups, "
        "share one across all heads, or turn attention off. The residual path remains the route for existing information."
    ) as vo:
        scene.play(FadeIn(head), FadeIn(foot))
        for mark, (g, route, bypass) in zip("af", rows):
            vo.wait_until(mark)
            scene.play(FadeIn(g))
            stream(scene, list(route), C.RF_BRANCH, run_time=1.35)
            stream(scene, [bypass], C.NUMBER, run_time=0.7)
        vo.wait_until("p")
        scene.play(FadeIn(cap))
        vo.wait_until("m")
        scene.play(Indicate(rows[0][0][3], color=C.QUERY))
    scene.clear_scene()


def mask_route(scene):
    head = heading("Hide the input. Keep the target on a separate route.")
    visible = (
        VGroup(
            *[
                VGroup(
                    RoundedRectangle(width=1.2, height=0.7, corner_radius=0.1, stroke_color=C.NUMBER),
                    Mono(str(v), font_size=28, color=C.NUMBER),
                )
                for v in [1, 4, 2]
            ]
        )
        .arrange(DOWN, buff=0.32)
        .move_to(LEFT * 5)
    )
    for g in visible:
        g[1].move_to(g[0])
    encoder = VGroup(
        RoundedRectangle(width=2.7, height=1.0, corner_radius=0.12, stroke_color=C.RF_BRANCH),
        Mono("context encoder", font_size=22, color=C.RF_BRANCH),
    ).move_to(LEFT * 0.6 + UP * 0.5)
    query = VGroup(
        RoundedRectangle(width=2.6, height=1.0, corner_radius=0.12, stroke_color=C.RF_TARGET),
        Mono("reconstruct item 2", font_size=21, color=C.RF_TARGET),
    ).move_to(RIGHT * 3.9 + UP * 0.5)
    loss = VGroup(
        RoundedRectangle(width=2.6, height=0.85, corner_radius=0.12, stroke_color=C.MASKED),
        Mono("loss", font_size=25, color=C.MASKED),
    ).move_to(RIGHT * 3.9 + DOWN * 1.9)
    for panel, position in [
        (encoder, LEFT * 0.6 + UP * 0.5),
        (query, RIGHT * 3.9 + UP * 0.5),
        (loss, RIGHT * 3.9 + DOWN * 1.9),
    ]:
        if panel[1].width > panel[0].width - 0.25:
            panel[1].scale_to_fit_width(panel[0].width - 0.25)
        panel[1].move_to(panel[0])
        panel.move_to(position)
    target = visible[1][1].copy()
    rail = (
        VMobject()
        .set_points_as_corners(
            [visible[1].get_left(), [-6.1, visible[1].get_y(), 0], [-6.1, -2.6, 0], [3.9, -2.6, 0], loss.get_bottom()]
        )
        .set_stroke(C.RF_TARGET, 2)
    )
    mark = SurroundingRectangle(visible[1], color=C.MASKED, buff=0.07)
    target_label = Mono("target: 4", font_size=20, color=C.RF_TARGET).next_to(loss, DOWN, buff=0.4)
    incoming = VGroup(
        *[Arrow(g.get_right(), encoder.get_left(), buff=0.1, color=C.RF_BRANCH, stroke_width=2) for g in visible]
    )
    encoded = Arrow(encoder.get_right(), query.get_left(), buff=0.1, color=C.RF_BRANCH)
    predicted = Arrow(query.get_bottom(), loss.get_top(), buff=0.1, color=C.RF_TARGET)
    footer = trace_tag("schematic masked reconstruction; no prediction value or learning score is asserted")
    with scene.voiceover(
        "Follow one hidden value through the learning objective. <bookmark mark='t'/> The original four is kept "
        "as a target. It travels on a separate route to the loss. <bookmark mark='h'/> The encoder receives a "
        "mask token at that position, plus the visible context. The hidden number itself does not enter that "
        "read. With skip enabled, even the mask token would be absent. <bookmark mark='r'/> A decoder asks for "
        "the missing value, and its prediction is compared with the pristine target. The reconstruction error "
        "can train the representations that made that prediction possible."
    ) as vo:
        scene.play(FadeIn(head), FadeIn(visible), FadeIn(footer), Create(mark))
        vo.wait_until("t")
        scene.play(FadeIn(loss), Create(rail))
        scene.add(target)
        scene.play(MoveAlongPath(target, rail), run_time=1.4)
        scene.remove(target)
        scene.play(FadeIn(target_label))
        vo.wait_until("h")
        visible[1][1].become(Mono("MASK", font_size=19, color=C.MASKED).move_to(visible[1]))
        scene.play(visible[1][0].animate.set_stroke(C.MASKED), FadeIn(encoder), *[GrowArrow(a) for a in incoming])
        transmit(scene, [visible[0], visible[2], visible[1]], encoder, C.RF_BRANCH, run_time=1.3)
        vo.wait_until("r")
        scene.play(FadeIn(query), GrowArrow(encoded), GrowArrow(predicted))
        transmit(scene, [encoder], query, C.RF_BRANCH, run_time=0.8)
        transmit(scene, [query], loss, C.RF_TARGET, run_time=0.8)
    scene.clear_scene()
