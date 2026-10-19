"""A synchronized training microscope, using committed 0.3.2 checkpoints."""
# ruff: noqa: F405

from __future__ import annotations

import numpy as np
from explainer import *  # noqa: F403
from videos.ml.relflow.common import C, D, Mono, Plot, label, show_episode_card
from videos.ml.relflow.current_visuals import heading, trace_tag

GRAD = "#D46BFF"
NEG = "#5CD0F0"
POS = "#FFA94D"


def matrix(values, limit, cell=0.21):
    """One fixed signed scale across checkpoints, never per-frame normalization."""
    a = np.asarray(values)
    return VGroup(
        *[
            Square(
                cell,
                stroke_width=0.35,
                stroke_color=GREY_D,
                fill_color=POS if v >= 0 else NEG,
                fill_opacity=0.07 + 0.93 * min(abs(float(v)) / limit, 1),
            )
            for v in a.flat
        ]
    ).arrange_in_grid(*a.shape, buff=0.018)


def attention(values, cell=0.30):
    return VGroup(
        *[
            Square(
                cell, stroke_width=0.5, stroke_color=GREY_D, fill_color=C.RF_TARGET, fill_opacity=0.04 + 0.96 * float(v)
            )
            for v in np.asarray(values).flat
        ]
    ).arrange_in_grid(4, 18, buff=0.035)


def order_cards(items):
    cards = VGroup()
    for i, it in enumerate(items):
        body = RoundedRectangle(
            width=1.65,
            height=1.05,
            corner_radius=0.09,
            stroke_color=C.CATEGORY if it["sku"] in ("S03", "S11") else GREY_D,
            stroke_width=2,
            fill_color=C.RF_BRANCH,
            fill_opacity=0.06,
        )
        text = (
            VGroup(
                Mono(it["sku"], font_size=23, color=C.CATEGORY),
                Mono(f"qty {it['quantity']}", font_size=19, color=C.NUMBER),
                Mono(f"${it['price']:.0f}", font_size=16, color=GREY_B),
            )
            .arrange(DOWN, buff=0.07)
            .move_to(body)
        )
        cards.add(VGroup(body, text))
    return cards.arrange(RIGHT, buff=0.16)


def probability(p, text, color):
    rail = RoundedRectangle(width=3.2, height=0.22, corner_radius=0.05, stroke_color=GREY_D, stroke_width=1)
    fill = Rectangle(width=max(0.008, 3.16 * p), height=0.17, stroke_width=0, fill_color=color, fill_opacity=0.9)
    fill.align_to(rail, LEFT).shift(RIGHT * 0.02)
    split = Line(rail.get_top() + DOWN * 0.28, rail.get_top() + UP * 0.08, color=GREY_B, stroke_width=1)
    name = Mono(text, font_size=17, color=GREY_A).next_to(rail, UP, buff=0.14).align_to(rail, LEFT)
    value = Mono(f"{100 * p:5.1f}%", font_size=21, color=color).next_to(rail, RIGHT, buff=0.18)
    return VGroup(rail, fill, split, name, value)


class RealRun(VoiceoverScene):
    def construct(self):
        show_episode_card(self)
        self.r = D("learning_current")
        self.task()
        self.learn()
        self.backward()
        self.geometry()
        self.intervention()

    def task(self):
        head = heading("Keep the order. Watch the model change.")
        cards = order_cards(self.r["probes"][0]["line_items"]).move_to(UP * 0.85)
        rule = label(r"returned if S03 or S11 is bought 3 or more at a time", font_size=29).move_to(DOWN * 0.45)
        brace = SurroundingRectangle(cards[2], color=C.RF_TARGET, buff=0.08)
        target = (
            VGroup(
                Mono("returned = True", font_size=27, color=C.RF_TARGET),
                Mono("hidden from model inputs", font_size=20, color=C.MASKED),
            )
            .arrange(DOWN, buff=0.2)
            .move_to(DOWN * 1.75)
        )
        tag = trace_tag("real training checkpoints; relflow 0.3.2; seeded synthetic task")
        with self.voiceover(
            "This time, don't just watch a loss curve. Watch the same order while the model learns. "
            "<bookmark mark='a'/> Six items. A fragile product, S eleven, bought four at a time. "
            "Another fragile product, S zero three, bought only once. <bookmark mark='b'/> Our synthetic rule says "
            "this order is returned. That label is hidden from the inputs. Four thousand and ninety six other "
            "orders provide the training signal. This order stays outside the training set."
        ) as vo:
            self.play(FadeIn(head), FadeIn(tag))
            vo.wait_until("a")
            self.play(LaggedStart(*[FadeIn(c, shift=UP * 0.15) for c in cards], lag_ratio=0.12))
            self.play(Create(brace))
            vo.wait_until("b")
            self.play(FadeIn(rule), FadeIn(target))
        self.clear_scene()

    def learn(self):
        snaps = self.r["snapshots"]
        head = heading("One clock. Three views of learning.")
        cards = order_cards(self.r["probes"][0]["line_items"]).scale(0.78).move_to(UP * 2.25 + LEFT * 2)
        grid = attention(snaps[0]["attention"][0]).move_to(LEFT * 3.35 + UP * 0.1)
        heads = VGroup(
            *[Mono(f"h{i}", font_size=17, color=GREY_B).next_to(grid[18 * i], LEFT, buff=0.17) for i in range(4)]
        )
        field_labels = VGroup(
            *[
                Mono(t, font_size=18, color=c).next_to(VGroup(*grid[6 * i : 6 * i + 6]), UP, buff=0.15)
                for i, (t, c) in enumerate((("SKU", C.CATEGORY), ("quantity", C.NUMBER), ("price", GREY_B)))
            ]
        )
        ids = VGroup(
            *[
                Mono(str(j + 1), font_size=13, color=C.RF_TARGET if j == 2 else GREY_B).next_to(
                    grid[j + 6 * f], DOWN, buff=1.08
                )
                for f in range(3)
                for j in range(6)
            ]
        )
        attcap = Mono("first summary query: all four heads", font_size=20, color=C.RF_TARGET).next_to(
            grid, UP, buff=0.55
        )
        attkey = Mono("0 dark → 1 bright | outlines: top read in h0, h1", font_size=14, color=GREY_B).move_to(
            LEFT * 3.35 + DOWN * 1.08
        )

        def top_reads(values):
            return VGroup(
                *[
                    SurroundingRectangle(
                        grid[18 * h + int(np.argmax(values[h]))], buff=0.015, color=C.NUMBER, stroke_width=1.8
                    )
                    for h in (0, 1)
                ]
            )

        reads = top_reads(snaps[0]["attention"][0])
        weights = np.asarray([s["weights"] for s in snaps])
        limit = float(np.max(abs(weights)))
        wg = matrix(snaps[0]["weights"], limit, 0.26).move_to(RIGHT * 4.3 + UP * 0.3)
        wc = Mono("learned Q projection", font_size=20, color=C.RF_BRANCH).next_to(wg, UP, buff=0.28)
        wk = (
            VGroup(
                Mono("same 8 × 8 slice | fixed scale", font_size=15, color=GREY_B),
                Mono(f"−{limit:.2f} cyan | +{limit:.2f} orange", font_size=15, color=GREY_B),
            )
            .arrange(DOWN, buff=0.12)
            .next_to(wg, DOWN, buff=0.2)
        )
        bars = (
            VGroup(
                probability(snaps[0]["prob"][0], "S11 × 4: returned", C.RF_TARGET),
                probability(snaps[0]["prob"][1], "S11 × 1: kept", C.RF_BRANCH),
            )
            .arrange(DOWN, buff=0.45)
            .move_to(RIGHT * 4.8 + DOWN * 2.0)
        )
        # Keep probability rails well clear of the matrix and inside the frame.
        bars.scale(0.83).move_to(RIGHT * 4.3 + DOWN * 2.65)
        clock = Mono("step 0 / 256", font_size=27, color=WHITE).move_to(RIGHT * 4.8 + UP * 2.25)
        loss = Plot(
            (0, 256),
            (0, 0.8),
            width=5.8,
            height=1.15,
            x_ticks=[0, 128, 256],
            y_ticks=[0, 0.8],
            font_size=16,
            grid=False,
        ).move_to(LEFT * 3.2 + DOWN * 2.45)
        lc = Mono("held-out binary cross entropy", font_size=18, color=GREY_A).next_to(loss, UP, buff=0.15)
        tag = trace_tag("measured checkpoints; fixed color scales; interpolated transitions; relflow 0.3.2")
        with self.voiceover(
            "<bookmark mark='a'/> At step zero, each row is one head of the first summary query, reading eighteen tokens: six products, "
            "six quantities, six prices. Bright means a larger share of attention. The first two heads "
            "start by reading the fourth item's price. "
            "<bookmark mark='b'/> The small signed matrix is different: learned query projection parameters. "
            "They persist between orders. Attention is computed anew from each order. "
            "<bookmark mark='c'/> Both versions of our order initially receive almost the same probability. "
            "Now advance the same clock for every panel."
        ) as vo:
            self.play(FadeIn(head), FadeIn(cards), FadeIn(clock), FadeIn(tag))
            vo.wait_until("a")
            self.play(
                FadeIn(grid),
                FadeIn(heads),
                FadeIn(field_labels),
                FadeIn(ids),
                FadeIn(attcap),
                FadeIn(attkey),
                FadeIn(reads),
            )
            vo.wait_until("b")
            self.play(FadeIn(wg), FadeIn(wc), FadeIn(wk))
            vo.wait_until("c")
            self.play(FadeIn(bars), Create(loss), FadeIn(lc))
        path = VMobject(stroke_color=C.VALUED, stroke_width=3).set_points_as_corners(
            [loss.c2p(0, snaps[0]["test_loss"])] * 2
        )
        self.add(path)
        with self.voiceover(
            "The first updates are uncertain. Even the held-out loss can rise. Then the two predictions separate. "
            "The first two heads switch from that price to the S eleven token. Its vector already carries "
            "context from its own quantity. The learned projection changes too. We are watching recorded "
            "checkpoints from eight epochs, not inventing a smooth "
            "learning trajectory. The transitions between them are animated. "
            "<bookmark mark='end'/> Notice that the heads need not all settle on the same token. A bright square "
            "is a read operation, not proof that this token alone caused the answer."
        ) as vo:
            for i, s in enumerate(snaps[1:], 1):
                target = attention(s["attention"][0]).move_to(grid)
                nb = (
                    VGroup(
                        probability(s["prob"][0], "S11 × 4: returned", C.RF_TARGET),
                        probability(s["prob"][1], "S11 × 1: kept", C.RF_BRANCH),
                    )
                    .arrange(DOWN, buff=0.45)
                    .scale(0.83)
                    .move_to(bars)
                )
                points = [loss.c2p(x["step"], x["test_loss"]) for x in snaps[: i + 1]]
                newpath = VMobject(stroke_color=C.VALUED, stroke_width=3).set_points_as_corners(points)
                # Measurements switch to the destination checkpoint; only the picture interpolates.
                clock.become(Mono(f"step {s['step']} / 256", font_size=27).move_to(clock))
                for current, destination in zip(bars, nb):
                    current[4].become(destination[4])
                self.play(
                    Transform(grid, target),
                    Transform(wg, matrix(s["weights"], limit, 0.26).move_to(wg)),
                    Transform(reads, top_reads(s["attention"][0])),
                    Transform(bars, nb),
                    Transform(path, newpath),
                    run_time=1.5,
                    rate_func=linear,
                )
                self.wait(0.25)
            vo.wait_until("end")
            self.play(Circumscribe(grid, color=C.RF_TARGET))
        self.clear_scene()

    def backward(self):
        u = self.r["update"]
        head = heading("The label sends an error backward.")
        names = [
            ("SKU vectors", C.CATEGORY),
            ("item encoder", C.RF_BRANCH),
            ("summary", C.RF_ROOT),
            ("prediction", C.RF_TARGET),
            ("loss", GRAD),
        ]
        nodes = (
            VGroup(
                *[
                    VGroup(
                        RoundedRectangle(width=2.1, height=0.85, corner_radius=0.1, stroke_color=c),
                        Mono(t, font_size=21, color=c),
                    )
                    for t, c in names
                ]
            )
            .arrange(RIGHT, buff=0.45)
            .move_to(UP * 1.65)
        )
        for n in nodes:
            n[1].move_to(n[0])
        arrows = VGroup(
            *[
                Arrow(a.get_right(), b.get_left(), buff=0.05, color=GREY_C, stroke_width=3)
                for a, b in zip(nodes, nodes[1:])
            ]
        )
        backward = VGroup(
            *[
                Arrow(b.get_bottom() + DOWN * 0.3, a.get_bottom() + DOWN * 0.3, buff=0.05, color=GRAD, stroke_width=3)
                for a, b in zip(nodes, nodes[1:])
            ]
        )
        cap = Mono("training batch 16: 128 orders | schematic route", font_size=17, color=GREY_B).move_to(
            LEFT + UP * 2.6
        )
        target = Mono("batch labels", font_size=18, color=C.MASKED).next_to(nodes[-1], UP, buff=0.5)
        supervision = Arrow(target.get_bottom(), nodes[-1].get_top(), buff=0.06, color=C.MASKED, stroke_width=2)
        before, grad, after = [np.array(u[k]) for k in ("before", "gradient", "after")]
        delta = after - before
        displays = VGroup()
        for a, t, c, limit in (
            (before, "before", C.RF_BRANCH, np.max(abs(before))),
            (grad, "gradient", GRAD, np.max(abs(grad))),
            (delta, "actual AdamW change", C.RF_ROOT, np.max(abs(delta))),
        ):
            g = matrix(a, float(limit), 0.235)
            title = Mono(t, font_size=21, color=c).next_to(g, UP, buff=0.22)
            scale = Mono(f"±{limit:.5f}", font_size=17, color=GREY_B).next_to(g, DOWN, buff=0.18)
            displays.add(VGroup(g, title, scale))
        displays.arrange(RIGHT, buff=1.2).move_to(DOWN * 0.9)
        formula = label(
            r"$\theta_{t+1}=\theta_t+\Delta\theta_{\mathrm{AdamW}}$", font_size=30, color=C.RF_ROOT
        ).move_to(DOWN * 2.6)
        row, col = np.unravel_index(abs(grad).argmax(), grad.shape)
        selected = VGroup(
            *[SurroundingRectangle(d[0][row * 8 + col], buff=0.025, color=WHITE, stroke_width=2) for d in displays]
        )
        readout = Mono(
            f"Q[{row},{col}]: {before[row, col]:.5f} → {after[row, col]:.5f} | gradient {grad[row, col]:+.5f}",
            font_size=20,
            color=GREY_A,
        ).move_to(DOWN * 3.08)
        tag = trace_tag("same Q-projection coordinates; each panel has its stated signed scale; relflow 0.3.2")
        with self.voiceover(
            "Where did those changes come from? <bookmark mark='f'/> A training batch flows forward: field vectors become context, "
            "summaries, and probabilities. The batch labels enter at the loss. "
            "<bookmark mark='b'/> Backward: derivatives reach the decoder, attention projections and input embeddings. "
            "The arrows are schematic. The matrices below are measured from training batch sixteen. "
            "<bookmark mark='g'/> Here is the same query projection slice before the update, its loss gradient, "
            "and the actual change applied by Adam W. Each panel states its own scale, so a tiny update stays visible. "
            "<bookmark mark='u'/> Adam W uses accumulated moments and weight decay. Its change is not simply "
            "a rescaled picture of this one gradient. Many small updates build the model we just watched."
        ) as vo:
            self.play(FadeIn(head), FadeIn(cap), FadeIn(tag), FadeIn(nodes))
            vo.wait_until("f")
            self.play(LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.18))
            self.play(FadeIn(target), GrowArrow(supervision))
            pulse = Dot(color=C.NUMBER, radius=0.08)
            for a in arrows:
                self.play(MoveAlongPath(pulse, a), run_time=0.28)
            self.remove(pulse)
            vo.wait_until("b")
            self.play(LaggedStart(*[GrowArrow(a) for a in reversed(backward)], lag_ratio=0.18))
            vo.wait_until("g")
            self.play(LaggedStart(*[FadeIn(d, shift=UP * 0.15) for d in displays], lag_ratio=0.25))
            vo.wait_until("u")
            self.play(FadeIn(formula), FadeIn(selected), FadeIn(readout))
        self.clear_scene()

    def geometry(self):
        snaps = self.r["snapshots"]
        head = heading("The representation learns a useful separation.")
        xy = np.asarray([s["xy"] for s in snaps])
        lo = xy.min(axis=(0, 1))
        hi = xy.max(axis=(0, 1))
        pad = (hi - lo) * 0.05
        lo -= pad
        hi += pad
        plot = Plot(
            (float(lo[0]), float(hi[0])),
            (float(lo[1]), float(hi[1])),
            width=8,
            height=4.6,
            x_ticks=[],
            y_ticks=[],
            font_size=18,
            grid=False,
        ).move_to(LEFT * 1.5 + DOWN * 0.15)
        ys = self.r["labels"]

        def dots(s):
            return VGroup(
                *[
                    plot.dot(x, y, color=C.RF_TARGET if target else C.RF_BRANCH, radius=0.034, opacity=0.75)
                    for (x, y), target in zip(s["xy"], ys)
                ]
            )

        cloud = dots(snaps[0])
        counter = Mono("step 0", font_size=27).move_to(RIGHT * 4.55 + UP * 1.45)
        legend = (
            VGroup(
                Mono("● returned", font_size=23, color=C.RF_TARGET),
                Mono("● kept", font_size=23, color=C.RF_BRANCH),
                Mono("512 held-out orders", font_size=19, color=GREY_B),
                Mono("same point = same order", font_size=18, color=GREY_B),
                Mono("same PCA basis", font_size=18, color=GREY_B),
                Mono("same axis ranges", font_size=18, color=GREY_B),
            )
            .arrange(DOWN, aligned_edge=LEFT, buff=0.23)
            .move_to(RIGHT * 4.6 + DOWN * 0.4)
        )
        tag = trace_tag("exported 64-d root vectors; fixed final PCA basis, center and axes; relflow 0.3.2")
        with self.voiceover(
            "There is another way to see learning. <bookmark mark='a'/> Each dot is the exported root vector "
            "of one held-out order, projected into two dimensions. The same dot keeps its identity through every "
            "checkpoint. We fix the projection and the axis ranges, so the camera cannot manufacture separation. "
            "<bookmark mark='b'/> As training advances, the representations move. Returned and kept orders become "
            "easier to distinguish. This is a two-dimensional view of sixty four dimensions, not the classifier's "
            "complete decision boundary."
        ) as vo:
            self.play(FadeIn(head), FadeIn(tag), Create(plot), FadeIn(counter), FadeIn(legend))
            vo.wait_until("a")
            self.play(FadeIn(cloud))
            vo.wait_until("b")
            for s in snaps[1:]:
                counter.become(Mono(f"step {s['step']}", font_size=27).move_to(counter))
                self.play(
                    Transform(cloud, dots(s)),
                    run_time=1.05,
                    rate_func=linear,
                )
        self.clear_scene()

    def intervention(self):
        s = self.r["snapshots"][-1]
        head = heading("Freeze the weights. Change one fact.")
        cards = order_cards(self.r["probes"][0]["line_items"]).move_to(UP * 1.65)
        focus = SurroundingRectangle(cards[2], color=C.RF_TARGET, buff=0.07)
        bar = probability(s["prob"][0], "P(returned)", C.RF_TARGET).scale(1.35).move_to(DOWN * 0.45)
        stamp = Mono("step 256 • parameters frozen", font_size=24, color=C.RF_ROOT).move_to(DOWN * 1.6)
        tag = trace_tag("controlled model-input intervention; not a claim about real-world causality; relflow 0.3.2")
        result = label(
            rf"held-out: {100 * s['test_accuracy']:.1f}\% accuracy on 512 orders", font_size=27, color=C.VALUED
        ).move_to(DOWN * 2.5)
        with self.voiceover(
            "Now test the story. <bookmark mark='a'/> Freeze every learned parameter. Keep the products, prices "
            "and positions identical. <bookmark mark='b'/> Change only S eleven's quantity, from four to one. "
            "The predicted return probability falls. The fragile product is still there; the conjunction is gone. "
            "<bookmark mark='c'/> Put four back, then move that whole item to the first slot. The prediction "
            "remains high. A controlled input change tells us more than one bright attention square. "
            f"<bookmark mark='d'/> On five hundred and twelve held-out orders, this run gets "
            f"{100 * s['test_accuracy']:.1f} percent accuracy. It is a small synthetic learning demonstration. "
            "The useful picture is the whole loop: examples, prediction, error, parameter updates, and a new prediction."
        ) as vo:
            self.play(FadeIn(head), FadeIn(tag), FadeIn(cards), Create(focus), FadeIn(stamp))
            vo.wait_until("a")
            self.play(FadeIn(bar))
            vo.wait_until("b")
            new = order_cards(self.r["probes"][1]["line_items"]).move_to(cards)
            next_bar = probability(s["prob"][1], "P(returned)", C.RF_BRANCH).scale(1.35).move_to(bar)
            bar[4].become(next_bar[4])
            cards[2][1][1].become(new[2][1][1])
            self.play(
                Transform(cards, new),
                Transform(bar, next_bar),
            )
            vo.wait_until("c")
            new = order_cards(self.r["probes"][0]["line_items"]).move_to(cards)
            next_bar = probability(s["prob"][0], "P(returned)", C.RF_TARGET).scale(1.35).move_to(bar)
            bar[4].become(next_bar[4])
            cards[2][1][1].become(new[2][1][1])
            self.play(
                Transform(cards, new),
                Transform(bar, next_bar),
            )
            # Preserve item identity during the move instead of crossfading the entire row.
            targets = [cards[i].get_center().copy() for i in range(6)]
            next_bar = probability(s["prob"][2], "P(returned)", C.RF_TARGET).scale(1.35).move_to(bar)
            bar[4].become(next_bar[4])
            self.play(
                cards[2].animate.move_to(targets[0]),
                cards[0].animate.move_to(targets[1]),
                cards[1].animate.move_to(targets[2]),
                focus.animate.move_to(targets[0]),
                Transform(bar, next_bar),
                run_time=1.5,
                path_arc=PI / 3,
            )
            vo.wait_until("d")
            self.play(FadeIn(result))
        self.clear_scene()
