from __future__ import annotations

import numpy as np

from explainer import *  # noqa: F403
from videos.relflow.common import (
    D, Mono, Plot, chip, label, mtex, note, real_tag, schematic_tag, show_episode_card, tile,
)

FIELDS = [("sku", C.CATEGORY), ("quantity", C.NUMBER), ("price", C.NUMBER)]
N_SLOTS = 8
LIVE = 5  # this order has five items; slots 5-7 are padding


def token(color, size=0.42, live=True) -> VGroup:
    return tile(color if live else C.PADDED, size, fill=0.85 if live else 0.3)


class InsideABranch(VoiceoverScene):
    def construct(self):
        show_episode_card(self)
        self.grid()
        self.coordinate()
        self.unroll()
        self.rope()
        self.layer()
        self.evidence()

    # ------------------------------------------------------------------
    def grid(self):
        rows = VGroup()
        for name, col in FIELDS:
            rows.add(VGroup(*[token(col, live=j < LIVE) for j in range(N_SLOTS)]).arrange(RIGHT, buff=0.22))
        rows.arrange(DOWN, buff=0.28).move_to(RIGHT * 0.6 + UP * 0.2)
        names = VGroup(*[Mono(n, font_size=24, color=c).next_to(r, LEFT, buff=0.35) for (n, c), r in zip(FIELDS, rows)])
        slots = VGroup(*[Mono(str(j), font_size=20, color=GREY_B).next_to(rows[0][j], UP, buff=0.15) for j in range(N_SLOTS)])
        sl = label(r"slot", font_size=22, color=GREY_B).next_to(slots, LEFT, buff=0.3)
        pad = Brace(VGroup(*rows[2][LIVE:]), DOWN, color=GREY_B)
        pad_l = label(r"padding", font_size=22, color=GREY_B).next_to(pad, DOWN, buff=0.1)
        with self.voiceover(
            "Every field has been embedded: one vector per field, per slot. Let's step inside a branch, an order's line items, and see what "
            "its encoder does with them. <bookmark mark='g'/> Arrange the tokens in a grid: one row per field, one "
            "column per slot. This order has five items, so the last three columns are padding."
        ) as vo:
            self.play(LaggedStart(*[FadeIn(r, lag_ratio=0.05) for r in rows], lag_ratio=0.3), run_time=2)
            vo.wait_until("g")
            self.play(FadeIn(names), FadeIn(slots), FadeIn(sl))
            self.play(GrowFromCenter(pad), FadeIn(pad_l))
        self.rows, self.names, self.slots, self.sl = rows, names, slots, sl
        self.pad = VGroup(pad, pad_l)

    # ------------------------------------------------------------------
    def coordinate(self):
        rows = self.rows
        boxes = VGroup(*[SurroundingRectangle(VGroup(rows[0][j], rows[2][j]), buff=0.08, color=C.RF_FIELD, stroke_width=2,
                                              corner_radius=0.08) for j in range(LIVE)])
        arcs = VGroup()
        for j in range(LIVE):
            for a, b in ((0, 1), (1, 2), (0, 2)):
                p, q = rows[a][j].get_right(), rows[b][j].get_right()
                arcs.add(ArcBetweenPoints(p, q, angle=-PI / 1.6 if (a, b) != (0, 2) else -PI / 1.3, color=C.ATTN,
                                          stroke_width=2).set_stroke(opacity=0.85))
        head = label(r"step 1, coordinate attention: the fields of one item attend to each other", font_size=28)
        head.to_edge(UP, buff=0.45)
        note1 = label(r"one transformer layer, run separately for every item:\\a sequence of just three tokens",
                      font_size=24, color=GREY_A).to_edge(DOWN, buff=0.6)
        with self.voiceover(
            "The first step is local. <bookmark mark='c'/> The SKU, the quantity and the price of one line item belong "
            "together. So each column, separately, goes through one small transformer layer: a sequence of just three "
            "tokens, attending only to each other. <bookmark mark='k'/> Afterwards, the SKU token knows its quantity, "
            "and the quantity token knows which product it's counting."
        ) as vo:
            vo.wait_until("c")
            self.play(FadeIn(head), FadeOut(self.pad))
            self.play(LaggedStart(*[Create(b) for b in boxes], lag_ratio=0.12), run_time=1.2)
            self.play(Create(arcs), FadeIn(note1), run_time=1.4)
            vo.wait_until("k")
            for j in range(LIVE):
                self.play(*[rows[i][j].sq.animate.set_fill(interpolate_color(ManimColor(FIELDS[i][1]), WHITE, 0.25))
                            for i in range(3)], run_time=0.25)
        self.play(FadeOut(boxes), FadeOut(arcs), FadeOut(head), FadeOut(note1))

    # ------------------------------------------------------------------
    def unroll(self):
        rows = self.rows
        seq = VGroup(*[t.copy() for r in rows for t in r])
        for t in seq:
            t.scale(0.78)
        seq.arrange(RIGHT, buff=0.08).move_to(UP * 0.6)
        pos = VGroup(*[Mono(str(k), font_size=14, color=GREY_B).next_to(seq[k], DOWN, buff=0.08) for k in range(24)])
        braces = VGroup()
        for i, (n, c) in enumerate(FIELDS):
            b = Brace(VGroup(*seq[8 * i:8 * i + 8]), UP, color=c, buff=0.08)
            braces.add(VGroup(b, Mono(n, font_size=20, color=c).next_to(b, UP, buff=0.08)))
        head = label(r"step 2: unroll into one sequence of 24 positions", font_size=28).to_edge(UP, buff=0.45)
        live = [k for k in range(24) if k % 8 < LIVE]
        arcs = VGroup()
        rng = np.random.default_rng(1)
        for _ in range(26):
            a, b = sorted(rng.choice(live, 2, replace=False))
            arcs.add(ArcBetweenPoints(seq[a].get_bottom() + DOWN * 0.3, seq[b].get_bottom() + DOWN * 0.3,
                                      angle=PI / 2.4, color=C.ATTN, stroke_width=1.5).set_stroke(opacity=0.6))
        stack = label(r"then $n_{\mathrm{layers}}$ ordinary transformer layers run across all of it:\\"
                      r"every item can compare itself with every other item", font_size=26, color=GREY_A)
        stack.to_edge(DOWN, buff=0.5)
        with self.voiceover(
            "Next, the grid is unrolled into a single sequence: <bookmark mark='u'/> all eight SKU slots, then all "
            "eight quantities, then all eight prices. Twenty-four positions. <bookmark mark='s'/> A stack of ordinary "
            "transformer layers runs across that whole sequence, so now every item can compare itself with every "
            "other item in the order."
        ) as vo:
            vo.wait_until("u")
            self.play(FadeIn(head), FadeOut(self.names), FadeOut(self.slots), FadeOut(self.sl),
                      ReplacementTransform(VGroup(*[t for r in rows for t in r]), seq), run_time=1.6)
            self.play(FadeIn(braces), FadeIn(pos))
            vo.wait_until("s")
            self.play(Create(arcs), FadeIn(stack), run_time=1.6)
        self.play(FadeOut(arcs), FadeOut(stack), FadeOut(head))
        self.seq, self.pos, self.braces = seq, pos, braces

    # ------------------------------------------------------------------
    def rope(self):
        r = D("rope")
        scores = np.array(r["scores"])
        n = scores.shape[0]
        for d in range(-n + 1, n):
            assert np.ptp(np.diagonal(scores, offset=d)) < 1e-4
        seq, pos, braces = self.seq, self.pos, self.braces
        top = VGroup(seq, pos, braces)
        # a rotating query arrow
        circ = Circle(radius=0.8, color=GREY_D, stroke_width=1.5).move_to(LEFT * 4.6 + DOWN * 1.8)
        ang = ValueTracker(0.4)
        arrow = always_redraw(lambda: Arrow(circ.get_center(), circ.get_center() + 0.8 * np.array(
            [np.cos(ang.get_value()), np.sin(ang.get_value()), 0]), buff=0, color=C.QUERY, stroke_width=4,
            max_tip_length_to_length_ratio=0.2))
        rl = label(r"rotate each query and key\\by an angle $\propto$ its position", font_size=24, color=GREY_A)
        rl.next_to(circ, UP, buff=0.25)
        vmax = np.abs(scores).max()
        cells = VGroup()
        for i in range(n):
            for j in range(n):
                v = scores[i, j] / vmax
                col = interpolate_color(ManimColor("#3BA7E0"), ManimColor("#F08A3C"), (v + 1) / 2)
                cells.add(Square(0.12, stroke_width=0, fill_color=col, fill_opacity=0.25 + 0.75 * abs(v)))
        cells.arrange_in_grid(n, n, buff=0.008).move_to(RIGHT * 0.2 + DOWN * 1.45)
        cl = label(r"query position $\downarrow$, key position $\rightarrow$", font_size=20, color=GREY_A)
        cl.next_to(cells, UP, buff=0.12)
        diag = DashedLine(cells[8].get_center(), cells[(n - 9) * n + n - 1].get_center(), color=WHITE, stroke_width=2)
        insight = VGroup(
            label(r"every diagonal is constant:", font_size=24),
            label(r"the score depends only on the distance", font_size=24, color=GREY_A),
            label(r"an item's SKU and its own quantity:", font_size=24),
            label(r"always exactly 8 positions apart", font_size=26, color=YELLOW),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12).to_edge(RIGHT, buff=0.35).shift(DOWN * 1.4)
        pair = VGroup()
        for j in (1, 3):
            a, b = seq[j], seq[8 + j]
            pair.add(ArcBetweenPoints(a.get_top() + UP * 0.05, b.get_top() + UP * 0.05, angle=-PI / 3, color=YELLOW,
                                      stroke_width=3))
        tag = real_tag(r"scores from relflow's \texttt{RotaryEmbedding} (head width 16)", corner=UR)
        with self.voiceover(
            "How does a token know where it sits? <bookmark mark='r'/> With rotary position embeddings. Every query and "
            "every key is rotated by an angle that grows with its position. <bookmark mark='d'/> Rotate both, and the "
            "score between two tokens depends only on how far apart they are. Here are real scores from relflow's "
            "rotary module, for one query and one key placed at every pair of positions: every diagonal is constant. "
            "<bookmark mark='p'/> And look what that buys inside a branch. An item's SKU and its own quantity are "
            "always exactly eight positions apart, whichever item it is. <bookmark mark='o'/> So 'my own quantity' is "
            "one fixed offset, the same for every item, and attention can learn it once."
        ) as vo:
            self.play(top.animate.shift(UP * 1.6), run_time=0.8)
            vo.wait_until("r")
            self.play(Create(circ), FadeIn(rl), FadeIn(arrow))
            self.play(ang.animate.set_value(0.4 + 2.4), run_time=2.0, rate_func=smooth)
            vo.wait_until("d")
            self.play(FadeIn(cells, lag_ratio=0.0005), FadeIn(cl), FadeIn(tag), run_time=1.5)
            self.play(Create(diag), FadeIn(insight[:2]))
            vo.wait_until("p")
            self.play(Create(pair), FadeIn(insight[2:]))
            vo.wait_until("o")
            self.play(Indicate(insight[3], color=YELLOW))
        arrow.clear_updaters()
        self.play(FadeOut(VGroup(circ, arrow, rl, cells, cl, diag, insight, pair, tag, top)))

    # ------------------------------------------------------------------
    def layer(self):
        x = mtex(r"x", font_size=40)
        steps = [r"\mathrm{LayerNorm}", r"\text{attention}", r"+", r"\mathrm{LayerNorm}", r"\text{feed-forward}", r"+"]
        f1 = mtex(r"x \leftarrow x + \text{Attention}\big(\mathrm{LN}(x)\big)", font_size=40)
        f2 = mtex(r"x \leftarrow x + \text{FFN}\big(\mathrm{LN}(x)\big)", font_size=40)
        f3 = mtex(r"x \leftarrow x \odot \text{present}", font_size=40)
        fs = VGroup(f1, f2, f3).arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to(LEFT * 2.6 + UP * 0.8)
        lab3 = label(r"padding zeroed going in and coming out,\\and masked out of every attention", font_size=24,
                     color=GREY_A).next_to(f3, DOWN, buff=0.25).align_to(f3, LEFT)
        modes = VGroup(
            label(r"key/value heads per query head:", font_size=26),
            VGroup(Mono('"mha"', font_size=24, color=C.KEY), label(r"one each", font_size=24, color=GREY_A)).arrange(RIGHT, buff=0.3),
            VGroup(Mono('"gqa"', font_size=24, color=C.KEY), label(r"one per pair", font_size=24, color=GREY_A)).arrange(RIGHT, buff=0.3),
            VGroup(Mono('"mqa"', font_size=24, color=C.KEY), label(r"one shared by all", font_size=24, color=GREY_A)).arrange(RIGHT, buff=0.3),
            VGroup(Mono("None", font_size=24, color=C.KEY), label(r"no attention at all", font_size=24, color=GREY_A)).arrange(RIGHT, buff=0.3),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).to_edge(RIGHT, buff=0.7).shift(UP * 0.5)
        with self.voiceover(
            "Each of those layers is the standard pre-norm block: <bookmark mark='a'/> normalize, attend, add the result "
            "back; <bookmark mark='f'/> normalize, feed forward, add back. <bookmark mark='p'/> Padding is zeroed on the "
            "way in and on the way out, and it's masked out of every attention. <bookmark mark='m'/> Each branch can "
            "also choose how many key and value heads to keep: one per query head, one per pair, one shared by all, or "
            "no attention at all."
        ) as vo:
            vo.wait_until("a")
            self.play(Write(f1))
            vo.wait_until("f")
            self.play(Write(f2))
            vo.wait_until("p")
            self.play(Write(f3), FadeIn(lab3))
            vo.wait_until("m")
            self.play(FadeIn(modes, lag_ratio=0.2))
        self.play(FadeOut(VGroup(fs, lab3, modes)))

    # ------------------------------------------------------------------
    def evidence(self):
        r = D("returns")
        res = r["results"]
        assert res["intact"]["auc"] > 0.999 and res["misaligned"]["auc"] < 0.95
        rule = VGroup(
            label(r"a real test of alignment", font_size=30),
            label(r"an order comes back $\iff$ it has a fragile SKU (S03 or S11)", font_size=26, color=GREY_A),
            label(r"bought three or more at a time", font_size=26, color=GREY_A),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15).to_corner(UL, buff=0.6)
        ex_a = VGroup(Mono("S11 · 4", font_size=26, color=WHITE), Mono("S07 · 1", font_size=26, color=WHITE)).arrange(DOWN, buff=0.1)
        ex_b = VGroup(Mono("S11 · 1", font_size=26, color=WHITE), Mono("S07 · 4", font_size=26, color=WHITE)).arrange(DOWN, buff=0.1)
        ta = label(r"returned", font_size=24, color=C.RF_TARGET).next_to(ex_a, DOWN, buff=0.15)
        tb = label(r"kept", font_size=24, color=GREY_A).next_to(ex_b, DOWN, buff=0.15)
        ga, gb = VGroup(ex_a, ta), VGroup(ex_b, tb)
        VGroup(ga, gb).arrange(RIGHT, buff=0.9).next_to(rule, DOWN, buff=0.5).align_to(rule, LEFT).shift(RIGHT * 0.4)
        same = label(r"same SKUs, same quantities,\\different pairs", font_size=22, color=GREY_A).next_to(VGroup(ga, gb), DOWN, buff=0.3)
        pl = Plot((0, 2), (0.5, 1.0), width=4.6, height=3.6, y_ticks=[0.5, 0.75, 1.0], font_size=22)
        pl.to_edge(RIGHT, buff=0.9).shift(DOWN * 0.4)
        b1 = pl.bar(0.5, res["intact"]["auc"], 0.6, C.VALUED)
        b2 = pl.bar(1.5, res["misaligned"]["auc"], 0.6, RED)
        v1 = label(f"{res['intact']['auc']:.3f}", font_size=26).next_to(b1, UP, buff=0.08)
        v2 = label(f"{res['misaligned']['auc']:.3f}", font_size=26).next_to(b2, UP, buff=0.08)
        x1 = label(r"test orders", font_size=22, color=GREY_A).next_to(pl.c2p(0.5, 0.5), DOWN, buff=0.15)
        x2 = label(r"quantities shuffled\\within each order", font_size=22, color=GREY_A).next_to(pl.c2p(1.5, 0.5), DOWN, buff=0.15)
        yt = label(r"held-out AUC", font_size=22, color=GREY_A).next_to(pl, UP, buff=0.5).align_to(pl, LEFT)
        tag = real_tag(rf"real run: {r['train_rows']:,} training orders, {r['test_rows']:,} test orders".replace(",", "{,}"))
        with self.voiceover(
            "Does that pairing really matter? <bookmark mark='t'/> Here's a real test. We trained a small relflow model "
            "on synthetic orders, where an order comes back only if it contains a fragile SKU, bought three or more at "
            "a time. <bookmark mark='e'/> So S11 times four, next to S07 times one, comes back. Swap the quantities, "
            "and it doesn't. <bookmark mark='a'/> On new orders, the model is perfect: an AUC of one. "
            "<bookmark mark='s'/> Now take the same test orders and shuffle the quantities among each order's items. "
            f"Same SKUs, same quantities, wrong pairs. <bookmark mark='d'/> The AUC drops to {res['misaligned']['auc']:.2f}. The model wasn't "
            "counting SKUs and quantities separately. It was reading them as pairs."
        ) as vo:
            vo.wait_until("t")
            self.play(FadeIn(rule, lag_ratio=0.2), FadeIn(tag))
            vo.wait_until("e")
            self.play(FadeIn(ga))
            self.play(FadeIn(gb), FadeIn(same))
            vo.wait_until("a")
            self.play(Create(pl), FadeIn(yt), GrowFromEdge(b1, DOWN), FadeIn(v1), FadeIn(x1))
            vo.wait_until("s")
            self.play(FadeIn(x2))
            vo.wait_until("d")
            self.play(GrowFromEdge(b2, DOWN), FadeIn(v2))
        self.wait(0.5)
        self.play(*[FadeOut(m) for m in self.mobjects])
