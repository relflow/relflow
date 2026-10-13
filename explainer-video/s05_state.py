from __future__ import annotations

import numpy as np

from explainer import *  # noqa: F403
from videos.relflow.common import (
    D, Mono, Plot, label, mtex, note, proof_tag, show_episode_card, state_tile, tile, vector_strip,
)


class ValueState(VoiceoverScene):
    def construct(self):
        show_episode_card(self)
        self.imputation()
        self.two_inputs()
        self.proof()
        self.padding()
        self.cases()

    # ------------------------------------------------------------------
    def imputation(self):
        def rec(val, ans, col):
            a = Mono(f"measurement: {val}", font_size=28, color=WHITE)
            b = Mono(f"target: {ans}", font_size=28, color=C.RF_TARGET)
            g = VGroup(a, b).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
            frame = SurroundingRectangle(g, buff=0.25, color=col, stroke_width=1.6, corner_radius=0.1)
            return VGroup(frame, g)

        zero = rec("0.0", "false", GREY_B)
        null = rec("null", "true", C.NULL_STATE)
        VGroup(zero, null).arrange(RIGHT, buff=1.4).move_to(UP * 1.2)
        cap0 = label(r"the sensor read zero", font_size=26, color=GREY_A).next_to(zero, UP, buff=0.25)
        cap1 = label(r"the sensor never reported", font_size=26, color=C.NULL_STATE).next_to(null, UP, buff=0.25)
        filled = rec("0.0", "true", RED).move_to(null)
        fill_note = label(r"\texttt{fillna(0)}", font_size=30, color=RED).next_to(null, DOWN, buff=0.5)
        same = label(r"identical inputs, different answers", font_size=34, color=RED).to_edge(DOWN, buff=1.2)
        with self.voiceover(
            "Why does relflow track a state at all? Because inside a tensor, a zero is just a zero. "
            "<bookmark mark='a'/> Take a measurement: one sensor read exactly zero, <bookmark mark='b'/> and another "
            "never reported. <bookmark mark='f'/> Fill the null with zero, as so many pipelines do, "
            "<bookmark mark='s'/> and these two records become identical, even though they need different answers."
        ) as vo:
            vo.wait_until("a")
            self.play(FadeIn(zero), FadeIn(cap0))
            vo.wait_until("b")
            self.play(FadeIn(null), FadeIn(cap1))
            vo.wait_until("f")
            self.play(FadeIn(fill_note, shift=UP * 0.1))
            self.play(ReplacementTransform(null, filled), cap1.animate.set_color(GREY_A))
            vo.wait_until("s")
            self.play(Write(same))
        self.play(FadeOut(VGroup(zero, filled, cap0, cap1, fill_note, same)))

    # ------------------------------------------------------------------
    def two_inputs(self):
        names = ["valued", "null", "padded", "masked", "other"]
        rows = VGroup()
        for k, n in enumerate(names):
            t = state_tile(k, 0.42)
            v = vector_strip(np.random.default_rng(10 + k).normal(0, 1, 12), color=C.ATTN, cell=0.17, gap=0.025,
                             direction=RIGHT)
            lab = label(n, font_size=26, color=GREY_A)
            rows.add(VGroup(t, lab, v))
        for r in rows:
            r[1].next_to(r[0], RIGHT, buff=0.2)
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        for r in rows:
            r[2].move_to(rows.get_left() + RIGHT * 3.3, aligned_edge=LEFT).set_y(r[0].get_y())
        rows.move_to(UP * 1.35)
        head = label(r"five state tokens, each with a learned vector", font_size=28).next_to(rows, UP, buff=0.4)
        head.align_to(rows, LEFT)
        other_note = note(r"(reserved for extensions)", font_size=20).next_to(rows[4][2], RIGHT, buff=0.2)

        state_v = rows[1][2].copy()
        content_v = vector_strip(np.random.default_rng(3).normal(0, 1, 12), color=C.NUMBER, cell=0.17, gap=0.025,
                                 direction=RIGHT)
        plus = mtex("+", font_size=44)
        eq = mtex("=", font_size=44)
        result = vector_strip(np.random.default_rng(4).normal(0, 1, 12), color=WHITE, cell=0.17, gap=0.025,
                              direction=RIGHT)
        sum_row = VGroup(content_v, plus, state_v, eq, result).arrange(RIGHT, buff=0.25)
        sum_row.move_to(DOWN * 1.75)
        lab_c = label(r"from the content", font_size=24, color=C.NUMBER).next_to(content_v, UP, buff=0.2)
        lab_s = label(r"from the state", font_size=24, color=C.ATTN).next_to(state_v, UP, buff=0.2)
        lab_r = label(r"the token", font_size=24).next_to(result, UP, buff=0.2)
        z_note = label(r"content and state are separate inputs: a zero never has to mean null or padding",
                       font_size=26, color=GREY_A).next_to(sum_row, DOWN, buff=0.45)
        with self.voiceover(
            "So relflow keeps two separate inputs for every slot. One is the content: the number, the label, the "
            "hash. <bookmark mark='s'/> The other is a state, one of five tokens: valued, null, padded, masked, and a "
            "fifth reserved for extensions. Each state has its own learned vector. <bookmark mark='e'/> A field's "
            "embedder computes a vector from the content, and combines it with the vector for the state. "
            "<bookmark mark='z'/> So a zero never has to stand in for null, or for padding."
        ) as vo:
            vo.wait_until("s")
            self.play(FadeIn(head), LaggedStart(*[FadeIn(r, shift=RIGHT * 0.1) for r in rows], lag_ratio=0.15),
                      FadeIn(other_note), run_time=2)
            vo.wait_until("e")
            self.play(FadeIn(content_v), FadeIn(lab_c))
            self.play(TransformFromCopy(rows[1][2], state_v), FadeIn(plus), FadeIn(lab_s))
            self.play(FadeIn(eq), FadeIn(result), FadeIn(lab_r))
            vo.wait_until("z")
            self.play(FadeIn(z_note, shift=UP * 0.1))
        self.play(FadeOut(VGroup(rows, head, other_note, sum_row, lab_c, lab_s, lab_r, z_note)))

    # ------------------------------------------------------------------
    def proof(self):
        m = D("proofs")["metrics"]["P038"]
        assert m["state_auc"] == 1.0 and m["prefilled_auc"] == 0.5
        bars = [
            (r"nulls kept\\as nulls", m["state_auc"], C.VALUED),
            (r"nulls filled\\with 0.0 first", m["prefilled_auc"], RED),
            (r"control: nulls\\at random", m["independent_auc"], GREY_B),
        ]
        pl = Plot((0, 3), (0, 1), width=6.0, height=3.6, y_ticks=[0, 0.5, 1.0], font_size=24)
        pl.move_to(DOWN * 0.3 + RIGHT * 2.2)
        rects = VGroup()
        texts = VGroup()
        for i, (t, v, col) in enumerate(bars):
            r = pl.bar(i + 0.5, v, 0.6, col)
            rects.add(r)
            texts.add(label(t, font_size=22, color=GREY_A).next_to(pl.c2p(i + 0.5, 0), DOWN, buff=0.2))
        vals = VGroup(*[label(f"{v:.2f}", font_size=28).next_to(r, UP, buff=0.1) for r, (_, v, _) in zip(rects, bars)])
        yt = label(r"held-out AUC", font_size=24, color=GREY_A).next_to(pl, UP, buff=0.25).align_to(pl, LEFT)
        chance = pl.hline(0.5, color=GREY_C)
        setup = VGroup(
            label(r"relflow's proof P038:", font_size=28),
            label(r"every present measurement is 0.0", font_size=26, color=GREY_A),
            label(r"only missingness predicts the label", font_size=26, color=GREY_A),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).to_corner(UL, buff=0.7)
        tag = proof_tag(r"P038, null versus zero")
        with self.voiceover(
            "relflow's own test suite checks exactly this. <bookmark mark='s'/> In proof P038, every measurement "
            "that's present is exactly zero, and missingness alone decides the label. <bookmark mark='a'/> The model "
            "separates the two classes perfectly: an AUC of one. <bookmark mark='f'/> Fill the nulls with zeros "
            "before training, and the AUC collapses to one half, a coin flip. <bookmark mark='c'/> A control with "
            "random nulls lands at chance too. The information was destroyed before the model ever saw it."
        ) as vo:
            vo.wait_until("s")
            self.play(FadeIn(setup, lag_ratio=0.2), FadeIn(tag))
            self.play(Create(pl), FadeIn(yt), Create(chance))
            vo.wait_until("a")
            self.play(GrowFromEdge(rects[0], DOWN), FadeIn(texts[0]), FadeIn(vals[0]))
            vo.wait_until("f")
            self.play(GrowFromEdge(rects[1], DOWN), FadeIn(texts[1]), FadeIn(vals[1]))
            vo.wait_until("c")
            self.play(GrowFromEdge(rects[2], DOWN), FadeIn(texts[2]), FadeIn(vals[2]))
        self.play(FadeOut(VGroup(pl, rects, texts, vals, yt, chance, setup, tag)))

    # ------------------------------------------------------------------
    def padding(self):
        states = [0, 0, 1, 0, 2, 2, 2, 2]
        toks = VGroup(*[state_tile(s, 0.62) for s in states]).arrange(RIGHT, buff=0.3).move_to(UP * 1.5)
        idx = VGroup(*[Mono(str(i), font_size=20, color=GREY_B).next_to(t, DOWN, buff=0.12) for i, t in enumerate(toks)])
        idx.add(label(r"slot", font_size=22, color=GREY_B).next_to(idx[0], LEFT, buff=0.3))
        live = [i for i, s in enumerate(states) if s != 2]
        arcs = VGroup()
        for a in live:
            for b in live:
                if a < b:
                    arcs.add(ArcBetweenPoints(toks[a].get_top() + UP * 0.05, toks[b].get_top() + UP * 0.05,
                                              angle=-PI / 2.2, color=C.ATTN, stroke_width=2).set_stroke(opacity=0.8))
        # the attention mask, as a matrix: rows attend over columns
        n = len(states)
        mat = VGroup()
        for i in range(n):
            for j in range(n):
                on = states[i] != 2 and states[j] != 2
                mat.add(Square(0.36, stroke_width=0.6, stroke_color=GREY_D, fill_color=C.ATTN if on else BACKGROUND,
                               fill_opacity=0.85 if on else 1))
        mat.arrange_in_grid(n, n, buff=0.02).move_to(DOWN * 1.55 + LEFT * 2.8)
        mlab = label(r"who may attend to whom", font_size=24, color=GREY_A).next_to(mat, UP, buff=0.15)
        notes = VGroup(
            label(r"null: a real coordinate, visible to attention", font_size=26, color=C.NULL_STATE),
            label(r"padding: no coordinate at all", font_size=26, color=GREY_A),
            label(r"zeroed, excluded from every attention", font_size=24, color=GREY_B),
            label(r"and every average", font_size=24, color=GREY_B),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18).next_to(mat, RIGHT, buff=0.9)
        with self.voiceover(
            "Null and padding are different, too. <bookmark mark='n'/> A null is a real coordinate: the item exists, "
            "its quantity is null, and attention can see that. <bookmark mark='p'/> Padding is no coordinate at all. "
            "Padded slots are zeroed, and they're excluded from every attention computation and every average. "
            "<bookmark mark='m'/> Here's the mask: live tokens attend to each other, and padding never takes part, "
            "however much room the branch reserves."
        ) as vo:
            self.play(LaggedStart(*[FadeIn(t, scale=0.7) for t in toks], lag_ratio=0.08), FadeIn(idx))
            vo.wait_until("n")
            self.play(Indicate(toks[2], color=C.NULL_STATE), FadeIn(notes[0]))
            self.play(Create(arcs), run_time=1.2)
            vo.wait_until("p")
            self.play(FadeIn(notes[1]), *[toks[i].animate.set_opacity(0.35) for i in range(4, 8)])
            self.play(FadeIn(notes[2]), FadeIn(notes[3]))
            vo.wait_until("m")
            self.play(FadeIn(mat, lag_ratio=0.01), FadeIn(mlab), run_time=1.2)
        self.play(FadeOut(VGroup(toks, idx, arcs, mat, mlab, notes)))

    # ------------------------------------------------------------------
    def cases(self):
        rows = [
            ("tags: []", 0, r"an empty list is a value"),
            ("tags: null", 1, r"a null list is not"),
            ("sku: Z99  # never seen", 0, r"an unknown label is still valued:\\only its content is unknown"),
            ("quantity: 2  # hidden", 3, r"hidden on purpose: masked\\(we'll need this later)"),
        ]
        g = VGroup()
        for i, (code_, st, text) in enumerate(rows):
            y = 2.0 - 1.25 * i
            c = Mono(code_, font_size=26).move_to([-6.2, y, 0], aligned_edge=LEFT)
            t = state_tile(st, 0.48).move_to([-0.9, y, 0])
            d = label(text, font_size=28, color=GREY_A).move_to([-0.3, y, 0], aligned_edge=LEFT)
            g.add(VGroup(c, t, d))
        with self.voiceover(
            "A few more cases, to pin it down. <bookmark mark='a'/> An empty list is valued: there is a list, and "
            "it's empty. <bookmark mark='b'/> A null list is not. <bookmark mark='c'/> A label the model has never "
            "seen is still valued; only its content is unknown. <bookmark mark='d'/> And a value we hide on purpose "
            "gets the fourth state, masked, which is how targets work. We'll get there."
        ) as vo:
            for mark, r in zip("abcd", g):
                vo.wait_until(mark)
                self.play(FadeIn(r, shift=RIGHT * 0.15))
        self.wait(0.4)
        self.play(FadeOut(g))
