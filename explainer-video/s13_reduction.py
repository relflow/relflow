from __future__ import annotations

import numpy as np

from explainer import *  # noqa: F403
from videos.relflow.common import (
    D, Mono, Plot, chip, label, mtex, note, real_tag, schematic_tag, show_episode_card, tile, vector_strip,
)
from videos.relflow.s11_branch import FIELDS


class Reduction(VoiceoverScene):
    def construct(self):
        show_episode_card(self)
        self.c = D("counting")["arms"]
        self.learned_queries()
        self.average()
        self.experiment()
        self.mass_lane()
        self.result()
        self.options()

    # ------------------------------------------------------------------
    def learned_queries(self):
        live = 5
        toks = VGroup()
        for i, (n, col) in enumerate(FIELDS):
            for j in range(8):
                toks.add(tile(col if j < live else C.PADDED, 0.34, fill=0.85 if j < live else 0.3))
        toks.arrange(RIGHT, buff=0.08).move_to(DOWN * 1.4)
        mem_l = label(r"the encoded collection: 24 slots (15 live)", font_size=24, color=GREY_A).next_to(toks, DOWN, buff=0.25)
        q = tile(C.QUERY, 0.5).move_to(UP * 1.7)
        q_l = label(r"a learned query", font_size=26, color=C.QUERY).next_to(q, LEFT, buff=0.3)
        rng = np.random.default_rng(4)
        w = rng.dirichlet(np.ones(15) * 0.6)
        lines = VGroup()
        k = 0
        for idx in range(24):
            if idx % 8 >= live:
                continue
            lines.add(Line(q.get_bottom(), toks[idx].get_top(), color=C.ATTN, stroke_width=1 + 14 * w[k],
                           stroke_opacity=0.25 + 0.75 * w[k] / w.max()))
            k += 1
        out = vector_strip(rng.normal(0, 1, 10), color=C.MASS, cell=0.17, direction=RIGHT).next_to(q, RIGHT, buff=1.6)
        out_l = label(r"one summary vector per query", font_size=24, color=C.MASS).next_to(out, UP, buff=0.2)
        arr = Arrow(q.get_right(), out.get_left(), buff=0.15, color=GREY_B, stroke_width=3)
        k_l = note(r"(\texttt{n\_outputs} queries: 1 by default, 4 in the medium preset)", font_size=22).next_to(out, DOWN, buff=0.3)
        tag = schematic_tag(corner=UR)
        with self.voiceover(
            "After its encoder, a branch still holds one token per field and slot: twenty-four of them, for an order with "
            "room for eight items of three fields each. "
            "<bookmark mark='p'/> The parent doesn't want twenty-four. It wants a summary. <bookmark mark='q'/> relflow's "
            "default reduction is cross-attention from learned queries: a few vectors, trained like any other weights, "
            "<bookmark mark='a'/> each of which asks the collection a question, <bookmark mark='o'/> and comes back "
            "with a weighted average of the answers. One summary vector per query."
        ) as vo:
            self.play(FadeIn(toks, lag_ratio=0.02), FadeIn(mem_l), FadeIn(tag))
            vo.wait_until("q")
            self.play(FadeIn(q, scale=0.6), FadeIn(q_l))
            vo.wait_until("a")
            self.play(LaggedStart(*[Create(l_) for l_ in lines], lag_ratio=0.05), run_time=1.5)
            vo.wait_until("o")
            self.play(GrowArrow(arr), FadeIn(out), FadeIn(out_l), FadeIn(k_l))
        self.play(FadeOut(VGroup(toks, mem_l, q, q_l, lines, out, out_l, arr, k_l, tag)))

    # ------------------------------------------------------------------
    def average(self):
        f = mtex(r"\text{out} = \sum_i w_i\, v_i, \qquad w_i = \operatorname{softmax}_i(q \cdot k_i), \qquad "
                 r"\sum_i w_i = 1", font_size=40).to_edge(UP, buff=0.6)

        def bag(n):
            ts = VGroup(*[tile(C.NUMBER, 0.62) for _ in range(n)]).arrange(RIGHT, buff=0.12)
            vals = VGroup(*[Mono("1.25", font_size=18, color=BACKGROUND).move_to(t) for t in ts])
            ws = VGroup(*[mtex(rf"\tfrac{{1}}{{{n}}}", font_size=26, color=C.ATTN).next_to(t, UP, buff=0.12) for t in ts])
            return VGroup(ts, vals, ws)

        one, six = bag(1), bag(6)
        VGroup(one, six).arrange(DOWN, buff=1.0, aligned_edge=LEFT).move_to(LEFT * 2.2 + DOWN * 0.4)
        l1 = label(r"one copy", font_size=26, color=GREY_A).next_to(one, LEFT, buff=0.5)
        l6 = label(r"six copies", font_size=26, color=GREY_A).next_to(six, LEFT, buff=0.5)
        l6.align_to(l1, RIGHT)
        r1 = mtex(r"\text{out} = 1.25\,v", font_size=34).next_to(one, RIGHT, buff=0.8)
        r6 = mtex(r"\text{out} = 6 \cdot \tfrac{1}{6} \cdot 1.25\,v = 1.25\,v", font_size=34).next_to(six, RIGHT, buff=0.8)
        r1.align_to(r6, LEFT)
        same = label(r"identical outputs: the count is gone", font_size=32, color=RED).to_edge(DOWN, buff=0.6)
        with self.voiceover(
            "Look closely at those weights. <bookmark mark='s'/> Softmax makes every weight positive, and makes them "
            "add up to one. So the output is always an average of the values. <bookmark mark='b'/> And averages have "
            "a blind spot: they can't count. <bookmark mark='o'/> One token carrying 1.25 gets all the weight. "
            "<bookmark mark='x'/> Six identical tokens split it, a sixth each, and add back up to exactly the same "
            "output. <bookmark mark='g'/> To the parent, one item and six items look identical."
        ) as vo:
            self.play(Write(f), run_time=1.8)
            vo.wait_until("o")
            self.play(FadeIn(one), FadeIn(l1))
            self.play(Write(r1))
            vo.wait_until("x")
            self.play(FadeIn(six), FadeIn(l6))
            self.play(Write(r6))
            vo.wait_until("g")
            self.play(FadeIn(same, shift=UP * 0.1))
        self.play(FadeOut(VGroup(f, one, six, l1, l6, r1, r6, same)))

    # ------------------------------------------------------------------
    def probe_plot(self):
        pl = Plot((0.5, 10.5), (0, 13.5), width=6.4, height=4.6, x_ticks=list(range(1, 11)), y_ticks=[0, 2.5, 5, 7.5, 10, 12.5],
                  font_size=22).move_to(LEFT * 3.3 + DOWN * 0.35)
        span = Rectangle(width=pl.c2p(10.5, 0)[0] - pl.c2p(6.5, 0)[0], height=4.6, stroke_width=0, fill_color=GREY_B,
                         fill_opacity=0.1).move_to((pl.c2p(6.5, 0) + pl.c2p(10.5, 13.5)) / 2)
        unseen = label(r"never trained on", font_size=20, color=GREY_A).next_to(span, UP, buff=0.08)
        xt = pl.x_title(r"copies of 1.25 in the bag", font_size=24)
        yt = label(r"predicted total", font_size=22, color=GREY_A).next_to(pl, UP, buff=0.15).align_to(pl, LEFT)
        truth = pl.line([0.5, 10.5], [0.625, 13.125], GREY_A, 3)
        truth = DashedLine(pl.c2p(0.5, 0.625), pl.c2p(10.5, 13.125), color=GREY_A, stroke_width=2.5)
        tl = label(r"true total", font_size=22, color=GREY_A).next_to(pl.c2p(10.3, 12.9), LEFT, buff=0.1).shift(UP * 0.15)
        return pl, VGroup(span, unseen), xt, yt, truth, tl

    def series(self, pl, arm, color, radius=0.065):
        ys = self.c[arm]["probes"]["pred"]
        xs = list(range(1, 11))
        dots = VGroup(*[pl.dot(x, y, color=color, radius=radius) for x, y in zip(xs, ys)])
        line = VMobject().set_points_as_corners([pl.c2p(x, y) for x, y in zip(xs, ys)]).set_stroke(color, 2.5, opacity=0.8)
        return VGroup(line, dots)

    def experiment(self):
        c = self.c
        pm, pn = c["mean"]["probes"]["pred"], c["attention_no_mass"]["probes"]["pred"]
        assert max(pm) - min(pm) < 1e-3 and max(pn) - min(pn) < 1e-3
        setup = VGroup(
            label(r"a real experiment", font_size=28),
            label(r"each record: a bag of 1 to 6 identical amounts", font_size=24, color=GREY_A),
            label(r"target: their total (hidden)", font_size=24, color=GREY_A),
            label(r"branch attention off, so only the reduction can count", font_size=24, color=GREY_A),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12).move_to(RIGHT * 3.6 + UP * 2.4)
        pl, span, xt, yt, truth, tl = self.probe_plot()
        s_mean = self.series(pl, "mean", C.PADDED)
        s_nm = self.series(pl, "attention_no_mass", RED)
        leg = VGroup(
            VGroup(Dot(color=C.PADDED), Mono("rf.Mean()", font_size=20, color=GREY_A),
                   label(rf"always {pm[0]:.2f}", font_size=22, color=GREY_A)),
            VGroup(Dot(color=RED), label(r"attention, extra lanes removed", font_size=22, color=RED),
                   label(rf"always {pn[0]:.2f}", font_size=22, color=RED)),
        )
        for g in leg:
            g.arrange(RIGHT, buff=0.15)
        leg.arrange(DOWN, aligned_edge=LEFT, buff=0.2).next_to(setup, DOWN, buff=0.5).align_to(setup, LEFT)
        abl = note(r"(an ablation: relflow's mass and count lanes deleted from the model)", font_size=18)
        abl.next_to(leg, DOWN, buff=0.1).align_to(leg, LEFT)
        tag = real_tag(r"three real training runs, 1{,}280 steps each, relflow 0.2.3")
        with self.voiceover(
            "Let's watch that happen, for real. <bookmark mark='s'/> Each training record holds a bag of identical "
            "amounts, between one and six of them, and the hidden target is their total. Branch attention is switched "
            "off, so only the reduction can do the counting. <bookmark mark='p'/> Then we ask each trained model about "
            "one copy of 1.25, two copies, and so on, up to ten. <bookmark mark='m'/> First, the plain average "
            f"reduction, rf dot Mean. It answers {pm[0]:.2f}, every time. <bookmark mark='n'/> Next, attention, with two of "
            f"relflow's lanes removed. {pn[0]:.2f}, every time. Both have learned to guess about three and a half times the "
            "amount, the average count, because the count itself never reaches them."
        ) as vo:
            vo.wait_until("s")
            self.play(FadeIn(setup, lag_ratio=0.2), FadeIn(tag))
            vo.wait_until("p")
            self.play(Create(pl), FadeIn(span), FadeIn(xt), FadeIn(yt), Create(truth), FadeIn(tl))
            vo.wait_until("m")
            self.play(Create(s_mean[0]), LaggedStart(*[FadeIn(d, scale=0.5) for d in s_mean[1]], lag_ratio=0.08),
                      FadeIn(leg[0]))
            vo.wait_until("n")
            self.play(Create(s_nm[0]), LaggedStart(*[FadeIn(d, scale=0.5) for d in s_nm[1]], lag_ratio=0.08),
                      FadeIn(leg[1]), FadeIn(abl))
        self.plot = VGroup(pl, span, xt, yt, truth, tl, s_mean, s_nm)
        self.side = VGroup(setup, leg, abl, tag)

    # ------------------------------------------------------------------
    def mass_lane(self):
        self.plot.save_state()
        self.play(FadeOut(self.side), self.plot.animate.set_opacity(0.15), run_time=0.8)
        f = MathTex(r"\text{out}_j", r"=", r"\mathrm{LN}\big(\text{attention}_j\big)", r"+",
                    r"W_j\,\frac{\sum_i e_i}{\sqrt{C}}", r"+", r"\frac{n}{C}\, d_j", font_size=42).move_to(UP * 1.6)
        f[4].set_color(C.MASS)
        f[6].set_color(C.MASS)
        b1 = Brace(f[4], DOWN, color=C.MASS)
        t1 = label(r"mass: the sum of the branch's\\input tokens, projected", font_size=24, color=C.MASS).next_to(b1, DOWN, buff=0.1)
        t1.shift(LEFT * 0.9)
        b2 = Brace(f[6], DOWN, color=C.MASS)
        t2 = label(r"count: present tokens $n$\\over capacity $C$", font_size=24, color=C.MASS).next_to(b2, DOWN, buff=0.1)
        t2.shift(RIGHT * 0.9)
        zero = label(r"$W_j$ and $d_j$ start at zero: the reduction begins as pure attention,\\"
                     r"and learns how much of each lane to use", font_size=26, color=GREY_A).to_edge(DOWN, buff=0.9)
        with self.voiceover(
            "So relflow's reduction adds two lanes that softmax can't provide. <bookmark mark='m'/> The first is mass: "
            "the sum of the branch's input tokens, a sum, not an average, scaled down by the square root of the "
            "branch's capacity, and sent through a learned projection. <bookmark mark='c'/> The second is a count: the "
            "number of present tokens, divided by the capacity, times a learned direction. <bookmark mark='z'/> Both are "
            "added to every summary vector, and both start at exactly zero, so the reduction begins life as pure "
            "attention and learns how much of each lane to use."
        ) as vo:
            self.play(Write(f[:3]), run_time=1.2)
            vo.wait_until("m")
            self.play(Write(f[3:5]), GrowFromCenter(b1), FadeIn(t1))
            vo.wait_until("c")
            self.play(Write(f[5:]), GrowFromCenter(b2), FadeIn(t2))
            vo.wait_until("z")
            self.play(FadeIn(zero, shift=UP * 0.1))
        self.play(FadeOut(VGroup(f, b1, t1, b2, t2, zero)), Restore(self.plot), run_time=0.8)

    # ------------------------------------------------------------------
    def result(self):
        pl = self.plot[0]
        a = self.c["attention"]
        ys = a["probes"]["pred"]
        assert all(abs(y - 1.25 * k) < 0.12 for k, y in zip(range(1, 7), ys[:6]))
        s_att = self.series(pl, "attention", C.MASS, radius=0.075)
        vals = VGroup(*[label(f"{y:.2f}", font_size=18, color=C.MASS).next_to(pl.c2p(k, y), UL, buff=0.05)
                        for k, y in zip(range(1, 7), ys[:6])])
        leg = VGroup(Dot(color=C.MASS), label(r"relflow's reduction, mass and count lanes on", font_size=24, color=C.MASS))
        leg.arrange(RIGHT, buff=0.15).move_to(RIGHT * 3.5 + UP * 3.0)
        rm = VGroup(
            label(r"error on new bags of 1--6:", font_size=24, color=GREY_A),
            label(rf"\texttt{{Mean}}: {self.c['mean']['seen']['rmse']:.2f}", font_size=24, color=GREY_A),
            label(rf"no lanes: {self.c['attention_no_mass']['seen']['rmse']:.2f}", font_size=24, color=RED),
            label(rf"with lanes: {a['seen']['rmse']:.3f}", font_size=24, color=C.MASS),
            label(rf"(guessing the average total: {a['seen']['baseline_rmse']:.2f})", font_size=20, color=GREY_B),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12).next_to(leg, DOWN, buff=0.45).align_to(leg, LEFT)
        flat = label(r"past six copies it flattens:\\a learned sum, not an exact one\\(P002: a bounded extension)",
                     font_size=24, color=GREY_A)
        flat.next_to(rm, DOWN, buff=0.45).align_to(leg, LEFT)
        doc = label(r"relflow's docs: exact business totals\\belong in preprocessing", font_size=22, color=GREY_B)
        doc.next_to(flat, DOWN, buff=0.25).align_to(leg, LEFT)
        with self.voiceover(
            f"And with those lanes switched on, the same model counts. <bookmark mark='v'/> {ys[0]:.2f}, {ys[1]:.2f}, "
            f"{ys[2]:.2f}, all the way to {ys[5]:.2f} for six copies: right on the line, across the whole range it was "
            f"trained on. <bookmark mark='r'/> Its typical error on new bags drops from about "
            f"{self.c['mean']['seen']['rmse']:.1f} to {a['seen']['rmse']:.2f}. <bookmark mark='f'/> "
            "Past six copies, though, our model flattens out. relflow's own proof of this, P002, with a different "
            "setup, does stretch somewhat beyond its training lengths, and calls that a bounded extension, not exact "
            "arithmetic. <bookmark mark='d'/> The documentation says it plainly: if you need an exact business total, "
            "compute it in preprocessing."
        ) as vo:
            self.play(FadeIn(leg))
            vo.wait_until("v")
            self.play(Create(s_att[0]), LaggedStart(*[FadeIn(d, scale=0.5) for d in s_att[1][:6]], lag_ratio=0.15),
                      FadeIn(vals, lag_ratio=0.15), run_time=2)
            vo.wait_until("r")
            self.play(FadeIn(rm, lag_ratio=0.15))
            vo.wait_until("f")
            self.play(LaggedStart(*[FadeIn(d, scale=0.5) for d in s_att[1][6:]], lag_ratio=0.15), FadeIn(flat))
            vo.wait_until("d")
            self.play(FadeIn(doc))
        self.play(FadeOut(VGroup(self.plot, s_att, vals, leg, rm, flat, doc)))

    # ------------------------------------------------------------------
    def options(self):
        rows = [
            (Mono("rf.Attention(n_outputs=k)", font_size=24, color=C.MASS), r"$k$ learned summaries, plus mass and count",
             r"default; bounded size"),
            (Mono("rf.Mean()", font_size=24, color=GREY_A), r"one average of the present tokens",
             r"cheap; erases counts"),
            (Mono("reduction=None", font_size=24, color=C.RF_BRANCH), r"every encoded slot, passed up as is",
             r"keeps items apart; longer parent"),
        ]
        g = VGroup()
        for i, (c_, a, b) in enumerate(rows):
            y = 1.6 - 1.15 * i
            g.add(VGroup(c_.move_to([-6.4, y, 0], aligned_edge=LEFT),
                         label(a, font_size=26).move_to([-0.7, y + 0.2, 0], aligned_edge=LEFT),
                         label(b, font_size=24, color=GREY_A).move_to([-0.7, y - 0.22, 0], aligned_edge=LEFT)))
        head = label(r"what a branch can pass to its parent", font_size=32).to_edge(UP, buff=0.7)
        with self.voiceover(
            "So a branch has three ways to report to its parent. <bookmark mark='a'/> The default: a fixed number of "
            "learned summaries, with the mass and count lanes. <bookmark mark='m'/> rf dot Mean: one plain average. "
            "Cheap, but as we've just seen, it forgets how many. <bookmark mark='n'/> Or reduction equals None, which "
            "passes every encoded slot straight up, so items stay distinct, at the price of a longer sequence in the "
            "parent."
        ) as vo:
            self.play(FadeIn(head))
            for mark, r in zip("amn", g):
                vo.wait_until(mark)
                self.play(FadeIn(r, shift=RIGHT * 0.1))
        self.wait(0.4)
        self.play(FadeOut(VGroup(head, g)))
