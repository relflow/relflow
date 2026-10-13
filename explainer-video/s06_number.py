from __future__ import annotations

import math

import numpy as np

from explainer import *  # noqa: F403
from videos.relflow.common import (
    D, Mono, Plot, chip, label, mtex, note, real_tag, show_episode_card, state_tile, vector_strip,
)


class Numbers(VoiceoverScene):
    def construct(self):
        show_episode_card(self)
        self.n = D("number")
        self.standardize()
        self.fourier()
        self.why()
        self.scalar_lane()
        self.assemble()

    # ------------------------------------------------------------------
    def standardize(self):
        stream = self.n["stream"]
        z = mtex(r"z = \frac{x - \mu}{\sigma}", font_size=48).to_edge(UP, buff=0.6)
        x0 = label(r"a price: $x = 19.99$", font_size=32).next_to(z, LEFT, buff=1.2)
        lanes = VGroup()
        summaries = VGroup()
        pl = Plot((0, 120), (0, 1), width=6.2, height=0.5, x_ticks=[0, 20, 40, 60, 80, 100, 120], font_size=20, grid=False)
        rows = VGroup()
        for i, s in enumerate(stream):
            dots = VGroup(*[Dot(radius=0.035, color=C.NUMBER, fill_opacity=0.75) for _ in s["batch"]])
            row = VGroup(dots)
            rows.add(row)
            chip_ = chip(rf"$n={s['n']}$, $\bar x={s['batch_mean']:.1f}$, $s^2={s['batch_var']:.0f}$", color=C.NUMBER,
                         font_size=20)
            summaries.add(chip_)
        y0 = 1.3
        for i, (row, s) in enumerate(zip(rows, stream)):
            y = y0 - 0.48 * i
            for d, v in zip(row[0], s["batch"]):
                d.move_to([-6.0 + 6.2 * min(v, 120) / 120, y, 0])
            summaries[i].move_to([1.2, y, 0], aligned_edge=LEFT)
        axis = Line([-6.0, y0 - 0.48 * 6 + 0.1, 0], [0.2, y0 - 0.48 * 6 + 0.1, 0], color=GREY_B, stroke_width=1.5)
        ticks = VGroup(*[Mono(str(t), font_size=16, color=GREY_B).move_to([-6.0 + 6.2 * t / 120, axis.get_y() - 0.22, 0])
                         for t in (0, 40, 80, 120)])
        head_l = label(r"training batches arrive one at a time", font_size=24, color=GREY_A).move_to([-2.9, y0 + 0.55, 0])
        head_r = label(r"each one is kept only as a summary", font_size=24, color=GREY_A).next_to(summaries, UP, buff=0.3)
        head_r.align_to(summaries, LEFT).set_y(head_l.get_y())
        merge = mtex(r"\mu \leftarrow \mu + \delta\,\tfrac{n_b}{n},\quad "
                     r"\sigma^2 \leftarrow \tfrac{n_a\sigma_a^2 + n_b\sigma_b^2}{n} + \delta^2 \tfrac{n_a n_b}{n^2},\quad "
                     r"\delta = \bar x_b - \mu", font_size=30)
        merge.to_edge(DOWN, buff=0.45)
        last = stream[-1]
        check = label(rf"after {last['count']} values: running $\mu={last['mean']:.4f}$, exact {last['exact_mean']:.4f}; "
                      rf"running $\sigma^2={last['var']:.2f}$, exact {last['exact_var']:.2f}", font_size=24, color=C.VALUED)
        check.next_to(merge, UP, buff=0.25)
        for s in stream:
            assert abs(s["mean"] - s["exact_mean"]) < 1e-3
        tag = real_tag(r"relflow's \texttt{GlobalOnlineNormalizer}, fed six real batches", corner=UR)
        with self.voiceover(
            "Start with one number: a price, say, nineteen ninety-nine. <bookmark mark='z'/> Networks like their inputs "
            "near zero, with a spread of about one, so the first step is to standardize: subtract a mean, divide by a "
            "standard deviation. <bookmark mark='s'/> But relflow streams its data. It never holds the whole column at "
            "once. <bookmark mark='b'/> So each training batch is boiled down to three numbers: how many values, their "
            "mean, and their variance. <bookmark mark='m'/> And summaries like these merge exactly. The new mean moves "
            "toward the batch's mean in proportion to its size, and the variance gets a correction for how far apart "
            "the two means were. <bookmark mark='c'/> After six batches, the running statistics match the exact mean "
            "and variance of every value seen, without keeping a single one of them."
        ) as vo:
            self.play(FadeIn(x0))
            vo.wait_until("z")
            self.play(Write(z))
            vo.wait_until("s")
            self.play(FadeIn(head_l), Create(axis), FadeIn(ticks))
            vo.wait_until("b")
            for i in range(len(stream)):
                self.play(LaggedStart(*[FadeIn(d, shift=DOWN * 0.15) for d in rows[i][0]], lag_ratio=0.01), run_time=0.45)
                self.play(TransformFromCopy(rows[i][0], summaries[i]), *([FadeIn(head_r)] if i == 0 else []),
                          run_time=0.45)
                if i == 2:
                    vo.wait_until("m")
                    self.play(Write(merge), FadeIn(tag), run_time=1.4)
            vo.wait_until("c")
            self.play(FadeIn(check, shift=UP * 0.1))
        self.play(FadeOut(VGroup(rows, summaries, axis, ticks, head_l, head_r, merge, check, x0, tag, z)))

    # ------------------------------------------------------------------
    def fourier(self):
        w = np.array(self.n["weights"])
        assert len(w) == 13
        zs = np.linspace(-2, 2, 1200)
        rows = VGroup()
        labels = VGroup()
        width, h = 7.6, 0.34
        x_left = -5.0
        for k, om in enumerate(w):
            y = 2.75 - 0.43 * k
            pts = [[x_left + width * (zz + 2) / 4, y + h * 0.5 * math.sin(om * zz), 0] for zz in zs]
            curve = VMobject().set_points_as_corners(pts).set_stroke(C.NUMBER, 1.6 if k < 10 else 1.0)
            rows.add(curve)
            exp = k - 8
            labels.add(mtex(rf"\pi \cdot 2^{{{exp}}}", font_size=20, color=GREY_A).move_to([x_left - 0.75, y, 0]))
        z0 = 0.7
        xz = x_left + width * (z0 + 2) / 4
        probe = DashedLine([xz, 3.1, 0], [xz, 2.75 - 0.43 * 12 - 0.3, 0], color=YELLOW, stroke_width=2)
        dots = VGroup(*[Dot([xz, 2.75 - 0.43 * k + h * 0.5 * math.sin(om * z0), 0], radius=0.05, color=YELLOW)
                        for k, om in enumerate(w)])
        zl = mtex(r"z = 0.7", font_size=28, color=YELLOW).next_to(probe, UP, buff=0.05)
        axis_l = VGroup(mtex("-2", font_size=20, color=GREY_B).move_to([x_left, -2.85, 0]),
                        mtex("0", font_size=20, color=GREY_B).move_to([x_left + width / 2, -2.85, 0]),
                        mtex("2", font_size=20, color=GREY_B).move_to([x_left + width, -2.85, 0]),
                        mtex("z", font_size=24, color=GREY_A).move_to([x_left + width + 0.35, -2.85, 0]))
        feats = np.concatenate([np.sin(w * z0), np.cos(w * z0)])
        vec = vector_strip(feats, color=C.NUMBER, cell=0.16, gap=0.025, vmax=1.0).move_to(RIGHT * 4.6 + DOWN * 0.1)
        vec_l = label(r"$\sin(\omega_k z),\ \cos(\omega_k z)$\\26 numbers", font_size=24, color=C.NUMBER)
        vec_l.next_to(vec, UP, buff=0.2)
        freq_l = label(r"13 frequencies $\omega_k$,\\each double the last", font_size=24, color=GREY_A)
        freq_l.next_to(vec, DOWN, buff=0.3)
        tag = real_tag(r"frequencies read from the real Number embedder: $\pi/256$ to $16\pi$")
        with self.voiceover(
            "Now the standardized value, z, becomes a vector. <bookmark mark='f'/> relflow evaluates sines and "
            "cosines of z at thirteen frequencies, each double the last, from pi over 256 up to sixteen pi. "
            "<bookmark mark='l'/> The slow ones change gently: they say roughly where z is. <bookmark mark='h'/> The "
            "fast ones wiggle many times per unit: they tell apart values that are very close. "
            "<bookmark mark='p'/> Read them all off at one value of z, <bookmark mark='v'/> and you get twenty-six "
            "numbers: a fingerprint that's different for every value."
        ) as vo:
            vo.wait_until("f")
            self.play(LaggedStart(*[Create(r) for r in rows], lag_ratio=0.08), FadeIn(labels), FadeIn(axis_l),
                      FadeIn(tag), run_time=2.4)
            vo.wait_until("l")
            self.play(*[Indicate(rows[k], color=WHITE, scale_factor=1.0) for k in range(4)], run_time=1.0)
            vo.wait_until("h")
            self.play(*[Indicate(rows[k], color=WHITE, scale_factor=1.0) for k in range(10, 13)], run_time=1.0)
            vo.wait_until("p")
            self.play(Create(probe), FadeIn(zl), LaggedStart(*[FadeIn(d, scale=0.5) for d in dots], lag_ratio=0.06))
            vo.wait_until("v")
            self.play(TransformFromCopy(dots, vec.cells[:13]), run_time=1.0)
            self.play(FadeIn(vec), FadeIn(vec_l), FadeIn(freq_l))
        self.play(FadeOut(VGroup(rows, labels, probe, dots, zl, axis_l, vec, vec_l, freq_l, tag)))

    # ------------------------------------------------------------------
    def why(self):
        w = np.array(self.n["weights"])
        zs = np.linspace(-2, 2, 600)
        target = np.exp(-((zs - 0.5) ** 2) / (2 * 0.3**2))
        lin = np.stack([np.ones_like(zs), zs], 1)
        four = np.concatenate([lin, np.sin(np.outer(zs, w)), np.cos(np.outer(zs, w))], 1)
        fit_lin = lin @ np.linalg.lstsq(lin, target, rcond=None)[0]
        fit_four = four @ np.linalg.lstsq(four, target, rcond=None)[0]
        err = float(np.sqrt(np.mean((fit_four - target) ** 2)))
        assert err < 0.02
        pl = Plot((-2, 2), (-0.3, 1.2), width=8.5, height=3.6, x_ticks=[-2, -1, 0, 1, 2], y_ticks=[0, 0.5, 1],
                  font_size=22).move_to(DOWN * 0.5)
        tgt = pl.line(zs, target, GREY_A, 5)
        l1 = pl.line(zs, fit_lin, RED, 4)
        l2 = pl.line(zs, fit_four, C.NUMBER, 4)
        leg = VGroup(
            VGroup(Line(ORIGIN, RIGHT * 0.5, color=GREY_A, stroke_width=5), label(r"a narrow effect around $z=0.5$", font_size=24)),
            VGroup(Line(ORIGIN, RIGHT * 0.5, color=RED, stroke_width=4), label(r"best straight line in $z$", font_size=24)),
            VGroup(Line(ORIGIN, RIGHT * 0.5, color=C.NUMBER, stroke_width=4),
                   label(r"best weighted sum of the 26 features", font_size=24)),
        )
        for g in leg:
            g.arrange(RIGHT, buff=0.2)
        leg.arrange(DOWN, aligned_edge=LEFT, buff=0.15).to_edge(UP, buff=0.5)
        cap = note(r"least-squares fits, computed for this frame", font_size=20).to_corner(DR, buff=0.25)
        with self.voiceover(
            "Why bother? <bookmark mark='t'/> Suppose the effect you need to learn is narrow: something special happens "
            "right around one value. <bookmark mark='l'/> From z alone, a linear layer can only tilt a line; carving out "
            "a bump takes many layers of work. <bookmark mark='f'/> But a plain weighted sum of the sines and cosines "
            "already traces the bump closely. The features hand the network the building blocks of sharp effects, "
            "from the very first layer."
        ) as vo:
            vo.wait_until("t")
            self.play(Create(pl), FadeIn(leg[0]), Create(tgt), FadeIn(cap))
            vo.wait_until("l")
            self.play(FadeIn(leg[1]), Create(l1))
            vo.wait_until("f")
            self.play(FadeIn(leg[2]), Create(l2), run_time=1.6)
        self.play(FadeOut(VGroup(pl, tgt, l1, l2, leg, cap)))

    # ------------------------------------------------------------------
    def scalar_lane(self):
        w = np.array(self.n["weights"])
        pl = Plot((-6, 6), (-1.4, 1.4), width=8.0, height=3.0, x_ticks=[-6, -3, 0, 3, 6], y_ticks=[-1, 0, 1],
                  font_size=22).move_to(LEFT * 2.7 + DOWN * 0.6)
        zs = np.linspace(-6, 6, 800)
        s = pl.line(zs, np.sin(w[7] * zs), C.NUMBER, 3.5)
        ruler = pl.line(zs, zs / 4.5, YELLOW, 4)
        box = Rectangle(width=pl.c2p(2, 0)[0] - pl.c2p(-2, 0)[0], height=3.0, stroke_width=0, fill_color=GREY_B,
                        fill_opacity=0.12).move_to(pl.c2p(0, 0))
        seen_l = label(r"values seen in training", font_size=22, color=GREY_A).next_to(box, UP, buff=0.1)
        lab_s = label(r"a periodic feature repeats:\\it can't tell 1 from 5", font_size=23, color=C.NUMBER)
        lab_r = label(r"the last lane is $z$ itself:\\a straight ruler for scale", font_size=23, color=YELLOW)
        VGroup(lab_s, lab_r).arrange(DOWN, aligned_edge=LEFT, buff=0.5).next_to(pl, RIGHT, buff=0.35)
        token = vector_strip(np.r_[np.random.default_rng(2).normal(0, 0.6, 15), 1.6], color=GREY_A, cell=0.2, gap=0.03,
                             vmax=1.6).to_edge(RIGHT, buff=0.45)
        token.cells[-1].set_fill(YELLOW, 1)
        tl = label(r"the token\\($d_{\mathrm{model}}$ wide)", font_size=22, color=GREY_A).next_to(token, UP, buff=0.15)
        arrow = Arrow(lab_r.get_right(), token.cells[-1].get_left(), buff=0.1, color=YELLOW, stroke_width=3,
                      max_tip_length_to_length_ratio=0.15)
        with self.voiceover(
            "Periodic features have one weakness. <bookmark mark='o'/> Outside the range seen in training, they simply "
            "repeat. A sine wave has no idea that five is bigger than one. <bookmark mark='r'/> So relflow keeps one "
            "more lane. The very last coordinate of a Number's vector is z itself, untouched: a straight ruler for "
            "scale, which later layers can add and multiply, even past the values they were trained on."
        ) as vo:
            self.play(Create(pl), FadeIn(box), FadeIn(seen_l))
            vo.wait_until("o")
            self.play(Create(s), FadeIn(lab_s), run_time=1.4)
            vo.wait_until("r")
            self.play(Create(ruler), FadeIn(lab_r))
            self.play(FadeIn(token), FadeIn(tl), GrowArrow(arrow))
        self.play(FadeOut(VGroup(pl, box, seen_l, s, ruler, lab_s, lab_r, token, tl, arrow)))

    # ------------------------------------------------------------------
    def assemble(self):
        bound = self.n["max_fourier_input"]
        f = mtex(r"\text{token} = E_{\text{state}}[s] + \mathrm{GELU}\big(W\,[\sin \omega z,\ \cos \omega z]\big) + "
                 r"w\,z, \quad \text{last lane} \leftarrow z", font_size=34).move_to(UP * 2.2)
        clamp = mtex(rf"|z| \le \frac{{10^4}}{{16\pi}} \approx {bound:.0f}", font_size=36).move_to(UP * 0.5 + LEFT * 3.2)
        clamp_l = label(r"extreme inputs are clamped, so the\\fastest sine never exceeds $10^4$ radians", font_size=24,
                        color=GREY_A).next_to(clamp, DOWN, buff=0.25)
        loss = mtex(r"\text{loss} = \frac{|\hat x - x|}{\sigma}", font_size=40).move_to(UP * 0.5 + RIGHT * 3.2)
        loss_l = label(r"decoding: predict a state and a value;\\errors are measured in standard deviations,\\"
                       r"so dollars and days weigh alike", font_size=24, color=GREY_A).next_to(loss, DOWN, buff=0.25)
        assert abs(bound - 1e4 / (16 * math.pi)) < 1e-3
        tag = real_tag(r"bound read from the embedder's buffer")
        with self.voiceover(
            "Put together, a Number's token is the vector for its state, plus a learned projection of the Fourier "
            "features through a GELU, plus a straight line in z, with the last lane set to z itself. "
            "<bookmark mark='c'/> One safety rail: a value so extreme that the fastest sine would pass ten thousand "
            "radians, about 199 standard deviations out, gets clamped. <bookmark mark='d'/> Decoding runs the other "
            "way. A Number's decoder predicts a state and a value, and its error is measured in standard deviations, "
            "so a field in dollars and a field in days pull on the model equally."
        ) as vo:
            self.play(Write(f), run_time=2.5)
            vo.wait_until("c")
            self.play(FadeIn(clamp), FadeIn(clamp_l), FadeIn(tag))
            vo.wait_until("d")
            self.play(FadeIn(loss), FadeIn(loss_l))
        self.wait(0.5)
        self.play(FadeOut(VGroup(f, clamp, clamp_l, loss, loss_l, tag)))
