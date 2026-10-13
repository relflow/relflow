from __future__ import annotations

import math

import numpy as np

from explainer import *  # noqa: F403
from videos.relflow.common import (
    D, Mono, Plot, chip, code, label, mtex, note, proof_tag, real_tag, show_episode_card, tile,
)


def ordinal(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def log_ticks(v):
    return rf"10^{{{int(round(math.log10(v)))}}}"


class QuantileEnum(VoiceoverScene):
    def construct(self):
        show_episode_card(self)
        self.d = D("quantile_enum")
        self.problem()
        self.ranks()
        self.features()
        self.tradeoff()
        self.enum()
        self.enum_evidence()

    # ------------------------------------------------------------------
    def problem(self):
        q = self.d["quantile"]
        vals = np.array(q["values_sample"])
        pl = Plot((0.5, 1e5), (0, 1), width=11.0, height=1.2, log_x=True, x_ticks=[1, 10, 100, 1e3, 1e4, 1e5],
                  y_ticks=[], x_fmt=log_ticks, font_size=24, grid=False).move_to(UP * 1.6)
        pl.y_axis.set_opacity(0)
        rng = np.random.default_rng(0)
        dots = VGroup(*[pl.dot(v, 0.15 + 0.7 * rng.random(), color=C.NUMBER, radius=0.03, opacity=0.6) for v in vals])
        xt = pl.x_title(r"4{,}000 real amounts (every 8th shown), log scale", font_size=22)
        # Number's view: standardized with the field's own running mean and standard deviation.
        zl = NumberLine(x_range=[-1, 40, 5], length=11.0, include_numbers=True, font_size=22, color=GREY_B)
        zl.move_to(DOWN * 1.7)
        z = (vals - q["number_mean"]) / q["number_std"]
        zdots = VGroup(*[Dot(zl.n2p(min(v, 40)), radius=0.03, color=C.NUMBER, fill_opacity=0.5)
                         .shift(UP * (0.12 + 0.4 * rng.random())) for v in z])
        zt = label(r"the same amounts as \texttt{rf.Number} sees them: $(x - \mu)/\sigma$", font_size=24, color=GREY_A)
        zt.next_to(zl, DOWN, buff=0.45)
        stats = label(rf"$\mu = {q['number_mean']:.0f}$, $\sigma = {q['number_std']:,.0f}$".replace(",", "{,}"),
                      font_size=28).next_to(zl, UP, buff=0.9).to_edge(RIGHT, buff=0.6)
        crowd = label(rf"{q['share_within_tenth_sd'] * 100:.0f}\% within $0.1\,\sigma$ of the mean", font_size=26,
                      color=YELLOW).next_to(zl.n2p(0), UP, buff=0.75).shift(RIGHT * 2.2)
        tag = real_tag(r"real field statistics from a trained model, relflow 0.2.3")
        with self.voiceover(
            "Some numbers are badly behaved. <bookmark mark='a'/> Here are four thousand amounts, spread over five "
            f"powers of ten, with {q['outliers']} enormous outliers. <bookmark mark='n'/> A Number field standardizes: "
            "subtract the mean, divide by the standard deviation. <bookmark mark='s'/> But those few outliers inflate "
            f"the standard deviation to about {q['number_std']:,.0f}. <bookmark mark='c'/> So "
            f"{q['share_within_tenth_sd'] * 100:.0f} percent of all the amounts land within a tenth of a standard "
            "deviation of the mean, squeezed into a sliver, while the outliers sit thirty or forty units away."
        ) as vo:
            vo.wait_until("a")
            self.play(Create(pl.x_axis), FadeIn(pl.x_labels), FadeIn(xt), FadeIn(tag))
            self.play(FadeIn(dots, lag_ratio=0.002), run_time=1.5)
            vo.wait_until("n")
            self.play(Create(zl), FadeIn(zt))
            vo.wait_until("s")
            self.play(FadeIn(stats))
            vo.wait_until("c")
            self.play(TransformFromCopy(dots, zdots), run_time=1.6)
            self.play(FadeIn(crowd))
        self.wait(0.4)
        self.play(FadeOut(VGroup(pl, dots, xt, zl, zdots, zt, stats, crowd, tag)))

    # ------------------------------------------------------------------
    def ranks(self):
        q = self.d["quantile"]
        probe, ranks = np.array(q["probe"]), np.array(q["ranks"])
        pl = Plot((0.5, 1e5), (0, 1), width=8.0, height=4.4, log_x=True, x_ticks=[1, 10, 100, 1e3, 1e4, 1e5],
                  y_ticks=[0, 0.25, 0.5, 0.75, 1.0], x_fmt=log_ticks, font_size=22).move_to(LEFT * 1.9 + DOWN * 0.35)
        curve = pl.line(probe, ranks, C.NUMBER, 4)
        xt = pl.x_title(r"amount", font_size=22)
        yt = label(r"\texttt{rf.Quantile}: the amount's percentile", font_size=24, color=C.NUMBER)
        yt.next_to(pl, UP, buff=0.2).align_to(pl, LEFT)
        rows = VGroup(*[
            VGroup(Mono(f"{m:>6,}", font_size=22), Mono("->", font_size=22, color=GREY_B),
                   Mono(f"{r:.2f}", font_size=22, color=C.NUMBER),
                   Mono(f"(Number: {zv:+.3f})", font_size=18, color=GREY_B)).arrange(RIGHT, buff=0.2)
            for m, r, zv in zip(q["marks"], q["mark_ranks"], q["mark_standardized"])
        ]).arrange(DOWN, aligned_edge=LEFT, buff=0.2).to_edge(RIGHT, buff=0.35).shift(DOWN * 0.2)
        dots = VGroup(*[pl.dot(m, r, color=YELLOW, radius=0.06) for m, r in zip(q["marks"], q["mark_ranks"])])
        digest = label(rf"a t-digest of the training values, kept as Arrow data: compression {q['compression']}",
                       font_size=22, color=GREY_A).to_edge(DOWN, buff=0.35)
        clip = label(r"beyond the largest value seen: 1", font_size=20, color=GREY_A).next_to(dots[-1], UP, buff=0.2)
        tag = real_tag(r"real percentiles from relflow's digest", corner=UR)
        assert abs(q["mark_ranks"][2] - 0.5) < 0.05 and q["mark_ranks"][-1] == 1.0
        with self.voiceover(
            "Quantile takes a different view. <bookmark mark='q'/> Instead of how far a value is from the mean, it "
            "asks where the value ranks among everything seen in training: its percentile, between zero and one. "
            "<bookmark mark='d'/> While training, each Quantile field keeps a compact summary of the distribution, a "
            "t-digest, stored as Arrow data rather than as rows. <bookmark mark='m'/> Now the typical amounts spread "
            f"out. Twenty sits at the {ordinal(round(q['mark_ranks'][1] * 100))} percentile; thirty-three, the median; a "
            f"hundred, the {ordinal(round(q['mark_ranks'][3] * 100))}. <bookmark mark='o'/> The outliers can't stretch the "
            "scale: they all end near one, and anything past the largest training value is clipped to one."
        ) as vo:
            vo.wait_until("q")
            self.play(Create(pl), FadeIn(xt), FadeIn(yt), FadeIn(tag))
            self.play(Create(curve), run_time=2.0)
            vo.wait_until("d")
            self.play(FadeIn(digest))
            vo.wait_until("m")
            self.play(LaggedStart(*[FadeIn(r, shift=LEFT * 0.1) for r in rows[:4]], lag_ratio=0.3),
                      FadeIn(dots[:4]), run_time=1.6)
            vo.wait_until("o")
            self.play(FadeIn(rows[4:]), FadeIn(dots[4:]), FadeIn(clip))
        self.wait(0.4)
        self.play(FadeOut(VGroup(pl, curve, xt, yt, rows, dots, digest, clip, tag)))

    # ------------------------------------------------------------------
    def features(self):
        q = self.d["quantile"]
        freqs = q["frequencies"]
        assert len(freqs) == 8 and abs(freqs[0] - math.pi) < 1e-5
        pl = Plot((0, 1), (-1, 1), width=9.0, height=3.4, x_ticks=[0, 0.25, 0.5, 0.75, 1], y_ticks=[-1, 0, 1],
                  font_size=22).move_to(DOWN * 0.6)
        xs = np.linspace(0, 1, 300)
        waves = VGroup(*[pl.line(xs, np.sin(f * xs), interpolate_color(ManimColor(C.NUMBER), ManimColor(C.EMBED), k / 7),
                                 2.2) for k, f in enumerate(freqs)])
        f = mtex(r"p \;\mapsto\; \big[\,p,\ \sin(k\pi p),\ \cos(k\pi p)\,\big]_{k=1}^{8}"
                 r"\;\to\; \text{linear} + \text{state}", font_size=34).to_edge(UP, buff=0.6)
        xt = pl.x_title(r"percentile $p$", font_size=22)
        with self.voiceover(
            "Then the percentile becomes a vector, much like a Number does: <bookmark mark='f'/> the percentile "
            "itself, plus eight sine and cosine pairs at frequencies of one to eight times pi, <bookmark mark='w'/> "
            "projected to the model's width and added to a learned vector for the value's state. Predictions come back "
            "in the original units: the decoder predicts a percentile, and relflow maps it back through the digest."
        ) as vo:
            vo.wait_until("f")
            self.play(Write(f), run_time=1.6)
            vo.wait_until("w")
            self.play(Create(pl), FadeIn(xt))
            self.play(LaggedStart(*[Create(w) for w in waves], lag_ratio=0.15), run_time=2.0)
        self.play(FadeOut(VGroup(f, pl, xt, waves)))

    # ------------------------------------------------------------------
    def tradeoff(self):
        m = D("proofs")["metrics"]["P077"]["input_tasks"]
        nc, qc = m["amount_number_contaminated"], m["amount_quantile_contaminated"]
        head = label(r"P077: predict an amount from a contaminated input", font_size=28).to_edge(UP, buff=0.7)
        rows = [(r"error, central values", nc["central_rmse"], qc["central_rmse"]),
                (r"error, the tails", nc["tail_rmse"], qc["tail_rmse"])]
        g = VGroup()
        for i, (t, a, b) in enumerate(rows):
            g.add(VGroup(label(t, font_size=26, color=GREY_A).move_to([-4.0, 1.0 - 0.9 * i, 0], aligned_edge=LEFT),
                         label(f"{a:.2f}", font_size=30, color=C.NUMBER if a < b else GREY_B).move_to([1.2, 1.0 - 0.9 * i, 0]),
                         label(f"{b:.2f}", font_size=30, color=C.CATEGORY if b < a else GREY_B).move_to([4.0, 1.0 - 0.9 * i, 0])))
        heads = VGroup(label(r"\texttt{rf.Number}", font_size=26, color=C.NUMBER).move_to([1.2, 1.9, 0]),
                       label(r"\texttt{rf.Quantile}", font_size=26, color=C.CATEGORY).move_to([4.0, 1.9, 0]))
        rule = VGroup(
            label(r"Quantile: when a value's position matters more than its size", font_size=26),
            label(r"Number: when magnitude matters, including beyond anything seen", font_size=26),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).to_edge(DOWN, buff=0.9)
        tag = proof_tag(r"P077, P068")
        assert qc["central_rmse"] < nc["central_rmse"] and qc["tail_rmse"] > nc["tail_rmse"]
        with self.voiceover(
            "So is Quantile simply better? relflow's proofs say: it depends. <bookmark mark='t'/> When the input is "
            "contaminated with outliers, the percentile view predicts typical values more accurately. "
            "<bookmark mark='l'/> But percentiles compress the extremes, so in the tails Number wins, and Quantile "
            "can never say anything about values beyond the range it learned. <bookmark mark='r'/> A rule of thumb: "
            "use Quantile when a value's position matters, and Number when its size does."
        ) as vo:
            self.play(FadeIn(head), FadeIn(heads), FadeIn(tag))
            vo.wait_until("t")
            self.play(FadeIn(g[0], shift=RIGHT * 0.1))
            vo.wait_until("l")
            self.play(FadeIn(g[1], shift=RIGHT * 0.1))
            vo.wait_until("r")
            self.play(FadeIn(rule, lag_ratio=0.3))
        self.play(FadeOut(VGroup(head, heads, g, rule, tag)))

    # ------------------------------------------------------------------
    def enum(self):
        e = self.d["enum"]
        c = code('tier=rf.Enum(values=("bronze", "silver", "gold"))', font_size=22).to_edge(UP, buff=0.6)
        def table(labels, counts, color, title_):
            rows = VGroup()
            for lab in labels:
                n = counts.get(lab) if counts else None
                rows.add(VGroup(tile(color, 0.34, fill=0.85 if (n is None or n) else 0.3),
                                Mono(lab, font_size=22, color=WHITE if (n is None or n) else GREY_B),
                                Mono("" if n is None else f"{n} seen", font_size=18, color=GREY_A))
                         .arrange(RIGHT, buff=0.2))
            rows.arrange(DOWN, aligned_edge=LEFT, buff=0.18)
            t = label(title_, font_size=24, color=color).next_to(rows, UP, buff=0.3)
            return VGroup(t, rows)
        enum_t = table(e["values"], e["counts"], C.CATEGORY, r"\texttt{Enum}: rows fixed by the declaration")
        cat_t = table(e["category_labels"], {}, C.CATEGORY, r"\texttt{Category}: rows only for labels seen")
        VGroup(enum_t, cat_t).arrange(RIGHT, buff=1.6, aligned_edge=UP).move_to(UP * 0.1)
        first, _, rest = e["error"].partition("; ")
        err = VGroup(label(r"an undeclared value is an error, not an unknown:", font_size=24),
                     Mono(first + ";", font_size=17, color=RED_B), Mono(rest, font_size=17, color=RED_B))
        err.arrange(DOWN, aligned_edge=LEFT, buff=0.12).to_edge(DOWN, buff=0.6)
        tag = real_tag(r"real model trained on bronze and silver only", corner=UR)
        assert e["counts"]["gold"] == 0 and e["rows"] == 3 and "gold" not in e["category_labels"]
        with self.voiceover(
            "Labels have a similar choice. A Category learns its vocabulary while it trains, so new labels can "
            "appear at any time. <bookmark mark='e'/> An Enum is for a set you can write down in advance: a customer "
            "tier, an order status. <bookmark mark='t'/> Here, a real model trained on data that only ever says bronze "
            "or silver. The Enum has all three rows from the start; gold has been seen zero times, but it exists, and "
            "can be predicted. <bookmark mark='c'/> A Category reading the same column only learned the two it saw. "
            "<bookmark mark='x'/> And an Enum is strict: a value outside the declaration, like platinum, is an error, "
            "in training and in prediction. Normalize spellings in a preprocessor, or declare a fallback label."
        ) as vo:
            self.play(FadeIn(c))
            vo.wait_until("e")
            self.play(Indicate(c, color=WHITE, scale_factor=1.03))
            vo.wait_until("t")
            self.play(FadeIn(enum_t, lag_ratio=0.1), FadeIn(tag))
            vo.wait_until("c")
            self.play(FadeIn(cat_t, lag_ratio=0.1))
            vo.wait_until("x")
            self.play(FadeIn(err, shift=UP * 0.1))
        self.play(FadeOut(VGroup(c, enum_t, cat_t, err, tag)))

    # ------------------------------------------------------------------
    def enum_evidence(self):
        text = VGroup(
            label(r"a declared label is structural support, not knowledge", font_size=30),
            label(r"P070: a class withheld from training is never predicted correctly at first;", font_size=24,
                  color=GREY_A),
            label(r"after training continues with its examples, it is, and the output head never changes", font_size=24,
                  color=GREY_A),
        ).arrange(DOWN, buff=0.3)
        tag = proof_tag(r"P070")
        with self.voiceover(
            "But declaring a label doesn't teach the model what it means. <bookmark mark='p'/> In relflow's proof "
            "P070, a class held out of training is never predicted correctly at first. Once training continues with "
            "its examples, it's learned, and the model's output head never had to change shape."
        ) as vo:
            self.play(FadeIn(text[0]))
            vo.wait_until("p")
            self.play(FadeIn(text[1:], lag_ratio=0.3), FadeIn(tag))
        self.wait(0.5)
        self.play(FadeOut(VGroup(text, tag)))
