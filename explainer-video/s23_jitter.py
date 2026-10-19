from __future__ import annotations

import numpy as np
from explainer import *  # noqa: F403
from videos.ml.relflow.common import C, D, Mono, label, mtex, show_episode_card
from videos.ml.relflow.current_visuals import heading, trace_tag


class Jitter(VoiceoverScene):
    def construct(self):
        show_episode_card(self)
        head = heading("Teach tolerance without moving the answer")
        d = D("current")["jitter"]
        samples = np.array(d["samples"])
        axis = NumberLine(
            x_range=[4.7, 5.3, 0.1],
            length=10,
            include_numbers=True,
            numbers_to_include=[4.8, 4.9, 5.0, 5.1, 5.2],
            font_size=24,
        ).move_to(UP * 0.6)
        origin = Dot(axis.n2p(d["base"]), color=C.NUMBER, radius=0.1)
        original = label(r"pristine input: 5.0", font_size=28, color=C.NUMBER).next_to(axis, UP, buff=0.8)
        # Deterministic vertical stacking visualizes the actual sampled distribution.
        counts = {}
        cloud = VGroup()
        for value in samples[::4]:
            bucket = int((value - 4.8) / 0.012)
            level = counts.get(bucket, 0)
            counts[bucket] = level + 1
            cloud.add(
                Dot(axis.n2p(value) + UP * (0.16 + 0.075 * level), radius=0.035, color=C.MASKED, fill_opacity=0.7)
            )
        target = Mono("target = 5.0", font_size=28, color=C.RF_TARGET).move_to(DOWN * 1.6)
        formula = mtex(r"J(v)=v[1+m(U_1-U_2)]+a(U_3-U_4)", font_size=34).move_to(DOWN * 2.5)
        tag = trace_tag("128 of 512 actual Jitter.apply samples; a=0.1, m=0.02, raw units")
        with self.voiceover(
            "Some measured values are estimates. A small change should leave their meaning intact. "
            '<bookmark mark="a"/> Jitter perturbs visible continuous inputs during training. These dots are actual samples around five. '
            '<bookmark mark="b"/> The answer stays exactly five. So do stored inputs and the values used to learn normalization statistics. '
            '<bookmark mark="c"/> Additive and multiplicative components each use the difference of two independent uniform draws. '
            "Small changes are more likely than extremes. Here the full allowed interval is four point eight to five point two. "
            "Choose noise in units that preserve the distinctions your task needs."
        ) as vo:
            self.play(FadeIn(head), Create(axis), FadeIn(origin), FadeIn(original), FadeIn(tag))
            vo.wait_until("a")
            self.play(LaggedStart(*[TransformFromCopy(origin, p) for p in cloud], lag_ratio=0.005), run_time=3)
            vo.wait_until("b")
            self.play(FadeIn(target))
            vo.wait_until("c")
            self.play(Write(formula))
        self.clear_scene()
        head = heading("Only the visible training lane moves")
        lanes = VGroup()
        for name, col in [("training", C.MASKED), ("validation", C.VALUED), ("prediction", C.RF_TARGET)]:
            line = Line(LEFT * 1.6, RIGHT * 2.8, color=col)
            title = Mono(name, font_size=25, color=col).next_to(line, LEFT, buff=0.45)
            point = Dot(line.get_center(), color=col, radius=0.09)
            lanes.add(VGroup(title, line, point))
        lanes.arrange(DOWN, aligned_edge=LEFT, buff=0.65).move_to(UP * 0.5)
        spaces = label(
            r"Number: raw or standardized\\Vector: supplied coordinates\\DateParts: sine/cosine pairs\\Text: pooled frozen-encoder features",
            font_size=25,
        ).move_to(DOWN * 2.0)
        with self.voiceover(
            "Only finite, valued training coordinates move. Nulls, padding, masks and target embeddings stay untouched. "
            '<bookmark mark="a"/> Validation and prediction are unperturbed. '
            '<bookmark mark="b"/> Number can use raw or standardized units. Vector, Date Parts and Text each have one continuous boundary. '
            "Jitter on a calendar pair does not move the timestamp or renormalize the pair. It is augmentation, not a persisted data transform."
        ) as vo:
            self.play(FadeIn(head), FadeIn(lanes))
            for shift in [0.5, -0.8, 0.3]:
                self.play(lanes[0][2].animate.shift(RIGHT * shift), run_time=0.5)
            vo.wait_until("a")
            self.play(Indicate(lanes[1:], color=C.VALUED))
            vo.wait_until("b")
            self.play(FadeIn(spaces))
        self.clear_scene()
