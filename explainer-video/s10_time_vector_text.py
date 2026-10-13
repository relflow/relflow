from __future__ import annotations

import numpy as np

from explainer import *  # noqa: F403
from videos.relflow.common import (
    D, Mono, box, code, heat_grid, label, mtex, note, proof_tag, real_tag, show_episode_card, vector_strip,
)


class TimeVectorText(VoiceoverScene):
    def construct(self):
        show_episode_card(self)
        self.d = D("time_vector_text")
        self.clock()
        self.parts()
        self.vector()
        self.text()
        self.choose()

    # ------------------------------------------------------------------
    def clock(self):
        dp = self.d["dateparts"]
        r = 2.2
        center = LEFT * 2.6 + DOWN * 0.3
        ring = Circle(radius=r, color=GREY_D, stroke_width=2).move_to(center)
        pts = VGroup()
        labels = VGroup()
        for h, (s, c) in enumerate(dp["day_coords"]):
            # relflow's (sin, cos) pair, drawn with midnight at the top and time running clockwise
            p = center + r * np.array([s, c, 0])
            pts.add(Dot(p, radius=0.06, color=C.NUMBER))
            if h % 3 == 0:
                labels.add(label(f"{h:02d}:15", font_size=18, color=GREY_A).move_to(center + (r + 0.45) * np.array([s, c, 0])))
        ex = dp["examples"]["hour_of_day"]
        cols = [YELLOW, YELLOW, YELLOW, C.RF_TARGET]
        marks = VGroup(*[Dot(center + r * np.array([s, c, 0]), radius=0.11, color=col) for (s, c), col in zip(ex, cols)])
        near = Line(marks[0].get_center(), marks[2].get_center(), color=YELLOW, stroke_width=3)
        far = DashedLine(marks[2].get_center(), marks[3].get_center(), color=C.RF_TARGET, stroke_width=2.5)
        code_ = code('created_at=rf.DateParts(dateparts=["hour_of_day", "day_of_week"])', font_size=19)
        code_.to_corner(UR, buff=0.4)
        side = VGroup(
            label(r"hour of day $\to (\sin, \cos)$ of $2\pi h/24$", font_size=24),
            label(rf"23:30 to 01:30: distance {dp['distance_2330_0130']:.2f}", font_size=24, color=YELLOW),
            label(rf"01:30 to 12:00: distance {dp['distance_0130_1200']:.2f}", font_size=24, color=C.RF_TARGET),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.25).move_to(RIGHT * 3.6 + DOWN * 0.3)
        tag = real_tag(r"coordinates from \texttt{model.encode}, relflow 0.2.3", corner=DL)
        assert dp["distance_2330_0130"] < 0.6 < 1.9 < dp["distance_0130_1200"]
        with self.voiceover(
            "Time has two faces. How long ago something happened is a number. But where it falls in a cycle, the "
            "hour, the weekday, the month, is a position on a circle. <bookmark mark='c'/> That's what DateParts "
            "encodes. <bookmark mark='r'/> Each part becomes a sine and a cosine, so the hours of a day go around a "
            "real circle. <bookmark mark='n'/> On a number line, 23:30 and 1:30 are twenty-two hours apart. On the "
            "circle they're neighbors, <bookmark mark='f'/> and noon is on the opposite side."
        ) as vo:
            vo.wait_until("c")
            self.play(FadeIn(code_), FadeIn(tag))
            vo.wait_until("r")
            self.play(Create(ring), FadeIn(side[0]))
            self.play(LaggedStart(*[FadeIn(p, scale=0.5) for p in pts], lag_ratio=0.06), FadeIn(labels), run_time=1.6)
            vo.wait_until("n")
            self.play(FadeIn(marks[:3]), Create(near), FadeIn(side[1]))
            vo.wait_until("f")
            self.play(FadeIn(marks[3]), Create(far), FadeIn(side[2]))
        self.wait(0.4)
        self.play(FadeOut(VGroup(ring, pts, labels, marks, near, far, code_, side, tag)))

    # ------------------------------------------------------------------
    def parts(self):
        dp = self.d["dateparts"]
        parts = [("day of year", 366), ("week of year", 53), ("month of year", 12), ("day of month", 31),
                 ("week of month", 6), ("day of week", 7), ("hour of day", 24), ("minute of hour", 60),
                 ("second of minute", 60)]
        g = VGroup(*[VGroup(label(n, font_size=24), label(rf"period {p}", font_size=20, color=GREY_A))
                     .arrange(RIGHT, buff=0.3) for n, p in parts]).arrange_in_grid(3, 3, buff=(0.8, 0.35))
        g.move_to(UP * 1.2)
        dw = dp["examples"]["day_of_week"]
        same_hour = label(r"23:30 Thursday and 00:30 Friday: neighbors by hour, different days by weekday",
                          font_size=24, color=YELLOW).next_to(g, DOWN, buff=0.6)
        assert dw[0] != dw[1]
        m = D("proofs")["metrics"]
        p44, p43 = m["P044"], m["P043"]
        proofs = VGroup(
            label(rf"P044: month from day of year, held-out years: {p44['periodic_accuracy'] * 100:.0f}\% accuracy,"
                  rf" {p44['permuted_accuracy'] * 100:.0f}\% with shuffled dates", font_size=22, color=GREY_A),
            label(rf"P043 (limited): too few parts can make the answer ambiguous: {p43['day_only_accuracy'] * 100:.0f}\%"
                  rf" $\to$ {p43['with_week_of_month_accuracy'] * 100:.0f}\% with week of month added", font_size=22,
                  color=GREY_A),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18).to_edge(DOWN, buff=0.55)
        tag = proof_tag(r"P043, P044", corner=UR)
        with self.voiceover(
            "There are nine parts to choose from, <bookmark mark='p'/> from the day of the year down to the second "
            "of the minute, and you list the ones your task needs. <bookmark mark='w'/> They answer different "
            "questions: half past eleven on Thursday and half past twelve on Friday are neighbors by the hour, and a "
            "day apart by the weekday. <bookmark mark='e'/> relflow's proofs test both sides: a model recovers the "
            "month from the day of the year on years it never trained on, and picking too few parts can leave the "
            "answer ambiguous."
        ) as vo:
            vo.wait_until("p")
            self.play(FadeIn(g, lag_ratio=0.08), run_time=1.4)
            vo.wait_until("w")
            self.play(FadeIn(same_hour))
            vo.wait_until("e")
            self.play(FadeIn(proofs, lag_ratio=0.3), FadeIn(tag))
        self.play(FadeOut(VGroup(g, same_hour, proofs, tag)))

    # ------------------------------------------------------------------
    def vector(self):
        v = self.d["vector"]
        c = code("embedding=rf.Vector(n_dim=4)", font_size=22).to_edge(UP, buff=0.6)
        vals = [0.12, -0.03, 0.44, 0.08]
        src = VGroup(*[label(f"{x:+.2f}", font_size=24) for x in vals]).arrange(DOWN, buff=0.12)
        src_box = SurroundingRectangle(src, buff=0.15, color=C.NUMBER, corner_radius=0.08)
        src_g = VGroup(src_box, src).move_to(LEFT * 4.2 + DOWN * 0.3)
        st = label(r"from another system:\\an embedding, a sensor window", font_size=22, color=GREY_A)
        st.next_to(src_g, DOWN, buff=0.3)
        lin = box(rf"linear\\{v['projection'][0]} $\to$ {v['projection'][1]}", color=C.EMBED, width=2.4, height=1.3,
                  font_size=24).move_to(LEFT * 0.6 + DOWN * 0.3)
        out = vector_strip(np.sin(np.arange(16) * 1.3), color=C.EMBED, cell=0.16).move_to(RIGHT * 2.6 + DOWN * 0.3)
        a1 = Arrow(src_g.get_right(), lin.get_left(), buff=0.15, color=GREY_B)
        a2 = Arrow(lin.get_right(), out.get_left(), buff=0.15, color=GREY_B)
        plus = label(r"+ state", font_size=24, color=GREY_A).next_to(out, RIGHT, buff=0.3)
        notes = VGroup(
            label(r"null vector: a state, not zeros", font_size=22, color=C.NULL_STATE),
            label(r"no normalizer: scale it upstream", font_size=22, color=GREY_A),
            label(r"wrong length: an error", font_size=22, color=RED_B),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18).to_corner(DR, buff=0.5)
        err = Mono(v["error"], font_size=16, color=RED_B).to_edge(DOWN, buff=0.25)
        tag = real_tag(r"real encodings, relflow 0.2.3", corner=UR)
        assert v["states"] == [0, 1] and "length 4" in v["error"]
        with self.voiceover(
            "Sometimes another system has already turned something into numbers: an embedding, or a window of sensor "
            "readings. <bookmark mark='v'/> That's a Vector: a fixed number of floats. <bookmark mark='p'/> relflow "
            "projects it to the model's width and adds the state. <bookmark mark='n'/> A missing vector gets the null "
            "state, not a row of zeros. relflow doesn't rescale vectors, so put them on a sensible scale upstream, "
            "and a vector of the wrong length is rejected."
        ) as vo:
            self.play(FadeIn(c), FadeIn(tag))
            vo.wait_until("v")
            self.play(FadeIn(src_g), FadeIn(st))
            vo.wait_until("p")
            self.play(GrowArrow(a1), FadeIn(lin), GrowArrow(a2), FadeIn(out), FadeIn(plus))
            vo.wait_until("n")
            self.play(FadeIn(notes, lag_ratio=0.3), FadeIn(err))
        self.play(FadeOut(VGroup(c, src_g, st, lin, out, a1, a2, plus, notes, err, tag)))

    # ------------------------------------------------------------------
    def text(self):
        t = self.d["text"]
        msgs = t["messages"]
        c = code('body=rf.Text(max_length=64, encoder_pooling="mean")', font_size=21).to_edge(UP, buff=0.45)
        frozen = box(r"frozen encoder\\" + rf"\texttt{{{t['model'].split('/')[1].replace('_', chr(92) + '_')}}}\\"
                     + rf"{t['frozen_params'] / 1e6:.1f}M weights, never trained", color=GREY_B, width=4.6, height=1.6,
                     font_size=20).move_to(LEFT * 3.6 + UP * 0.6)
        pooled = label(rf"mean-pooled\\{t['hidden']} numbers", font_size=22, color=GREY_A)
        proj = box(rf"learned projection\\{t['projection'][0]} $\to$ {t['projection'][1]}, + state", color=C.EMBED,
                   width=3.6, height=1.3, font_size=21).move_to(RIGHT * 3.2 + UP * 0.6)
        a = Arrow(frozen.get_right(), proj.get_left(), buff=0.15, color=GREY_B)
        pooled.next_to(a, UP, buff=0.1)
        sim = np.array(t["similarity"])
        grid = heat_grid(sim, C.EMBED, cell=0.55, vmin=0.5, vmax=1.0).move_to(RIGHT * 2.2 + DOWN * 2.2)
        nums = VGroup()
        for i in range(4):
            for j in range(4):
                nums.add(label(f"{sim[i, j]:.2f}", font_size=16, color=BACKGROUND if sim[i, j] > 0.8 else WHITE)
                         .move_to(grid[i][j]))
        names = VGroup(*[label(rf"{k + 1}. {m}", font_size=20, color=GREY_A) for k, m in enumerate(msgs)])
        names.arrange(DOWN, aligned_edge=LEFT, buff=0.15).next_to(grid, LEFT, buff=0.6)
        idx = VGroup(*[label(str(k + 1), font_size=18, color=GREY_A).next_to(grid[0][k], UP, buff=0.1) for k in range(4)])
        tag = real_tag(r"real frozen embeddings, cosine similarity", corner=DR)
        hi = sim[0, 1]
        lo = sim[0, 2]
        assert hi > lo and sim[1, 3] > sim[2, 3]
        with self.voiceover(
            "And text. <bookmark mark='f'/> A Text field runs each string through a small, frozen language model from "
            f"Hugging Face. By default it's a two-layer BERT with about {t['frozen_params'] / 1e6:.0f} million weights, "
            "and relflow never trains them. <bookmark mark='p'/> What relflow learns is how to use the result: a "
            "projection of the pooled vector, added to the state. <bookmark mark='s'/> Here are four real support "
            "messages through the frozen encoder. The two about a missing transfer are the closest pair, at "
            f"{hi:.2f}; the transfer and the password question are farther apart, at {lo:.2f}. "
            "<bookmark mark='l'/> Text is for meaning that's in the words. relflow doesn't fine-tune the encoder or "
            "generate text, and the encoder's files aren't stored in the checkpoint, so ship them with the model."
        ) as vo:
            self.play(FadeIn(c))
            vo.wait_until("f")
            self.play(FadeIn(frozen))
            vo.wait_until("p")
            self.play(GrowArrow(a), FadeIn(pooled), FadeIn(proj))
            vo.wait_until("s")
            self.play(FadeIn(names, lag_ratio=0.2), FadeIn(grid, lag_ratio=0.02), FadeIn(nums), FadeIn(idx), FadeIn(tag))
            vo.wait_until("l")
            self.play(Indicate(VGroup(grid[0][1], grid[1][0]), color=WHITE))
        self.play(FadeOut(VGroup(c, frozen, pooled, proj, a, grid, nums, names, idx, tag)))

    # ------------------------------------------------------------------
    def choose(self):
        rows = [
            (r"a recurring calendar position", r"\texttt{DateParts}"),
            (r"elapsed time, age, recency", r"derive a \texttt{Number}"),
            (r"a fixed-width vector from elsewhere", r"\texttt{Vector}"),
            (r"meaning in free text", r"\texttt{Text}"),
        ]
        g = VGroup(*[VGroup(label(a, font_size=28, color=GREY_A), label(b, font_size=28))
                     .arrange(RIGHT, buff=0.6) for a, b in rows])
        for r in g:
            r[1].move_to([2.6, r[1].get_y(), 0], aligned_edge=LEFT)
            r[0].move_to([2.2, r[0].get_y(), 0], aligned_edge=RIGHT)
        g.arrange(DOWN, buff=0.4)
        for r in g:
            r[1].set_x(2.6, LEFT)
            r[0].set_x(2.2, RIGHT)
        with self.voiceover(
            "So: a position in a cycle is DateParts, and elapsed time is a Number you derive. A vector from another "
            "system is a Vector, and meaning in free text is Text. Every one of them gets the same five states as "
            "every other field."
        ):
            self.play(FadeIn(g, lag_ratio=0.25), run_time=2.0)
        self.wait(0.5)
        self.play(FadeOut(g))
