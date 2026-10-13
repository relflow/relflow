from __future__ import annotations

import numpy as np

from explainer import *  # noqa: F403
from videos.relflow.common import (
    D, Mono, Plot, chip, label, mtex, note, real_tag, schematic_tag, show_episode_card, state_tile,
)


def table_rows(n, filled, cell_w=1.7, cell_h=0.3, color=C.CATEGORY) -> VGroup:
    rows = VGroup()
    for i in range(n):
        on = i < filled
        rows.add(Rectangle(width=cell_w, height=cell_h, stroke_color=color if on else GREY_D, stroke_width=1,
                           fill_color=color, fill_opacity=0.55 if on else 0.05))
    rows.arrange(DOWN, buff=0.03)
    return rows


class Vocabulary(VoiceoverScene):
    def construct(self):
        show_episode_card(self)
        self.v = D("vocab")
        self.online()
        self.binding()
        self.growth()
        self.optimizer()
        self.unknown()

    # ------------------------------------------------------------------
    def online(self):
        old = VGroup(
            label(r"the classic recipe:", font_size=28),
            label(r"1. scan the whole dataset", font_size=26, color=GREY_A),
            label(r"2. collect every label", font_size=26, color=GREY_A),
            label(r"3. allocate a table of exactly the right size", font_size=26, color=GREY_A),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18).to_corner(UL, buff=0.7)
        cross = VGroup(*[Line(r.get_left() + LEFT * 0.08, r.get_right() + RIGHT * 0.08, color=RED, stroke_width=3)
                         for r in old[1:]])
        stream = ["grocery", "books", "grocery", "garden", "books", "toys"]
        chips = VGroup(*[chip(s, color=C.CATEGORY, font_size=24, mono=True) for s in stream]).arrange(RIGHT, buff=0.25)
        chips.move_to(DOWN * 0.4)
        vocab = VGroup()
        seen = []
        for s in stream:
            if s not in seen:
                seen.append(s)
        rows = VGroup(*[VGroup(Mono(str(i), font_size=22, color=GREY_B), Mono(s, font_size=24, color=C.CATEGORY))
                        .arrange(RIGHT, buff=0.3) for i, s in enumerate(seen)]).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        rows.to_corner(DR, buff=0.9)
        vt = label(r"the field's vocabulary", font_size=24, color=GREY_A).next_to(rows, UP, buff=0.25).align_to(rows, LEFT)
        with self.voiceover(
            "Categories are labels: SKUs, merchants, countries. <bookmark mark='c'/> The classic recipe is to scan the "
            "whole dataset first, collect every label, and allocate an embedding table of exactly the right size. "
            "<bookmark mark='x'/> relflow streams, so it can't scan first. <bookmark mark='o'/> Instead, every "
            "Category field learns its vocabulary online. The first time a training batch contains a new label, that "
            "label gets the next free row. Labels are never evicted and never renumbered."
        ) as vo:
            vo.wait_until("c")
            self.play(FadeIn(old, lag_ratio=0.2))
            vo.wait_until("x")
            self.play(Create(cross), old.animate.set_opacity(0.4))
            vo.wait_until("o")
            self.play(FadeIn(chips, lag_ratio=0.1), FadeIn(vt))
            k = 0
            for s, c in zip(stream, chips):
                if s in seen[:k]:
                    self.play(Indicate(c, color=GREY_A), Indicate(rows[seen.index(s)], color=WHITE), run_time=0.45)
                else:
                    self.play(TransformFromCopy(c, rows[k]), run_time=0.55)
                    k += 1
        self.play(FadeOut(VGroup(old, cross, chips, rows, vt)))

    # ------------------------------------------------------------------
    def binding(self):
        worker = RoundedRectangle(width=4.4, height=3.4, corner_radius=0.15, stroke_color=GREY_B, stroke_width=2,
                                  fill_color=GREY_B, fill_opacity=0.06).move_to(LEFT * 4.2 + DOWN * 0.2)
        wt = label(r"data worker (no model access)", font_size=24, color=GREY_A).next_to(worker, UP, buff=0.15)
        local = VGroup(*[Mono(f"{s} → {i}", font_size=24, color=C.CATEGORY) for i, s in
                         enumerate(["A12", "B07", "C33", "D01", "E90"])]).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        local.move_to(worker)
        lt = note(r"a dictionary local to this batch", font_size=22).next_to(local, DOWN, buff=0.2)
        model = RoundedRectangle(width=4.4, height=3.4, corner_radius=0.15, stroke_color=C.CATEGORY, stroke_width=2,
                                 fill_color=C.CATEGORY, fill_opacity=0.06).move_to(RIGHT * 4.2 + DOWN * 0.2)
        mt = label(r"the model's vocabulary", font_size=24, color=C.CATEGORY).next_to(model, UP, buff=0.15)
        glob = VGroup(*[Mono(f"{i:>2}  {s}", font_size=24, color=WHITE if s != "E90" else C.VALUED) for i, s in
                        [(16, "B07"), (17, "A12"), (18, "D01"), (19, "C33"), (20, "E90")]]).arrange(DOWN, aligned_edge=LEFT,
                                                                                                buff=0.12)
        glob.move_to(model)
        new = note(r"E90 is new: admitted as row 20", font_size=22, color=C.VALUED).next_to(glob, DOWN, buff=0.2)
        bind = Arrow(worker.get_right(), model.get_left(), buff=0.15, color=YELLOW, stroke_width=4)
        bl = label(r"bind, just before forward", font_size=26, color=YELLOW).next_to(bind, UP, buff=0.15)
        remap = VGroup(*[Mono(f"{a}→{b}", font_size=22, color=YELLOW) for a, b in
                         ((0, 17), (1, 16), (2, 19), (3, 18), (4, 20))]).arrange(DOWN, buff=0.08).next_to(bind, DOWN, buff=0.2)
        sch = schematic_tag(corner=UR)
        dist = note(r"with several GPUs, every rank gathers the new labels in rank order and admits them identically",
                    font_size=22).to_edge(DOWN, buff=0.35)
        with self.voiceover(
            "There's a subtlety. <bookmark mark='w'/> Batches are encoded in data-loader worker processes, and a worker "
            "can't touch the model. So a worker numbers labels with a dictionary that's local to its own batch: "
            "zero, one, two, and so on. <bookmark mark='m'/> Then, in the training process, just before the "
            "forward pass, relflow binds those local numbers to the model's vocabulary, <bookmark mark='a'/> admits "
            "any new labels, <bookmark mark='r'/> and remaps the batch's tensors. <bookmark mark='d'/> With several "
            "GPUs, every rank gathers the batch's new labels in rank order, so all of them admit the same labels in "
            "the same order."
        ) as vo:
            vo.wait_until("w")
            self.play(Create(worker), FadeIn(wt), FadeIn(local, lag_ratio=0.1), FadeIn(lt), FadeIn(sch))
            vo.wait_until("m")
            self.play(Create(model), FadeIn(mt), FadeIn(glob[:4]))
            self.play(GrowArrow(bind), FadeIn(bl))
            vo.wait_until("a")
            self.play(FadeIn(glob[4], shift=UP * 0.1), FadeIn(new))
            vo.wait_until("r")
            self.play(FadeIn(remap, lag_ratio=0.15))
            vo.wait_until("d")
            self.play(FadeIn(dist))
        self.play(FadeOut(VGroup(worker, wt, local, lt, model, mt, glob, new, bind, bl, remap, dist, sch)))

    # ------------------------------------------------------------------
    def growth(self):
        ev = self.v["events"]
        xs = [e["batch"] for e in ev]
        labels_ = [e["labels_after"] for e in ev]
        rows = [e["rows_after"] for e in ev]
        assert sorted(set(rows)) == [1, 4, 8, 16, 32, 64]
        pl = Plot((0, 14), (0, 70), width=7.0, height=4.4, x_ticks=[0, 2, 4, 6, 8, 10, 12, 14],
                  y_ticks=[0, 16, 32, 48, 64], font_size=22).move_to(LEFT * 2.4 + DOWN * 0.3)
        step_pts = []
        for i, (x, r) in enumerate(zip(xs, rows)):
            step_pts += [pl.c2p(x, r), pl.c2p(x + 1, r)]
        step = VMobject().set_points_as_corners(step_pts).set_stroke(C.CATEGORY, 4)
        fill = Polygon(*([pl.c2p(0, 0)] + step_pts + [pl.c2p(14, 0)]), stroke_width=0, fill_color=C.CATEGORY,
                       fill_opacity=0.15)
        dots = VGroup(*[pl.dot(x + 0.5, l_, color=WHITE, radius=0.06) for x, l_ in zip(xs, labels_)])
        xt = pl.x_title(r"training batch", font_size=24)
        leg = VGroup(
            VGroup(Square(0.25, stroke_width=0, fill_color=C.CATEGORY, fill_opacity=0.8),
                   label(r"rows allocated", font_size=26, color=C.CATEGORY)).arrange(RIGHT, buff=0.15),
            VGroup(Dot(radius=0.07, color=WHITE), label(r"labels admitted", font_size=26)).arrange(RIGHT, buff=0.15),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).next_to(pl, RIGHT, buff=0.6).shift(UP * 1.2)
        seq = mtex(r"1 \to 4 \to 8 \to 16 \to 32 \to 64", font_size=40, color=C.CATEGORY).next_to(leg, DOWN, buff=0.6)
        seq.align_to(leg, LEFT)
        rule = label(r"capacity: the next power of two\\that holds every label", font_size=24, color=GREY_A)
        rule.next_to(seq, DOWN, buff=0.3).align_to(leg, LEFT)
        tag = real_tag(r"a real training run: new merchants keep arriving, relflow 0.2.3")
        with self.voiceover(
            "And when the table runs out of rows, it grows. <bookmark mark='p'/> Here's a real training run in which new "
            "merchants keep showing up, three more in every batch. <bookmark mark='l'/> The white dots count the labels "
            "admitted. <bookmark mark='r'/> The orange staircase is the storage behind them: one row, then four, eight, "
            "sixteen, thirty-two, sixty-four. Each time, the capacity jumps to the next power of two that holds every "
            "label, so growing is rare, and cheap on average."
        ) as vo:
            vo.wait_until("p")
            self.play(Create(pl), FadeIn(xt), FadeIn(tag))
            vo.wait_until("l")
            self.play(FadeIn(leg[1]), LaggedStart(*[FadeIn(d, scale=0.5) for d in dots], lag_ratio=0.08), run_time=1.5)
            vo.wait_until("r")
            self.play(FadeIn(leg[0]), Create(step), FadeIn(fill), run_time=2.0)
            self.play(Write(seq), FadeIn(rule))
        self.play(FadeOut(VGroup(pl, step, fill, dots, xt, leg, seq, rule, tag)))

    # ------------------------------------------------------------------
    def optimizer(self):
        ev = self.v["events"]
        grows = [e for e in ev if e["rows_after"] > e["rows_before"] and "moments_kept" in e]
        assert all(e["moments_kept"] and e["new_moments_zero"] and e["weights_kept"] for e in grows)

        def block(n_rows, filled, color, w=1.3):
            g = VGroup(*[Rectangle(width=w, height=0.19, stroke_width=0.8, stroke_color=GREY_D,
                                   fill_color=color if i < filled else BACKGROUND, fill_opacity=0.65 if i < filled else 1)
                         for i in range(n_rows)]).arrange(DOWN, buff=0.025)
            return g

        cols = [(r"weights", C.CATEGORY), (r"Adam $m$", C.ADAMW), (r"Adam $v$", C.ADAMW)]
        before = VGroup(*[block(8, 7, c) for _, c in cols]).arrange(RIGHT, buff=0.25)
        after = VGroup(*[block(16, 7, c) for _, c in cols]).arrange(RIGHT, buff=0.25)
        before.move_to(LEFT * 3.9 + UP * 1.35)
        after.move_to(RIGHT * 2.6).to_edge(UP, buff=1.0)
        heads_b = VGroup(*[label(t, font_size=22, color=c).next_to(b, UP, buff=0.15) for (t, c), b in zip(cols, before)])
        heads_a = VGroup(*[label(t, font_size=22, color=c).next_to(b, UP, buff=0.15) for (t, c), b in zip(cols, after)])
        tb = label(r"8 rows, 7 labels", font_size=24, color=GREY_A).next_to(before, DOWN, buff=0.3)
        ta = label(r"16 rows: old rows copied, new moments zero", font_size=24, color=GREY_A).next_to(after, DOWN, buff=0.3)
        arrow = Arrow(before.get_right(), after.get_left(), buff=0.3, color=GREY_B, stroke_width=3)
        new_bracket = Brace(VGroup(*[after[0][i] for i in range(7, 16)]), RIGHT, buff=0.1, color=C.VALUED)
        checks = VGroup(label(r"at every growth in the run: old rows and Adam moments\\identical bit for bit, new moments exactly 0",
                              font_size=22, color=GREY_A),
                        *[label(rf"batch {e['batch']}: {e['rows_before']} $\to$ {e['rows_after']} rows \checkmark", font_size=24,
                                color=C.VALUED) for e in grows]).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        checks.to_corner(DL, buff=0.5)
        tag = real_tag(r"checked inside the real run, at every growth", corner=UR)
        with self.voiceover(
            "The tricky part is the optimizer. <bookmark mark='a'/> Adam keeps two running averages for every single "
            "parameter: one of its gradient, and one of the gradient squared. <bookmark mark='g'/> Growing the table "
            "means replacing the parameter with a bigger one, and a naive swap would throw that history away. "
            "<bookmark mark='c'/> relflow copies the old rows, their pending gradients, and Adam's moments into the new "
            "storage, and starts the new rows with zero moments, as if they had just been created. "
            "<bookmark mark='v'/> We checked this inside the run you just saw. At every growth, the old rows and their "
            "moments came through identical, bit for bit, and the new moments were exactly zero."
        ) as vo:
            vo.wait_until("a")
            self.play(FadeIn(before), FadeIn(heads_b), FadeIn(tb))
            vo.wait_until("g")
            self.play(GrowArrow(arrow))
            vo.wait_until("c")
            self.play(TransformFromCopy(before, VGroup(*[VGroup(*col[:8]) for col in after])), FadeIn(heads_a))
            self.play(FadeIn(VGroup(*[VGroup(*col[8:]) for col in after])), FadeIn(ta), GrowFromCenter(new_bracket))
            vo.wait_until("v")
            self.play(FadeIn(tag), LaggedStart(*[FadeIn(c, shift=UP * 0.1) for c in checks], lag_ratio=0.25), run_time=2)
        self.play(FadeOut(VGroup(before, after, heads_b, heads_a, tb, ta, arrow, new_bracket, checks, tag)))

    # ------------------------------------------------------------------
    def unknown(self):
        entries = [(r"training", r"admits new labels, grows storage"),
                   (r"validation, test", r"reuse the mapping: unseen labels stay unknown"),
                   (r"prediction, serving", r"requests never teach new labels")]
        table = VGroup()
        for i, (a, b) in enumerate(entries):
            y = 2.3 - 0.6 * i
            table.add(VGroup(label(a, font_size=28).move_to([-1.6, y, 0], aligned_edge=RIGHT),
                             label(b, font_size=26, color=GREY_A).move_to([-1.1, y, 0], aligned_edge=LEFT)))
        unk = VGroup(Mono("sku: Z99", font_size=28), state_tile(0, 0.48), label(r"valued, zero content", font_size=26,
                                                                                  color=GREY_A)).arrange(RIGHT, buff=0.3)
        unk.move_to(DOWN * 0.9)
        pu = VGroup(Mono("p_unavailable=0.01", font_size=26, color=C.CATEGORY),
                    label(r"during training, 1\% of known labels are hidden on purpose,\\so the model learns to lean on "
                          r"the rest of the context", font_size=26, color=GREY_A)).arrange(DOWN, buff=0.2)
        pu.next_to(unk, DOWN, buff=0.5)
        with self.voiceover(
            "Only training admits labels. <bookmark mark='v'/> Validation and testing reuse the mapping, so a label "
            "they've never seen stays unknown, <bookmark mark='p'/> and a model being served never learns anything "
            "from its requests. <bookmark mark='u'/> An unknown label is still valued, with zero content. "
            "<bookmark mark='h'/> To prepare the model for that, relflow hides one percent of known labels during "
            "training, by default, so the model learns to lean on the rest of the context when a label tells it "
            "nothing."
        ) as vo:
            self.play(FadeIn(table[0]))
            vo.wait_until("v")
            self.play(FadeIn(table[1]))
            vo.wait_until("p")
            self.play(FadeIn(table[2]))
            vo.wait_until("u")
            self.play(FadeIn(unk))
            vo.wait_until("h")
            self.play(FadeIn(pu, shift=UP * 0.1))
        self.wait(0.4)
        self.play(FadeOut(VGroup(table, unk, pu)))
