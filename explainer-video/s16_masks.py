from __future__ import annotations

import numpy as np

from explainer import *  # noqa: F403
from videos.relflow.common import (
    D, Mono, label, mtex, note, proof_tag, real_tag, show_episode_card, tile,
)


def cell(kind: str, size=0.24) -> VMobject:
    """One coordinate of a field: valued, behind a learned mask, removed by a mask, or structural padding."""
    if kind == "V":
        return tile(C.VALUED, size, "V", fill=0.8)
    if kind == "M":
        return tile(C.MASKED, size, "M", fill=0.95)
    if kind == "skip":
        sq = RoundedRectangle(width=size, height=size, corner_radius=size * 0.16, stroke_color=C.MASKED,
                              stroke_width=1.6, fill_opacity=0)
        return VGroup(DashedVMobject(sq, num_dashes=8))
    return tile(C.PADDED, size, fill=0.25)


def grid(states, present, lengths, size=0.24, gap=0.04) -> VGroup:
    g = VGroup()
    for i, (srow, prow) in enumerate(zip(states, present)):
        for j, (s, p) in enumerate(zip(srow, prow)):
            real = j < lengths[i]
            kind = "P" if not real else ("skip" if not p else ("M" if s == 3 else "V"))
            g.add(cell(kind, size))
    g.arrange_in_grid(len(states), len(states[0]), buff=gap)
    return g


class Masks(VoiceoverScene):
    def construct(self):
        show_episode_card(self)
        self.d = D("masks")
        self.intro()
        self.masks()
        self.batch()
        self.branch()
        self.determinism()
        self.stages()
        self.evidence()

    # ------------------------------------------------------------------
    def intro(self):
        steps = VGroup(
            label(r"1. \textbf{select} some values", font_size=34),
            label(r"2. \textbf{hide} them from the encoder", font_size=34),
            label(r"3. optionally, \textbf{reconstruct} them", font_size=34),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.4).move_to(UP * 0.4)
        uses = label(r"targets $\cdot$ dropout $\cdot$ self-supervision $\cdot$ ablation", font_size=30, color=C.MASKED)
        uses.next_to(steps, DOWN, buff=0.7)
        with self.voiceover(
            "In relflow, every objective comes from one mechanism: the mask. <bookmark mark='s'/> A mask selects some "
            "values, <bookmark mark='h'/> hides them from the encoder, <bookmark mark='r'/> and, if you ask, makes "
            "the model reconstruct them. <bookmark mark='u'/> That one idea gives you supervised targets, dropout, "
            "self-supervised pretraining, and ablations."
        ) as vo:
            for mark, s in zip("shr", steps):
                vo.wait_until(mark)
                self.play(FadeIn(s, shift=RIGHT * 0.15))
            vo.wait_until("u")
            self.play(FadeIn(uses, shift=UP * 0.1))
        self.play(FadeOut(VGroup(steps, uses)))

    # ------------------------------------------------------------------
    def batch(self):
        d = self.d
        lengths = d["lengths"]
        panels = VGroup()
        specs = [("visible", r"\texttt{quantity=rf.Number}", "train"),
                 ("dropout", r"\texttt{mask=0.25}", "train"),
                 ("skip", r"\texttt{Mask(rate=0.25, skip=True)}", "train"),
                 ("target", r"\texttt{mask=True}", "train")]
        for key, title_, strata in specs:
            g = grid(d[key][strata]["state"], d[key][strata]["present"], lengths)
            t = label(title_, font_size=19, color=GREY_A).next_to(g, UP, buff=0.18)
            panels.add(VGroup(g, t))
        panels.arrange(RIGHT, buff=0.45).move_to(DOWN * 0.2)
        head = label(r"the \texttt{quantity} of 8 real orders, 8 item slots each, in one training batch", font_size=26)
        head.to_edge(UP, buff=0.5)
        legend = VGroup(*[VGroup(cell(k, 0.24), label(t, font_size=20, color=GREY_A)).arrange(RIGHT, buff=0.12)
                          for k, t in (("V", "a visible value"), ("M", "hidden behind a learned mask vector"),
                                       ("skip", "removed: not in attention at all"), ("P", "padding: no item"))])
        legend.arrange(RIGHT, buff=0.45).to_edge(DOWN, buff=0.75)
        n_m = int((np.array(d["dropout"]["train"]["state"]) == 3).sum())
        n_real = sum(lengths)
        tag = real_tag(r"real encodings from \texttt{model.encode}, relflow 0.2.3", corner=DR)
        assert np.array(d["target"]["train"]["present"]).sum() == 0
        with self.voiceover(
            f"Here's what that means for one real batch: the quantities of eight orders, {n_real} items in all. "
            "<bookmark mark='v'/> As an ordinary input, every item's quantity is a visible token. "
            f"<bookmark mark='m'/> With mask equals 0.25, about a quarter of them, {n_m} here, are swapped for a "
            "learned mask vector. They're still tokens in the sequence, so the model knows something is hidden there. "
            "<bookmark mark='k'/> Add skip equals True, and the selected values are removed instead: they leave "
            "attention entirely, as if those items had no quantity at all. <bookmark mark='t'/> And mask equals True "
            "skips every quantity, always. That's a target: the model never sees it, and learns to predict it."
        ) as vo:
            self.play(FadeIn(head), FadeIn(legend), FadeIn(tag))
            for mark, p in zip("vmkt", panels):
                vo.wait_until(mark)
                self.play(FadeIn(p, lag_ratio=0.01), run_time=1.0)
        self.play(FadeOut(VGroup(head, legend, panels, tag)))

    # ------------------------------------------------------------------
    def branch(self):
        d = self.d
        lengths = d["lengths"]
        fields = ("/line_items/sku", "/line_items/quantity", "/line_items/price")
        rows = VGroup()
        for f in fields:
            pres = d["branch"][f]
            g = grid([[0] * 8] * 8, pres, lengths, size=0.22)
            rows.add(VGroup(g, Mono(f.split("/")[-1], font_size=18, color=GREY_A).next_to(g, UP, buff=0.15)))
        rows.arrange(RIGHT, buff=0.6).move_to(DOWN * 0.3)
        code_ = Mono('line_items=rf.Branch(length=8, mask=rf.Mask(rate=0.25, skip=True), ...)', font_size=20,
                     color=C.MASKED).to_edge(UP, buff=0.6)
        same = label(r"the same items vanish from every field", font_size=26, color=GREY_A).to_edge(DOWN, buff=0.8)
        assert d["branch"][fields[0]] == d["branch"][fields[1]] == d["branch"][fields[2]]
        with self.voiceover(
            "Put the mask on the branch instead, <bookmark mark='g'/> and one decision covers the whole subtree. "
            "<bookmark mark='s'/> A selected line item loses its SKU, its quantity and its price together, and any "
            "branches nested inside it. A mask on the root does the same for whole observations."
        ) as vo:
            self.play(FadeIn(code_))
            vo.wait_until("g")
            self.play(FadeIn(rows, lag_ratio=0.02), run_time=1.2)
            vo.wait_until("s")
            self.play(FadeIn(same))
        self.play(FadeOut(VGroup(code_, rows, same)))

    # ------------------------------------------------------------------
    def masks(self):
        rows = [
            ("mask=True", r"a supervised target: never embedded, always reconstructed"),
            ("mask=0.15", r"training-only dropout: 15\% hidden behind a learned mask token"),
            ("mask=rf.Mask(rate=0.15, skip=True)", r"training-only dropout that removes the value entirely"),
            ("mask=rf.Mask(rate=0.15, reconstruct=True)", r"self-supervised: reconstruct a sampled 15\%"),
            ('mask=rf.Mask(query="hide_amount", ...)', r"the data decides, with a Boolean column"),
            ("orders=rf.Branch(mask=rf.Mask(rate=0.1, skip=True))", r"one decision hides a whole subtree"),
        ]
        g = VGroup()
        for i, (c_, t) in enumerate(rows):
            y = 2.55 - 0.92 * i
            g.add(VGroup(Mono(c_, font_size=21, color=C.MASKED).move_to([-6.6, y + 0.18, 0], aligned_edge=LEFT),
                         label(t, font_size=24, color=GREY_A).move_to([-6.2, y - 0.22, 0], aligned_edge=LEFT)))
        head = label(r"one mechanism, many uses: masks", font_size=30).to_edge(UP, buff=0.3)
        with self.voiceover(
            "Here are the policies you'll actually write. <bookmark mark='a'/> mask equals True makes a supervised target: the value is never embedded, and always "
            "reconstructed. <bookmark mark='b'/> mask equals 0.15 is training-only dropout: fifteen percent of values "
            "hide behind a learned mask token. <bookmark mark='k'/> Add skip equals True, and they're removed "
            "instead. <bookmark mark='c'/> A Mask with a rate and reconstruct equals True is "
            "self-supervised learning: hide a sample, and learn to predict it from the rest. <bookmark mark='d'/> A "
            "Mask with a query lets a Boolean column in the data decide, <bookmark mark='e'/> and a mask on a branch "
            "makes one decision for its entire subtree."
        ) as vo:
            self.play(FadeIn(head))
            for mark, r in zip("abkcde", g):
                vo.wait_until(mark)
                self.play(FadeIn(r, shift=RIGHT * 0.1))
        self.play(FadeOut(VGroup(head, g)))

    # ------------------------------------------------------------------
    def determinism(self):
        m = D("encode")["masking"]
        present = np.array(m["present"])
        ep = {k: np.array(v) for k, v in m["epochs"].items()}
        assert (ep["train0"] <= present).all()

        def grid(sel, title_):
            cells = VGroup()
            for i in range(12):
                for j in range(16):
                    if not present[i, j]:
                        cells.add(Square(0.13, stroke_width=0, fill_color=C.PADDED, fill_opacity=0.25))
                    elif sel[i, j]:
                        cells.add(Square(0.13, stroke_width=0, fill_color=C.MASKED, fill_opacity=1))
                    else:
                        cells.add(Square(0.13, stroke_width=0, fill_color=C.VALUED, fill_opacity=0.55))
            cells.arrange_in_grid(12, 16, buff=0.025)
            t = label(title_, font_size=20, color=GREY_A).next_to(cells, UP, buff=0.12)
            return VGroup(cells, t)

        grids = VGroup(grid(ep["train0"], r"train, epoch 0"), grid(ep["train0"], r"train, epoch 0 again"),
                       grid(ep["train1"], r"train, epoch 1"), grid(ep["validate5"], r"validation, epoch 5"))
        grids.arrange(RIGHT, buff=0.45).move_to(DOWN * 0.3)
        f = mtex(r"\text{score} = \text{rng}\big(\,\mathrm{SHA256}(\text{seed},\ \text{stage},\ \text{epoch},\ "
                 r"\text{address},\ \text{policy})\big)", font_size=30).to_edge(UP, buff=0.5)
        same = label(r"identical", font_size=24, color=C.VALUED).next_to(VGroup(grids[0], grids[1]), DOWN, buff=0.3)
        val = label(r"validation always uses epoch 0:\\comparable across epochs", font_size=22, color=GREY_A)
        val.next_to(grids[3], DOWN, buff=0.3)
        tag = real_tag(r"real selections at a 15\% rate, relflow's masking code")
        with self.voiceover(
            "Sampling is deterministic. <bookmark mark='f'/> Every coordinate gets a score from a random generator "
            "seeded by a SHA-256 hash of the seed, the stage, the epoch, the field's address and the policy. "
            "<bookmark mark='g'/> Here are real selections at a fifteen percent rate, on a batch of orders. "
            "<bookmark mark='s'/> Epoch zero picks the same coordinates every time you run it. <bookmark mark='n'/> "
            "Epoch one picks fresh ones. <bookmark mark='v'/> And validation always pretends it's epoch zero, so a "
            "validation score means the same thing from one epoch to the next."
        ) as vo:
            vo.wait_until("f")
            self.play(Write(f), run_time=1.5)
            vo.wait_until("g")
            self.play(FadeIn(grids[0]), FadeIn(tag))
            vo.wait_until("s")
            self.play(FadeIn(grids[1]), FadeIn(same))
            vo.wait_until("n")
            self.play(FadeIn(grids[2]))
            vo.wait_until("v")
            self.play(FadeIn(grids[3]), FadeIn(val))
        self.play(FadeOut(VGroup(f, grids, same, val, tag)))

    # ------------------------------------------------------------------
    def stages(self):
        head = VGroup(*[label(t, font_size=24, color=GREY_A) for t in (r"policy", r"train", r"validate / test",
                                                                       r"predict")])
        rows_ = [
            (r"dropout (\texttt{mask=0.15})", r"on", r"off", r"off"),
            (r"fixed target (\texttt{mask=True})", r"on", r"on", r"on"),
            (r"sampled reconstruction", r"sampled", r"sampled, epoch 0", r"off"),
        ]
        cols_x = [-6.4, -0.6, 1.6, 4.6]
        g = VGroup()
        for c, x in zip(head, cols_x):
            c.move_to([x, 1.8, 0], aligned_edge=LEFT)
        g.add(head)
        for i, r in enumerate(rows_):
            row = VGroup()
            for k, (t, x) in enumerate(zip(r, cols_x)):
                col = WHITE if k == 0 else (C.VALUED if t.startswith("on") or t.startswith("sampled") else GREY_B)
                row.add(label(t, font_size=26, color=col).move_to([x, 0.9 - 0.75 * i, 0], aligned_edge=LEFT))
            g.add(row)
        line = Line([-6.5, 1.45, 0], [6.6, 1.45, 0], color=GREY_D, stroke_width=1.5)
        foot = label(r"a sampled reconstruction trains its decoder, but asks for no predictions when serving",
                     font_size=24, color=GREY_A).to_edge(DOWN, buff=0.9)
        with self.voiceover(
            "And the stages differ. <bookmark mark='d'/> Dropout only acts during training. <bookmark mark='t'/> A fixed "
            "target is hidden everywhere: that's what makes it a prediction. <bookmark mark='s'/> And a sampled "
            "reconstruction trains its decoder, and is measured during validation, but asks for no predictions when "
            "the model is serving."
        ) as vo:
            self.play(FadeIn(head), Create(line))
            for mark, r in zip("dts", g[1:]):
                vo.wait_until(mark)
                self.play(FadeIn(r, shift=RIGHT * 0.1))
            self.play(FadeIn(foot))
        self.wait(0.4)
        self.play(FadeOut(VGroup(g, line, foot)))

    # ------------------------------------------------------------------
    def evidence(self):
        m = D("proofs")["metrics"]
        p51, p53 = m["P051"], m["P053"]
        y = p51["directions"]["y"]
        hh, vh = p53["patterns"]["hiddenhidden"], p53["patterns"]["visiblehidden"]
        rows = VGroup(
            label(rf"P051, sampled reconstruction: normalized error {y['nrmse']:.2f}"
                  rf" ({y['shuffled_nrmse']:.2f} with the context shuffled)", font_size=24),
            label(rf"P053, two branches that both carry the answer: hide one, {vh['nrmse']:.2f};"
                  rf" hide both, {hh['nrmse']:.2f}", font_size=24),
            label(r"hidden values never leak: their measured drift is exactly 0", font_size=24, color=GREY_A),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        tag = proof_tag(r"P051, P053")
        assert y["hidden_value_drift"] == 0.0 and hh["hidden_value_drift"] == 0.0
        with self.voiceover(
            "relflow's proof catalog checks this machinery directly. <bookmark mark='a'/> A model trained to "
            "reconstruct sampled values gets its error down to about a tenth of a constant guess, and loses it all "
            "when the context is shuffled. <bookmark mark='b'/> Hide one of two branches that both carry the answer, "
            "and nothing is lost; hide both, and the model falls back to guessing, as it should. "
            "<bookmark mark='c'/> And in every case, changing a hidden value changes nothing the model sees. "
            "<bookmark mark='n'/> Next: what masks make possible, learning without labels."
        ) as vo:
            vo.wait_until("a")
            self.play(FadeIn(rows[0]), FadeIn(tag))
            vo.wait_until("b")
            self.play(FadeIn(rows[1]))
            vo.wait_until("c")
            self.play(FadeIn(rows[2]))
        self.wait(0.4)
        self.play(FadeOut(VGroup(rows, tag)))
