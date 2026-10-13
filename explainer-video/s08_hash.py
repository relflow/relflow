from __future__ import annotations

import numpy as np

from explainer import *  # noqa: F403
from videos.relflow.common import (
    D, Mono, Plot, chip, label, mtex, note, proof_tag, real_tag, show_episode_card, vector_strip,
)


def sim_grid(sim, ids, cell=0.36, color=C.HASH) -> VGroup:
    sim = np.asarray(sim)
    n = len(ids)
    sq = VGroup()
    for i in range(n):
        for j in range(n):
            v = sim[i, j]
            same = abs(v - 1) < 1e-5
            a = 1.0 if same else max(0.0, (v - 0.5) / 0.5) * 0.45
            sq.add(Square(cell, stroke_width=1.5 if same else 0.4, stroke_color=WHITE if same else GREY_D,
                          fill_color=color, fill_opacity=0.08 + 0.92 * a))
    sq.arrange_in_grid(n, n, buff=0.02)
    g = VGroup(sq)
    g.sq = sq
    return g


class Hashing(VoiceoverScene):
    def construct(self):
        show_episode_card(self)
        self.h = D("hash")
        self.problem()
        self.fingerprint()
        self.salt()
        self.evidence()

    # ------------------------------------------------------------------
    def problem(self):
        ids = self.h["ids"]
        acct = VGroup(Mono("account_id:", font_size=28, color=C.RF_FIELD), Mono(ids[0], font_size=28, color=YELLOW))
        acct.arrange(RIGHT, buff=0.2)
        rows = VGroup(*[VGroup(Mono("- counterparty_id:", font_size=26, color=C.RF_FIELD),
                               Mono(c, font_size=26, color=YELLOW if c == ids[0] else WHITE)).arrange(RIGHT, buff=0.2)
                        for c in ids[1:]])
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        tr = Mono("transfers:", font_size=28, color=C.RF_BRANCH)
        block = VGroup(acct, tr, rows).arrange(DOWN, aligned_edge=LEFT, buff=0.15).to_edge(LEFT, buff=0.8).shift(UP * 0.4)
        rows.shift(RIGHT * 0.4)
        q = label(r"did this account send money to itself?", font_size=30, color=YELLOW).next_to(block, DOWN, buff=0.5)
        q.align_to(block, LEFT)
        pts = VGroup(
            label(r"millions of IDs, most seen once or twice", font_size=28, color=GREY_A),
            label(r"a vocabulary row per ID memorizes noise", font_size=28, color=GREY_A),
            label(r"and knows nothing about tomorrow's new IDs", font_size=28, color=GREY_A),
            label(r"often only \emph{equality} matters", font_size=30),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.25).to_edge(RIGHT, buff=0.6).shift(UP * 0.4)
        same = VGroup(*[SurroundingRectangle(r[1], buff=0.05, color=YELLOW, stroke_width=2) for r, c in zip(rows, ids[1:]) if c == ids[0]])
        with self.voiceover(
            "Some values are identifiers: account numbers, device IDs, card numbers. <bookmark mark='m'/> There can be "
            "millions of them, and most show up once or twice. A vocabulary row for each one would memorize noise, and "
            "would know nothing about the new IDs arriving tomorrow. <bookmark mark='e'/> Often, what matters isn't "
            "which ID it is at all, only whether two of them are the same. <bookmark mark='q'/> Did this account just "
            "send money to itself?"
        ) as vo:
            self.play(FadeIn(block, lag_ratio=0.05), run_time=1.5)
            vo.wait_until("m")
            self.play(LaggedStart(*[FadeIn(p, shift=LEFT * 0.1) for p in pts[:3]], lag_ratio=0.5), run_time=3)
            vo.wait_until("e")
            self.play(FadeIn(pts[3], shift=LEFT * 0.1))
            vo.wait_until("q")
            self.play(FadeIn(q), Create(same), Create(SurroundingRectangle(acct[1], buff=0.05, color=YELLOW, stroke_width=2)))
        self.play(*[FadeOut(m) for m in self.mobjects])

    # ------------------------------------------------------------------
    def fingerprint(self):
        b = self.h["batches"][0]
        lanes = np.array(b["lanes"][0])
        vec = np.array(b["vectors"][0])
        idt = chip("acct-17", color=YELLOW, font_size=28, mono=True).move_to(LEFT * 5.6 + UP * 0.4)
        h = VGroup(*[Mono(f"lane {k}: {v:+.3f}", font_size=24, color=C.HASH) for k, v in enumerate(lanes)])
        h.arrange(DOWN, aligned_edge=LEFT, buff=0.15).move_to(LEFT * 2.3 + UP * 0.4)
        a1 = Arrow(idt.get_right(), h.get_left(), buff=0.15, color=GREY_B, stroke_width=3)
        l1 = label(r"two salted 64-bit hashes,\\scaled to $[-1,1)$", font_size=22, color=GREY_A).next_to(h, DOWN, buff=0.45)
        strip = vector_strip(vec, color=C.HASH, cell=0.13, gap=0.018, direction=RIGHT)
        strip.scale_to_fit_width(5.6).move_to(RIGHT * 3.75 + UP * 0.4)
        a2 = Arrow(h.get_right(), strip.get_left(), buff=0.15, color=GREY_B, stroke_width=3)
        l2 = label(r"$\sin, \cos$ at fixed frequencies,\\summed over lanes", font_size=22, color=GREY_A)
        l2.next_to(strip, DOWN, buff=0.45)
        l3 = label(r"the token: no learned table at all", font_size=26, color=C.HASH).next_to(strip, UP, buff=0.35)
        tag = real_tag(r"the real Hash embedder, $d_{\mathrm{model}}=64$, two lanes")
        with self.voiceover(
            "relflow's Hash type keeps exactly that, and nothing more. <bookmark mark='h'/> Each ID is hashed to 64-bit "
            "integers, in a few independent lanes, scaled to lie between minus one and one. <bookmark mark='f'/> Then "
            "the lanes go through sines and cosines at fixed frequencies, straight into model space. "
            "<bookmark mark='t'/> There's no learned table at all: the vector is a pure function of the hash."
        ) as vo:
            self.play(FadeIn(idt), FadeIn(tag))
            vo.wait_until("h")
            self.play(GrowArrow(a1), FadeIn(h, lag_ratio=0.3), FadeIn(l1))
            vo.wait_until("f")
            self.play(GrowArrow(a2), FadeIn(strip, lag_ratio=0.01), FadeIn(l2), run_time=1.4)
            vo.wait_until("t")
            self.play(FadeIn(l3))
        self.play(*[FadeOut(m) for m in self.mobjects])

    # ------------------------------------------------------------------
    def salt(self):
        bs = self.h["batches"]
        ids = self.h["ids"]
        names = [r"training batch 1", r"training batch 2", r"training batch 3"]
        strips = VGroup()
        for k in range(3):
            v = np.array(bs[k]["vectors"][0])
            s = vector_strip(v, color=C.HASH, cell=0.1, gap=0.014, direction=RIGHT)
            strips.add(VGroup(label(names[k], font_size=22, color=GREY_A), s).arrange(RIGHT, buff=0.3))
        strips.arrange(DOWN, aligned_edge=RIGHT, buff=0.18).to_edge(UP, buff=0.55)
        sl = label(r"acct-17's vector in three batches: a fresh salt each time", font_size=26, color=C.HASH)
        sl.next_to(strips, UP, buff=0.15)
        grids = VGroup()
        for k in range(3):
            g = sim_grid(bs[k]["similarity"], ids, cell=0.3)
            grids.add(g)
        grids.arrange(RIGHT, buff=0.9).next_to(strips, DOWN, buff=0.8)
        labs = VGroup(*[label(names[k], font_size=22, color=GREY_A).next_to(grids[k], DOWN, buff=0.15) for k in range(3)])
        ticks = VGroup()
        for i, t in enumerate(["17", "17", "92", "17", "5", "92", "33"]):
            ticks.add(Mono(t, font_size=14, color=YELLOW if t == "17" else GREY_B).next_to(grids[0].sq[i * 7], LEFT, buff=0.08))
        gl = label(r"cosine similarity between the 7 IDs of one observation: equal IDs give identical vectors (outlined)",
                   font_size=22, color=GREY_A).next_to(labs, DOWN, buff=0.25)
        pred = label(r"test and prediction: salt $= 0$, deterministic", font_size=24, color=GREY_A).to_edge(DOWN, buff=0.2)
        for k in range(3):
            sim = np.array(bs[k]["similarity"])
            assert abs(sim[0, 1] - 1) < 1e-5 and abs(sim[0, 3] - 1) < 1e-5 and abs(sim[2, 5] - 1) < 1e-5
        assert bs[0]["lanes"][0] != bs[1]["lanes"][0]
        with self.voiceover(
            "Now the trick. <bookmark mark='s'/> During training, every batch mixes a fresh random salt into the hash, "
            "so the very same account comes out as a completely different vector in the next batch. The model has no "
            "stable vector to memorize. <bookmark mark='g'/> But inside one batch, equal IDs always produce identical "
            "vectors, in every Hash field at once. Here are the similarities between the seven IDs of one observation, "
            "in three batches. The vectors change every time; <bookmark mark='p'/> the pattern of equalities never "
            "does. <bookmark mark='d'/> At test and prediction time the salt is zero, so serving is deterministic."
        ) as vo:
            vo.wait_until("s")
            self.play(FadeIn(sl), LaggedStart(*[FadeIn(s_) for s_ in strips], lag_ratio=0.4), run_time=1.8)
            vo.wait_until("g")
            self.play(FadeIn(grids[0]), FadeIn(labs[0]), FadeIn(ticks), FadeIn(gl))
            self.play(FadeIn(grids[1]), FadeIn(labs[1]))
            self.play(FadeIn(grids[2]), FadeIn(labs[2]))
            vo.wait_until("p")
            outlined = [VGroup(*[g.sq[i] for i in range(49) if g.sq[i].get_stroke_width() > 1]) for g in grids]
            self.play(*[Indicate(o, color=WHITE, scale_factor=1.05) for o in outlined])
            vo.wait_until("d")
            self.play(FadeIn(pred))
        self.play(*[FadeOut(m) for m in self.mobjects])

    # ------------------------------------------------------------------
    def evidence(self):
        p = D("proofs")["metrics"]
        m21, m28 = p["P021"], p["P028"]
        assert m21["hash_auc"] > 0.999 and m21["category_oov_auc"] == 0.5 and m28["intact_auc"] < 0.5
        pl = Plot((0, 3), (0, 1), width=5.4, height=3.4, y_ticks=[0, 0.5, 1.0], font_size=22).move_to(LEFT * 3.2 + DOWN * 0.5)
        bars = [(r"Hash", m21["hash_auc"], C.HASH), (r"Category\\(unseen IDs)", m21["category_oov_auc"], C.CATEGORY),
                (r"control: labels\\shuffled", m21["shuffled_auc"], GREY_B)]
        rects, texts, vals = VGroup(), VGroup(), VGroup()
        for i, (t, v, c) in enumerate(bars):
            r = pl.bar(i + 0.5, v, 0.6, c)
            rects.add(r)
            texts.add(label(t, font_size=22, color=GREY_A).next_to(pl.c2p(i + 0.5, 0), DOWN, buff=0.2))
            vals.add(label(f"{v:.4f}" if v > 0.99 else f"{v:.2f}", font_size=24).next_to(r, UP, buff=0.08))
        head = label(r"P021: does a pair of \emph{never seen} IDs match?", font_size=26).next_to(pl, UP, buff=0.95)
        yt = label(r"held-out AUC", font_size=22, color=GREY_A).next_to(pl, UP, buff=0.5).align_to(pl, LEFT)

        lim = VGroup(
            label(r"the limit (P028, recorded as \emph{Limited}):", font_size=26, color=RED),
            label(r"compare two sibling collections", font_size=26, color=GREY_A),
            label(r"after each has been summarized:", font_size=26, color=GREY_A),
            label(rf"AUC {m28['intact_auc']:.2f}, chance", font_size=30, color=RED),
            label(r"equality helps only while the tokens\\can still attend to each other", font_size=26),
            Mono("transfers=rf.Branch(reduction=None, ...)", font_size=20, color=C.RF_BRANCH),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18).to_edge(RIGHT, buff=0.5).shift(UP * 0.2)
        tag = proof_tag(r"P021 and P028")
        with self.voiceover(
            "Does it work? <bookmark mark='p'/> In relflow's proof P021, the answer depends only on whether two IDs "
            "match, and every test ID is brand new. <bookmark mark='h'/> Hash scores an AUC of 0.9999. "
            "<bookmark mark='c'/> A Category vocabulary, facing IDs it has never seen, can only guess: one half. "
            "<bookmark mark='l'/> But equality only helps while the tokens can still attend to each other. "
            "<bookmark mark='s'/> In proof P028, two sibling collections are each summarized first, and then compared. "
            "The signal is gone: chance. relflow records that one as a limitation. <bookmark mark='r'/> It's why the "
            "documentation's Hash example keeps its transfers unsummarized, with reduction set to None."
        ) as vo:
            self.play(Create(pl), FadeIn(yt), FadeIn(tag))
            vo.wait_until("p")
            self.play(FadeIn(head))
            vo.wait_until("h")
            self.play(GrowFromEdge(rects[0], DOWN), FadeIn(texts[0]), FadeIn(vals[0]))
            vo.wait_until("c")
            self.play(GrowFromEdge(rects[1], DOWN), FadeIn(texts[1]), FadeIn(vals[1]))
            self.play(GrowFromEdge(rects[2], DOWN), FadeIn(texts[2]), FadeIn(vals[2]))
            vo.wait_until("l")
            self.play(FadeIn(lim[0]))
            vo.wait_until("s")
            self.play(FadeIn(lim[1:4], lag_ratio=0.3))
            vo.wait_until("r")
            self.play(FadeIn(lim[4:], lag_ratio=0.3))
        self.wait(0.4)
        self.play(*[FadeOut(m) for m in self.mobjects])
