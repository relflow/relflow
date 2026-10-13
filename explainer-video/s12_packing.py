from __future__ import annotations

import numpy as np

from explainer import *  # noqa: F403
from videos.relflow.common import (
    D, Mono, Plot, fmt_int, label, mtex, note, real_tag, show_episode_card,
)


def sq(color, size=0.24, fill=0.9) -> Square:
    return Square(size, stroke_width=0, fill_color=color, fill_opacity=fill)


class Packing(VoiceoverScene):
    def construct(self):
        show_episode_card(self)
        self.p = D("packing")
        self.dense()
        self.pack()
        self.unpack()
        self.timing()
        self.fine_print()

    # ------------------------------------------------------------------
    def dense(self):
        sm = self.p["small"]
        lengths, cap = sm["lengths"], sm["capacity"]
        cells = VGroup()
        for i, n in enumerate(lengths):
            for j in range(cap):
                cells.add(sq(C.LIVE if j < n else C.PADDED, fill=0.9 if j < n else 0.35))
        cells.arrange_in_grid(len(lengths), cap, buff=0.04).move_to(LEFT * 3.3 + DOWN * 0.1)
        self.cells = cells
        rl = label(r"16 rows, room for 12 tokens each", font_size=24, color=GREY_A).next_to(cells, UP, buff=0.25)
        assert sm["pairs_dense"] == 16 * 12 * 12 and sm["pairs_packed"] == sum(n * n for n in lengths)
        cost = VGroup(
            label(r"attention compares every pair of positions:", font_size=26),
            mtex(rf"\text{{dense: }} 16 \times 12^2 = {fmt_int(sm['pairs_dense'])} \text{{ pairs}}", font_size=34),
            mtex(rf"\text{{live tokens only: }} \textstyle\sum_i \ell_i^2 = {sm['pairs_packed']} \text{{ pairs}}",
                 font_size=34, color=C.LIVE),
            label(rf"72\% of the work would multiply padding", font_size=28, color=RED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).next_to(cells, RIGHT, buff=0.8)
        frac = 1 - sm["pairs_packed"] / sm["pairs_dense"]
        assert abs(frac - 0.72) < 0.01
        with self.voiceover(
            "Padding costs the model's answers nothing. It isn't free for the hardware. <bookmark mark='d'/> A dense "
            "batch is a rectangle: every row is as long as the branch's capacity, whether it's full or not. "
            "<bookmark mark='c'/> And attention compares every pair of positions, so its cost grows with the square "
            "of that length. In this small real batch, 72 percent of the attention work would be spent on padding."
        ) as vo:
            self.play(FadeIn(cells, lag_ratio=0.003), FadeIn(rl), run_time=1.5)
            vo.wait_until("d")
            self.play(Indicate(VGroup(*[cells[i * cap + j] for i, n in enumerate(lengths) for j in range(n, cap)]),
                               color=GREY_A, scale_factor=1.0), run_time=1.2)
            vo.wait_until("c")
            self.play(FadeIn(cost[0]), Write(cost[1]))
            self.play(Write(cost[2]))
            self.play(FadeIn(cost[3]))
        self.play(FadeOut(cost), FadeOut(rl))

    # ------------------------------------------------------------------
    def pack(self):
        sm = self.p["small"]
        lengths, cap = sm["lengths"], sm["capacity"]
        cells = self.cells
        groups = sm["groups"]
        rects = VGroup()
        targets = {}
        for g in groups:
            ln, rows = g["length"], g["rows"]
            block = VGroup(*[VGroup(*[sq(C.LIVE, 0.2) for _ in range(ln)]).arrange(RIGHT, buff=0.035) for _ in rows])
            block.arrange(DOWN, buff=0.035)
            lab = label(rf"$\ell={ln}$: {len(rows)} row{'s' if len(rows) > 1 else ''}", font_size=20, color=GREY_A)
            lab.next_to(block, LEFT, buff=0.2)
            grp = VGroup(lab, block)
            grp.block = block
            rects.add(grp)
            for k, r in enumerate(rows):
                for j in range(ln):
                    targets[(r, j)] = block[k][j]
        rects.arrange(DOWN, aligned_edge=RIGHT, buff=0.16).to_edge(RIGHT, buff=0.6)
        if rects.height > 6.6:
            rects.scale_to_fit_height(6.6)
        head = label(r"group rows by length: each group is a perfect rectangle again", font_size=24, color=GREY_A)
        head.next_to(rects, UP, buff=0.2).align_to(rects, RIGHT)
        if head.get_top()[1] > 3.9:
            head.to_edge(UP, buff=0.15)
        flat = VGroup(*[sq(C.LIVE, 0.11) for _ in range(sm["live"])]).arrange(RIGHT, buff=0.014).move_to(DOWN * 3.4)
        fl = label(rf"the {sm['live']} live tokens, gathered once (with their original positions)", font_size=20,
                   color=GREY_A).next_to(flat, UP, buff=0.1)
        movers = []
        anims = []
        for i, n in enumerate(lengths):
            for j in range(n):
                c = cells[i * cap + j].copy()
                movers.append(c)
                anims.append(c.animate.become(targets[(i, j)].copy()))
        with self.voiceover(
            "So whenever some slots are empty, relflow packs. <bookmark mark='g'/> It gathers the live tokens into one "
            "flat list, remembering where each one came from. <bookmark mark='r'/> Then it groups the rows by how many "
            "live tokens they have: the three rows with one item together, the two rows with three items together, "
            "and so on. <bookmark mark='a'/> Inside a group every row has the same length, so attention runs on a "
            "perfect rectangle again, with no mask and no padding at all."
        ) as vo:
            vo.wait_until("g")
            self.play(FadeIn(fl), TransformFromCopy(VGroup(*[cells[i * cap + j] for i, n in enumerate(lengths)
                                                             for j in range(n)]), flat), run_time=1.4)
            vo.wait_until("r")
            self.play(FadeIn(head), FadeIn(VGroup(*[r[0] for r in rects])))
            cells.save_state()
            self.play(LaggedStart(*anims, lag_ratio=0.004), cells.animate.set_opacity(0.25), run_time=2.2)
            for m in movers:
                self.remove(m)
            self.add(VGroup(*[r.block for r in rects]))
            vo.wait_until("a")
            boxes = VGroup(*[SurroundingRectangle(r.block, buff=0.05, color=C.ATTN, stroke_width=2) for r in rects])
            self.play(Create(boxes), run_time=1.0)
        self.rects, self.flat, self.fl, self.head, self.boxes = rects, flat, fl, head, boxes

    # ------------------------------------------------------------------
    def unpack(self):
        t = self.p["timing"]
        worst = max(x["max_abs_diff"] for x in t)
        exact = VGroup(
            label(r"then every token goes back to its own slot;", font_size=26),
            label(r"padding is restored as zeros", font_size=26),
            label(r"largest difference from the dense result: about $10^{-6}$ (float rounding)", font_size=26,
                  color=C.LIVE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18).to_corner(UL, buff=0.5)
        assert worst < 2e-6
        with self.voiceover(
            "Afterwards, every token is scattered back to its own slot, and the padding is restored as zeros. "
            "<bookmark mark='x'/> The rotary positions are the original ones, so the result is the same as the dense "
            "computation. We checked: the largest difference is about one part in a million, floating-point rounding."
        ) as vo:
            self.play(FadeOut(self.boxes), FadeOut(self.head), FadeOut(self.flat), FadeOut(self.fl),
                      Restore(self.cells), FadeOut(self.rects), run_time=1.4)
            self.play(FadeIn(exact[:2]))
            vo.wait_until("x")
            self.play(FadeIn(exact[2]))
        self.play(FadeOut(exact), FadeOut(self.cells))

    # ------------------------------------------------------------------
    def timing(self):
        p = self.p
        t = p["timing"]
        pl = Plot((0, 3), (0, 310), width=7.4, height=4.4, y_ticks=[0, 50, 100, 150, 200, 250, 300], font_size=22)
        pl.move_to(LEFT * 1.6 + DOWN * 0.3)
        bars = VGroup()
        labs = VGroup()
        for i, x in enumerate(t):
            b1 = pl.bar(i + 0.32, x["dense_ms"], 0.36, C.PADDED, opacity=0.9)
            b2 = pl.bar(i + 0.68, x["packed_ms"], 0.36, C.LIVE)
            v1 = label(f"{x['dense_ms']:.0f}", font_size=22, color=GREY_A).next_to(b1, UP, buff=0.06)
            v2 = label(f"{x['packed_ms']:.0f}", font_size=22).next_to(b2, UP, buff=0.06)
            sp = label(rf"{x['dense_ms'] / x['packed_ms']:.1f}$\times$", font_size=28, color=YELLOW)
            sp.move_to(pl.c2p(i + 0.5, 296))
            xl = label(rf"{x['mean_length']:.0f} live tokens\\per row, on average", font_size=20, color=GREY_A)
            xl.next_to(pl.c2p(i + 0.5, 0), DOWN, buff=0.15)
            bars.add(VGroup(b1, b2, v1, v2, sp, xl))
        yt = label(r"milliseconds", font_size=22, color=GREY_A).next_to(pl, UP, buff=0.15).align_to(pl, LEFT)
        leg = VGroup(
            VGroup(Square(0.25, stroke_width=0, fill_color=C.PADDED, fill_opacity=0.9), label(r"dense, masked", font_size=24)),
            VGroup(Square(0.25, stroke_width=0, fill_color=C.LIVE, fill_opacity=0.9), label(r"packed", font_size=24)),
        )
        for g in leg:
            g.arrange(RIGHT, buff=0.15)
        leg.arrange(DOWN, aligned_edge=LEFT, buff=0.2).next_to(pl, RIGHT, buff=0.6).shift(UP * 1.0)
        setup = label(rf"two encoder layers, {p['rows']} rows $\times$ {p['capacity']} slots,\\"
                      rf"$d_{{\mathrm{{model}}}}={p['d_model']}$, {p['threads']} CPU threads", font_size=22, color=GREY_A)
        setup.next_to(leg, DOWN, buff=0.5).align_to(leg, LEFT)
        tag = real_tag(r"timed with relflow's own packing code on the computer that rendered this video")
        sp0 = t[0]["dense_ms"] / t[0]["packed_ms"]
        sp1, sp2 = (t[i]["dense_ms"] / t[i]["packed_ms"] for i in (1, 2))
        assert sp0 > sp1 > sp2 > 1.5
        say = {2: "twice", 3: "three times"}
        with self.voiceover(
            "And it's fast. <bookmark mark='s'/> Here's relflow's own packing code, timed on this machine: two encoder "
            "layers over 256 rows, with room for 128 tokens each. <bookmark mark='a'/> When rows hold about four live "
            f"tokens on average, dense attention takes {t[0]['dense_ms']:.0f} milliseconds; packed, "
            f"{t[0]['packed_ms']:.0f}. {sp0:.0f} times faster. <bookmark mark='b'/> At twelve tokens it's "
            f"{sp1:.0f} times faster, <bookmark mark='c'/> and even at thirty-two, a quarter full, packing is "
            f"{say.get(round(sp2), f'{sp2:.1f} times')} as fast."
        ) as vo:
            vo.wait_until("s")
            self.play(Create(pl), FadeIn(yt), FadeIn(leg), FadeIn(setup), FadeIn(tag))
            for mark, b in zip("abc", bars):
                vo.wait_until(mark)
                self.play(GrowFromEdge(b[0], DOWN), GrowFromEdge(b[1], DOWN), FadeIn(b[2]), FadeIn(b[3]), FadeIn(b[5]))
                self.play(FadeIn(b[4], scale=1.3), run_time=0.5)
        self.play(FadeOut(VGroup(pl, bars, yt, leg, setup, tag)))

    # ------------------------------------------------------------------
    def fine_print(self):
        items = VGroup(
            label(r"the fine print", font_size=30),
            label(r"$\cdot$ the grouping costs one trip from the GPU to the host per encoder stack, to read the lengths",
                  font_size=24, color=GREY_A),
            label(r"$\cdot$ packing steps aside when dropout is active, so random draws match the dense schedule",
                  font_size=24, color=GREY_A),
            label(r"$\cdot$ and when anyone has hooked into or replaced the modules, so they still see dense tensors",
                  font_size=24, color=GREY_A),
            label(r"$\cdot$ rows with no live tokens at all are skipped entirely, in every encoder and pool",
                  font_size=24, color=GREY_A),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to(ORIGIN)
        with self.voiceover(
            "A little fine print. <bookmark mark='a'/> Grouping needs one trip from the GPU back to the host per "
            "encoder stack, to read the lengths. <bookmark mark='b'/> Packing steps aside when dropout is active, so "
            "the random numbers match the dense schedule exactly, <bookmark mark='c'/> and when anyone has hooked "
            "into or replaced the modules, so they keep seeing the dense shapes they expect. <bookmark mark='d'/> And "
            "rows with nothing in them at all are skipped entirely, everywhere."
        ) as vo:
            self.play(FadeIn(items[0]))
            for mark, it in zip("abcd", items[1:]):
                vo.wait_until(mark)
                self.play(FadeIn(it, shift=RIGHT * 0.1))
        self.wait(0.4)
        self.play(FadeOut(items))
