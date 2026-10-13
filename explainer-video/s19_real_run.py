from __future__ import annotations

import numpy as np

from explainer import *  # noqa: F403
from videos.relflow.common import (
    D, Mono, Plot, code, label, mtex, note, real_tag, show_episode_card, tile,
)

SCHEMA_CODE = """
model = rf.Model(
    d_model=64, n_layers=1, n_heads=4, embed=True,
    line_items=rf.Branch(
        length=8,
        sku=rf.Category,
        quantity=rf.Number,
        price=rf.Number,
    ),
    returned=rf.Boolean(mask=True),
)
"""


class RealRun(VoiceoverScene):
    def construct(self):
        show_episode_card(self)
        self.r = D("returns")
        self.setup_()
        self.training()
        self.attention()
        self.embedding()
        self.output()

    # ------------------------------------------------------------------
    def setup_(self):
        r = self.r
        c = code(SCHEMA_CODE, font_size=21).to_corner(UL, buff=0.6)
        rule = VGroup(
            label(r"the hidden rule:", font_size=28),
            label(r"an order comes back if it has a fragile SKU,", font_size=26, color=GREY_A),
            label(r"S03 or S11, bought 3 or more at a time", font_size=26, color=GREY_A),
            label(rf"{r['train_rows']:,} training orders, 20 SKUs,".replace(",", "{,}", 1), font_size=24, color=GREY_B),
            label(r"1 to 8 items each, random prices", font_size=24, color=GREY_B),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15).to_corner(DR, buff=0.7)
        with self.voiceover(
            "Let's watch a real model learn. <bookmark mark='s'/> Here's the schema: orders with up to "
            "eight line items, each with a SKU, a quantity and a price, and one hidden Boolean: was the order returned? "
            "<bookmark mark='r'/> The rule, which the model never sees: an order comes back if it contains a fragile "
            "SKU, S03 or S11, bought three or more at a time."
        ) as vo:
            vo.wait_until("s")
            self.play(FadeIn(c, shift=RIGHT * 0.2))
            vo.wait_until("r")
            self.play(FadeIn(rule, lag_ratio=0.2))
        self.play(FadeOut(c), FadeOut(rule))

    # ------------------------------------------------------------------
    def training(self):
        r = self.r
        loss = np.array(r["loss"])
        steps = np.arange(1, len(loss) + 1)
        pl = Plot((0, 130), (0, 1.6), width=7.0, height=4.0, x_ticks=[0, 32, 64, 96, 128], y_ticks=[0, 0.5, 1.0, 1.5],
                  font_size=22).move_to(LEFT * 2.4 + DOWN * 0.2)
        curve = pl.line(steps, loss, C.LOSS if hasattr(C, "LOSS") else RED, 3)
        curve.set_stroke(color=ORANGE)
        xt = pl.x_title(r"training step (batches of 128 orders)", font_size=22)
        yt = label(r"training loss", font_size=22, color=ORANGE).next_to(pl, UP, buff=0.15).align_to(pl, LEFT)
        ep_marks = VGroup()
        for e in r["epochs"]:
            ep_marks.add(VGroup(pl.vline(e["step"], color=GREY_D, stroke_width=1.5),
                                label(rf"AUC {e['auc']:.2f}", font_size=20, color=C.VALUED).move_to(pl.c2p(e["step"], 1.45))
                                .shift(LEFT * 0.42)))
        res = r["results"]["intact"]
        assert r["seconds"] < 30, "time the run on an idle CPU"
        side = VGroup(
            label(rf"{r['seconds']:.0f} seconds on this CPU", font_size=28),
            label(r"validation AUC after each epoch:", font_size=24, color=GREY_A),
            label(r", ".join(f"{e['auc']:.2f}" for e in r["epochs"]), font_size=28, color=C.VALUED),
            label(rf"held-out test: AUC {res['auc']:.3f}, accuracy {res['accuracy']:.3f}", font_size=24, color=GREY_A),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).to_edge(RIGHT, buff=0.5)
        tag = real_tag(r"real training run, relflow 0.2.3 with Lightning, CPU only")
        assert r["epochs"][1]["auc"] > 0.999 and res["auc"] > 0.999
        misses = round((1 - res["accuracy"]) * r["test_rows"])
        with self.voiceover(
            "We train on four thousand synthetic orders, on this machine's processor. <bookmark mark='t'/> The whole run "
            f"takes about {round(r['seconds'])} seconds. <bookmark mark='l'/> The loss falls fast, <bookmark mark='a'/> and by the end "
            "of the second epoch, validation AUC has reached one, and it stays there. On two thousand fresh test "
            "orders, it ranks every returned order above every kept one. "
            + ("And at a fifty percent threshold, it doesn't miss a single one." if misses == 0 else
               f"At a fifty percent threshold, {misses} of them still land on the wrong side.")
        ) as vo:
            self.play(Create(pl), FadeIn(xt), FadeIn(yt), FadeIn(tag))
            vo.wait_until("t")
            self.play(FadeIn(side[0]))
            vo.wait_until("l")
            self.play(Create(curve), run_time=2.5, rate_func=linear)
            vo.wait_until("a")
            self.play(FadeIn(ep_marks, lag_ratio=0.3), FadeIn(side[1:], lag_ratio=0.3), run_time=1.5)
        self.play(FadeOut(VGroup(pl, curve, xt, yt, ep_marks, side, tag)))

    # ------------------------------------------------------------------
    def order_view(self, order, weights, prob, cell=0.42):
        w = np.array(weights)  # heads x 24
        items = order["line_items"]
        n = len(items)
        heads = VGroup()
        for h in (0, 1):
            grid = VGroup()
            for f in range(3):
                for j in range(8):
                    v = w[h, 8 * f + j]
                    col = C.ATTN
                    sq_ = Square(cell, stroke_width=0.8, stroke_color=GREY_D,
                                 fill_color=col if j < n else C.PADDED, fill_opacity=(0.06 + 0.94 * v) if j < n else 0.2)
                    grid.add(sq_)
            grid.arrange_in_grid(3, 8, buff=0.04)
            hl = label(rf"head {h}", font_size=20, color=GREY_A).next_to(grid, LEFT, buff=0.95)
            heads.add(VGroup(grid, hl))
        heads.arrange(DOWN, buff=0.3)
        fields = VGroup(*[Mono(t, font_size=14, color=c).next_to(heads[0][0][8 * k], LEFT, buff=0.08)
                          for k, (t, c) in enumerate((("sku", C.CATEGORY), ("qty", C.NUMBER), ("price", C.NUMBER)))])
        cap = VGroup()
        for j, it in enumerate(items[:8]):
            frag = it["sku"] in ("S03", "S11")
            col = (C.BOOLEAN if it["quantity"] >= 3 else YELLOW) if frag else GREY_A
            cap.add(Mono(f"{it['sku']}", font_size=15, color=col).next_to(heads[1][0][16 + j], DOWN, buff=0.12))
            cap.add(Mono(f"×{it['quantity']}", font_size=15, color=col).next_to(cap[-1], DOWN, buff=0.04))
        p = label(rf"$P(\text{{returned}}) = {prob:.3f}$", font_size=26,
                  color=C.RF_TARGET if prob > 0.5 else GREY_A).next_to(heads, UP, buff=0.25)
        return VGroup(heads, fields, cap, p)

    def attention(self):
        a = self.r["attention"]
        orders, weights, probs = a["orders"], a["weights"], a["prob"]
        w0 = np.array(weights[0])
        hit0 = [i for i, it in enumerate(orders[0]["line_items"]) if it["sku"] in ("S03", "S11") and it["quantity"] >= 3][0]
        assert w0[0, hit0] > 0.9 and w0[1, hit0] > 0.9
        v = [self.order_view(o, w, p) for o, w, p in zip(orders, weights, probs)]
        v[0].move_to(LEFT * 3.4 + DOWN * 0.2)
        v[1].move_to(RIGHT * 3.4 + DOWN * 0.2)
        head = label(r"where the line-item summary query looks: real attention weights, two of its four heads", font_size=26)
        head.to_edge(UP, buff=0.3)
        mark = SurroundingRectangle(VGroup(v[0][0][0][0][hit0], v[0][0][1][0][hit0]), buff=0.05, color=YELLOW,
                                    stroke_width=3)
        hit1 = [i for i, it in enumerate(orders[1]["line_items"]) if it["sku"] in ("S03", "S11") and it["quantity"] >= 3][0]
        w1 = np.array(weights[1])
        item1 = [w1[h, hit1] + w1[h, 8 + hit1] + w1[h, 16 + hit1] for h in (0, 1)]
        assert min(item1) > 0.9
        sku_only = all(w1[h, hit1] > 0.9 for h in (0, 1))
        second = ("the same two heads go straight to it." if sku_only else
                  "the same two heads go straight to that item again, though one of them now reads its quantity "
                  "token rather than its SKU.")
        mark1 = SurroundingRectangle(VGroup(*[v[1][0][h][0][8 * f + hit1] for h in (0, 1) for f in range(3)]), buff=0.05,
                                     color=YELLOW, stroke_width=3)
        legend = note(r"orange = attention weight, recomputed from the trained model's own projections;\\"
                      r"pink SKU = fragile and bought 3+;  yellow = fragile, bought once", font_size=20).to_edge(DOWN, buff=0.2)
        tag = VGroup()
        with self.voiceover(
            "But how does it decide? We can open up the line items branch, and recompute the attention weights of its "
            "learned summary query. <bookmark mark='a'/> Here's an order that came back: S11, four at a time, in the "
            "fifth slot. <bookmark mark='h'/> Two of the four heads put essentially all of their weight on one single "
            "token: the SKU of exactly that item. And thanks to coordinate attention, that SKU token already knows its "
            "own quantity. <bookmark mark='b'/> Here's a second returned order, with the fragile item in the first "
            f"slot: {second}"
        ) as vo:
            self.play(FadeIn(head), FadeIn(tag), FadeIn(legend))
            vo.wait_until("a")
            self.play(FadeIn(v[0]))
            vo.wait_until("h")
            self.play(Create(mark))
            vo.wait_until("b")
            self.play(FadeIn(v[1]))
            self.play(Create(mark1))
        self.play(FadeOut(VGroup(v[0], v[1], mark, mark1)))
        v[2].move_to(LEFT * 3.4 + DOWN * 0.2)
        v[3].move_to(RIGHT * 3.4 + DOWN * 0.2)
        with self.voiceover(
            "And here are two orders that were kept. <bookmark mark='k'/> Each one contains a fragile SKU, but bought "
            "only once. There's no single decisive item, and the attention spreads out and wanders: over quantities, "
            "over other products. The model says no, with 99 percent confidence."
        ) as vo:
            self.play(FadeIn(v[2]), FadeIn(v[3]))
            vo.wait_until("k")
            self.play(*[Indicate(m, color=YELLOW) for m in (v[2][2], v[3][2])], run_time=1.2)
        self.play(FadeOut(VGroup(v[2], v[3], head, tag, legend)))

    # ------------------------------------------------------------------
    def embedding(self):
        p = self.r["pca"]
        xy = np.array(p["xy"])
        y = np.array(p["label"], dtype=bool)
        assert xy[y, 0].min() > xy[~y, 0].max()
        pl = Plot((-0.45, 0.85), (-0.3, 0.3), width=7.6, height=4.2, x_ticks=[-0.4, 0, 0.4, 0.8], y_ticks=[-0.2, 0, 0.2],
                  font_size=20, grid=False).move_to(LEFT * 1.4 + DOWN * 0.3)
        pts_k = VGroup(*[pl.dot(a, b, color=GREY_B, radius=0.035, opacity=0.6) for (a, b), t in zip(xy, y) if not t])
        pts_r = VGroup(*[pl.dot(a, b, color=C.RF_TARGET, radius=0.035, opacity=0.8) for (a, b), t in zip(xy, y) if t])
        xt = pl.x_title(rf"first principal direction ({100 * p['explained'][0]:.0f}\% of the variance)", font_size=22)
        yt = label(r"second", font_size=20, color=GREY_A).next_to(pl, UP, buff=0.1).align_to(pl, LEFT)
        leg = VGroup(
            VGroup(Dot(color=C.RF_TARGET), label(r"returned", font_size=24, color=C.RF_TARGET)).arrange(RIGHT, buff=0.15),
            VGroup(Dot(color=GREY_B), label(r"kept", font_size=24, color=GREY_B)).arrange(RIGHT, buff=0.15),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).next_to(pl, RIGHT, buff=0.5).shift(UP * 1.0)
        side = label(r"one exported 64-d root\\vector per order\\(\texttt{embed=True}),\\600 test orders", font_size=22,
                     color=GREY_A).next_to(leg, DOWN, buff=0.5).align_to(leg, LEFT)
        tag = real_tag(r"PCA of the real exported embeddings")
        with self.voiceover(
            "Because we set embed equals True, the model also exports its root vector for every order. "
            "<bookmark mark='p'/> Project six hundred test orders onto their two main directions, <bookmark mark='s'/> "
            "and the returned orders sit cleanly apart from the kept ones, along the very first direction. That vector "
            "is what a downstream model, or a nearest-neighbor search, would use."
        ) as vo:
            vo.wait_until("p")
            self.play(Create(pl), FadeIn(xt), FadeIn(yt), FadeIn(tag), FadeIn(pts_k, lag_ratio=0.002), run_time=1.5)
            vo.wait_until("s")
            self.play(FadeIn(pts_r, lag_ratio=0.004), FadeIn(leg), FadeIn(side), run_time=1.5)
        self.play(FadeOut(VGroup(pl, pts_k, pts_r, xt, yt, leg, side, tag)))

    # ------------------------------------------------------------------
    def output(self):
        ex = self.r["example_prediction"]
        st, emb = ex["/returned"]["state"], ex["/"]
        assert abs(emb["embedding_norm"] - 1) < 1e-4 and emb["width"] == 64
        head = ", ".join(f"{v:.3f}" for v in emb["embedding_head"])
        lines = [
            ("predictions:", WHITE),
            ("  /returned:", C.RF_TARGET),
            (f"    state:     {{valued: {st['valued']:.4f}, null: {st['null']:.5f}, ...}}", GREY_A),
            (f"    content:   {{probability: {ex['/returned']['content']['probability']:.3f}}}", WHITE),
            (f"    inferred:  {str(ex['/returned']['inferred']).lower()}", GREY_A),
            ("  /:", C.RF_ROOT),
            (f"    embedding: [{head}, ...]  # {emb['width']} floats, norm 1", GREY_A),
        ]
        g = VGroup(*[Mono(t, font_size=22, color=c) for t, c in lines]).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        g.move_to(UP * 0.4)
        x0 = g.get_left()[0]
        cw = Mono("M" * 10, font_size=22).width / 10
        for (t, _), m in zip(lines, g):
            m.shift(RIGHT * (x0 + cw * (len(t) - len(t.lstrip(" "))) - m.get_left()[0]))
        foot = label(r"one Arrow table: every address carries a state, its decoded content, and an \texttt{inferred} flag;\\"
                     r"\texttt{rf.Writer} streams it straight to Parquet", font_size=24, color=GREY_A).to_edge(DOWN, buff=0.8)
        tag = real_tag(r"the real output row for the first order above")
        with self.voiceover(
            "And predictions come back as an Arrow table. <bookmark mark='a'/> For every address: a probability for "
            "each state, the decoded content, and a flag saying whether the value was inferred. Exported embeddings "
            "sit at their own addresses. <bookmark mark='w'/> In a batch job, a writer callback streams each batch of "
            "predictions straight to Parquet, so nothing piles up in memory."
        ) as vo:
            vo.wait_until("a")
            self.play(FadeIn(g, lag_ratio=0.1), FadeIn(tag), run_time=1.6)
            vo.wait_until("w")
            self.play(FadeIn(foot))
        self.wait(0.4)
        self.play(FadeOut(g), FadeOut(foot), FadeOut(tag))
