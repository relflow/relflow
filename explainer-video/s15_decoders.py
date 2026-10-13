from __future__ import annotations

import numpy as np

from explainer import *  # noqa: F403
from videos.relflow.common import (
    D, Mono, Tree, chip, label, mtex, note, real_tag, schematic_tag, show_episode_card, state_tile, tile,
)
from videos.relflow.s01_hook import CUSTOMER_TREE


class Decoders(VoiceoverScene):
    def construct(self):
        show_episode_card(self)
        self.heritage()
        self.queries()
        self.heads()

    # ------------------------------------------------------------------
    def heritage(self):
        tree = Tree(CUSTOMER_TREE, font_size=20, x_gap=0.45, y_gap=0.3).to_edge(LEFT, buff=0.4)
        nodes = {n["address"]: n for n in D("schema")["customer"]["nodes"]}
        assert nodes["/churned"]["heritage"] == ["/", "/churned"]
        assert nodes["/orders/line_items/sku"]["heritage"] == ["/", "/orders", "/orders/line_items", "/orders/line_items/sku"]

        def path(addrs, color):
            hl = VGroup(*[SurroundingRectangle(tree.cards[a], buff=0.07, color=color, stroke_width=3, corner_radius=0.1)
                          for a in addrs])
            edges = VGroup(*[tree.edge_to[a].copy().set_stroke(color, 4) for a in addrs if a in tree.edge_to])
            return VGroup(hl, edges)

        p1 = path(["/", "/churned"], C.RF_TARGET)
        p2 = path(["/", "/orders", "/orders/line_items", "/orders/line_items/sku"], YELLOW)
        t1 = VGroup(label(r"\texttt{churned} reads:", font_size=24, color=C.RF_TARGET),
                    label(r"the root's 4 vectors", font_size=24, color=GREY_A)).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        t2 = VGroup(label(r"a line item's \texttt{sku} reads:", font_size=24, color=YELLOW),
                    label(r"the root's 4 vectors", font_size=24, color=GREY_A),
                    label(r"+ the order summaries", font_size=24, color=GREY_A),
                    label(r"+ every order's item summaries", font_size=24, color=GREY_A),
                    label(r"+ its own (masked) slots", font_size=24, color=GREY_A)).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        VGroup(t1, t2).arrange(DOWN, aligned_edge=LEFT, buff=0.5).to_edge(RIGHT, buff=0.3)
        tag = real_tag(r"heritage paths read from \texttt{model.schema}")
        with self.voiceover(
            "In a relflow model, encoders carry information up the tree, from the leaves to the root. "
            "Decoders read it back down. <bookmark mark='h'/> Each target's "
            "decoder looks at the representations along its own path to the root: its heritage. <bookmark mark='c'/> "
            "For churned, that's simply the root's four vectors. <bookmark mark='s'/> For a field deep in the tree, "
            "like a line item's SKU, it's the root, the order summaries, the item summaries of every order, and the "
            "field's own slots."
        ) as vo:
            self.play(FadeIn(tree), FadeIn(tag))
            vo.wait_until("c")
            self.play(Create(p1), FadeIn(t1))
            vo.wait_until("s")
            self.play(Create(p2), FadeIn(t2, lag_ratio=0.2), run_time=2)
        self.play(FadeOut(VGroup(tree, p1, p2, t1, t2, tag)))

    # ------------------------------------------------------------------
    def queries(self):
        mem = VGroup(
            VGroup(*[tile(C.RF_ROOT, 0.28) for _ in range(4)]).arrange(RIGHT, buff=0.05),
            VGroup(*[tile(C.MASS, 0.28) for _ in range(4)]).arrange(RIGHT, buff=0.05),
            VGroup(*[tile(C.MASS, 0.28, fill=0.6) for _ in range(12)]).arrange(RIGHT, buff=0.05),
            VGroup(*[tile(C.MASKED, 0.28, fill=0.5) for _ in range(8)]).arrange(RIGHT, buff=0.05),
        ).arrange(RIGHT, buff=0.3).move_to(DOWN * 1.8)
        ml = VGroup(*[label(t, font_size=18, color=GREY_A).next_to(m, DOWN, buff=0.12) for t, m in
                      zip([r"root", r"orders", r"item summaries", r"own slots"], mem)])
        mem_l = label(r"the heritage: one long memory", font_size=24, color=GREY_A).next_to(mem, UP, buff=0.25)
        qs = VGroup(*[tile(C.QUERY, 0.3) for _ in range(8)]).arrange(RIGHT, buff=0.1).move_to(UP * 1.6)
        ql = label(r"learned queries shaped like the target: one per slot", font_size=24, color=C.QUERY).next_to(qs, UP, buff=0.2)
        lines = VGroup()
        rng = np.random.default_rng(0)
        flat = [t for g in mem for t in g]
        for q in qs:
            for k in rng.choice(len(flat), 5, replace=False):
                lines.add(Line(q.get_bottom(), flat[k].get_top(), color=C.ATTN, stroke_width=1.2).set_stroke(opacity=0.45))
        sib = VGroup(*[tile(C.NUMBER, 0.24) for _ in range(8)]).arrange(RIGHT, buff=0.16).next_to(qs, UP, buff=0.0)
        sib.move_to(qs.get_center() + UP * 0.0)
        sib_row = VGroup(*[tile(C.NUMBER, 0.24) for _ in range(8)])
        for t, q in zip(sib_row, qs):
            t.next_to(q, RIGHT, buff=0.06).shift(UP * 0.25)
        sib_l = label(r"each query is nudged by the visible siblings at its coordinate\\"
                      r"(predicting an item's SKU? start from that item's quantity)", font_size=22, color=C.NUMBER)
        sib_l.to_edge(UP, buff=0.25)
        counts = note(r"\texttt{churned}: 1 query.\quad a line item's \texttt{sku}: $64 \times 16 = 1{,}024$ queries, with positions",
                      font_size=22).to_edge(DOWN, buff=0.3)
        tag = schematic_tag(corner=UR)
        with self.voiceover(
            "The decoder doesn't average its heritage. <bookmark mark='m'/> It lays it out as one long memory, "
            "<bookmark mark='q'/> and it brings learned queries of its own, shaped like the target: a single query for "
            "churned, or one for each of the 64 times 16 item slots, for SKU. <bookmark mark='a'/> Each query "
            "cross-attends into the memory and pulls out what it needs. <bookmark mark='s'/> And before it does, each "
            "query is nudged by the visible siblings at its own coordinate. When the decoder predicts an item's SKU, "
            "the query for that slot starts out knowing the item's quantity."
        ) as vo:
            vo.wait_until("m")
            self.play(FadeIn(mem, lag_ratio=0.05), FadeIn(ml), FadeIn(mem_l), FadeIn(tag))
            vo.wait_until("q")
            self.play(FadeIn(qs, lag_ratio=0.1), FadeIn(ql))
            self.play(FadeIn(counts))
            vo.wait_until("a")
            self.play(Create(lines), run_time=1.5)
            vo.wait_until("s")
            self.play(FadeIn(sib_row, lag_ratio=0.1), FadeIn(sib_l), FadeOut(ql))
            self.play(*[sr.animate.move_to(q) .set_opacity(0) for sr, q in zip(sib_row, qs)],
                      *[q.sq.animate.set_fill(interpolate_color(ManimColor(C.QUERY), ManimColor(C.NUMBER), 0.35))
                        for q in qs], run_time=1.2)
        self.play(FadeOut(VGroup(mem, ml, mem_l, qs, lines, sib_row, sib_l, counts, tag)))

    # ------------------------------------------------------------------
    def heads(self):
        pooled = VGroup(*[tile(C.QUERY, 0.32) for _ in range(1)]).move_to(LEFT * 5 + UP * 0.3)
        pl = label(r"pooled\\vector", font_size=22, color=GREY_A).next_to(pooled, DOWN, buff=0.2)
        state = VGroup(*[state_tile(k, 0.34) for k in range(5)]).arrange(RIGHT, buff=0.08)
        rows = VGroup(
            VGroup(label(r"every type:", font_size=24), state, label(r"5-way state", font_size=22, color=GREY_A)),
            VGroup(label(r"Number:", font_size=24, color=C.NUMBER), mtex(r"\hat x", font_size=34),
                   label(r"a value in source units", font_size=22, color=GREY_A)),
            VGroup(label(r"Boolean:", font_size=24, color=C.BOOLEAN), mtex(r"p(\text{true})", font_size=30),
                   label(r"a probability", font_size=22, color=GREY_A)),
            VGroup(label(r"Category:", font_size=24, color=C.CATEGORY), mtex(r"\operatorname{softmax}", font_size=30),
                   label(r"over the labels it has discovered", font_size=22, color=GREY_A)),
        )
        for r in rows:
            r.arrange(RIGHT, buff=0.3)
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to(RIGHT * 0.6 + UP * 0.3)
        arrows = VGroup(*[Arrow(pooled.get_right(), r.get_left(), buff=0.15, color=GREY_C, stroke_width=2,
                                max_tip_length_to_length_ratio=0.08) for r in rows])
        loss = mtex(r"\mathcal{L} = \sum_{\text{fields}} w_f \Big[\, \mathrm{CE}(\text{state}) + "
                    r"\mathcal{L}_{\text{content}} \,\Big]_{\text{selected coordinates}}", font_size=34).to_edge(DOWN, buff=0.7)
        with self.voiceover(
            "Then a small head for each datatype. <bookmark mark='s'/> Every decoder predicts the five-way state, so "
            "the model can say that a value will be null. <bookmark mark='c'/> A Number adds a value, a Boolean a "
            "probability, and a Category a distribution over the labels it has discovered so far. "
            "<bookmark mark='l'/> The loss adds the state's cross-entropy and the content's own loss, counted only on "
            "the coordinates selected for reconstruction, and each field's total can be weighted."
        ) as vo:
            self.play(FadeIn(pooled), FadeIn(pl))
            vo.wait_until("s")
            self.play(GrowArrow(arrows[0]), FadeIn(rows[0]))
            vo.wait_until("c")
            for a, r in zip(arrows[1:], rows[1:]):
                self.play(GrowArrow(a), FadeIn(r), run_time=0.6)
            vo.wait_until("l")
            self.play(Write(loss), run_time=1.6)
        self.play(FadeOut(VGroup(pooled, pl, rows, arrows, loss)))
