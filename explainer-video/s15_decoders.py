from __future__ import annotations

from explainer import *  # noqa: F403
from videos.ml.relflow.common import (
    D,
    Mono,
    Tree,
    label,
    mtex,
    note,
    real_tag,
    schematic_tag,
    show_episode_card,
    state_tile,
    tile,
)
from videos.ml.relflow.s01_hook import CUSTOMER_TREE
from videos.ml.relflow.story import stream, transmit, tree_up


class Decoders(VoiceoverScene):
    def construct(self):
        show_episode_card(self)
        self.heritage()
        self.queries()
        self.heads()

    # ------------------------------------------------------------------
    def heritage(self):
        tree = Tree(CUSTOMER_TREE, font_size=20, x_gap=0.45, y_gap=0.3).to_edge(LEFT, buff=0.4)
        nodes = {n["address"]: n for n in D("schema_current")["customer"]["nodes"]}
        assert nodes["/churned"]["heritage"] == ["/", "/churned"]
        assert nodes["/orders/line_items/sku"]["heritage"] == [
            "/",
            "/orders",
            "/orders/line_items",
            "/orders/line_items/sku",
        ]

        def path(addrs, color):
            hl = VGroup(
                *[
                    SurroundingRectangle(tree.cards[a], buff=0.07, color=color, stroke_width=3, corner_radius=0.1)
                    for a in addrs
                ]
            )
            edges = VGroup(*[tree.edge_to[a].copy().set_stroke(color, 4) for a in addrs if a in tree.edge_to])
            return VGroup(hl, edges)

        p1 = path(["/", "/churned"], C.RF_TARGET)
        p2 = path(["/", "/orders", "/orders/line_items", "/orders/line_items/sku"], YELLOW)
        t1 = VGroup(
            label(r"\texttt{churned} reads:", font_size=24, color=C.RF_TARGET),
            label(r"the root's 4 vectors", font_size=24, color=GREY_A),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        t2 = VGroup(
            label(r"a line item's \texttt{sku} reads:", font_size=24, color=YELLOW),
            label(r"the root's 4 vectors", font_size=24, color=GREY_A),
            label(r"+ the order summaries", font_size=24, color=GREY_A),
            label(r"+ every order's item summaries", font_size=24, color=GREY_A),
            label(r"+ its own (masked) slots", font_size=24, color=GREY_A),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        VGroup(t1, t2).arrange(DOWN, aligned_edge=LEFT, buff=0.5).to_edge(RIGHT, buff=0.3)
        tag = real_tag(r"heritage paths read from \texttt{model.schema}, relflow 0.3.2")
        with self.voiceover(
            "In a relflow model, encoders carry information up the tree, from the leaves to the root. "
            "Decoders read it back down. <bookmark mark='h'/> Each target's "
            "decoder looks at the representations along its own path to the root: its heritage. <bookmark mark='c'/> "
            "For churned, that's simply the root's four vectors. <bookmark mark='s'/> For a field deep in the tree, "
            "like a line item's SKU, it's the root, the order summaries, the item summaries of every order, and the "
            "field's own slots."
        ) as vo:
            self.play(FadeIn(tree), FadeIn(tag))
            tree_up(self, tree, run_time=1.5)
            vo.wait_until("c")
            self.play(Create(p1), FadeIn(t1))
            stream(self, [tree.edge_to["/churned"]], C.RF_TARGET)
            vo.wait_until("s")
            self.play(Create(p2), FadeIn(t2, lag_ratio=0.2), run_time=2)
            stream(
                self,
                [tree.edge_to[a] for a in ("/orders", "/orders/line_items", "/orders/line_items/sku")],
                YELLOW,
                run_time=1.7,
            )
        self.play(FadeOut(VGroup(tree, p1, p2, t1, t2, tag)))

    # ------------------------------------------------------------------
    def queries(self):
        from videos.ml.relflow.story import transmit

        mem = (
            VGroup(
                *[
                    VGroup(*[tile(col, 0.3) for _ in range(n)]).arrange(RIGHT, buff=0.06)
                    for n, col in [(4, C.RF_ROOT), (4, C.MASS), (8, C.RF_BRANCH), (4, C.MASKED)]
                ]
            )
            .arrange(RIGHT, buff=0.42)
            .move_to(DOWN * 1.65)
        )
        ml = VGroup(
            *[
                Mono(t, font_size=18, color=GREY_A).next_to(m, DOWN, buff=0.18)
                for t, m in zip(["root", "orders", "item summaries", "own slots"], mem)
            ]
        )
        qs = VGroup(*[tile(C.QUERY, 0.48) for _ in range(3)]).arrange(RIGHT, buff=2.7).move_to(UP * 0.7)
        positions = VGroup(
            *[
                Mono(f"item {i + 1}: SKU?", font_size=23, color=C.QUERY).next_to(q, UP, buff=0.22)
                for i, q in enumerate(qs)
            ]
        )
        siblings = VGroup(
            *[
                Mono(f"quantity {n}", font_size=22, color=C.NUMBER).next_to(q, UP, buff=1.05)
                for q, n in zip(qs, [1, 4, 2])
            ]
        )
        head = label("One learned question. Different local context.", font_size=37).to_edge(UP, buff=0.4)
        note_ = label(
            r"shared query parameters; skipped own slots are absent from memory", font_size=22, color=GREY_A
        ).move_to(DOWN * 2.7)
        counts = note(
            r"1 query for churned; fully populated capped SKU: $64\times16=1{,}024$ positions", font_size=21
        ).to_edge(DOWN, buff=0.45)
        tag = schematic_tag(corner=UR)
        reads = [
            [mem[0][0], mem[1][1], mem[2][0]],
            [mem[0][2], mem[1][2], mem[2][3]],
            [mem[0][1], mem[1][3], mem[2][6]],
        ]
        routes = VGroup(
            *[
                Line(q.get_bottom(), m.get_top(), color=C.ATTN, stroke_width=2, stroke_opacity=0.4)
                for q, keys in zip(qs, reads)
                for m in keys
            ]
        )
        with self.voiceover(
            "A decoder lays its heritage out as one long memory. <bookmark mark='m'/> Root vectors, order "
            "summaries, item summaries and the field's available own slots remain distinct sources. "
            "<bookmark mark='q'/> Learned queries align to the current target positions. The same learned "
            "question is reused: one for churned, or one per runtime item position for SKU. A fully populated "
            "capped example has sixty four times sixteen positions. <bookmark mark='s'/> Before reading, each "
            "query receives the visible siblings at its own coordinate. The quantity four belongs to item two; "
            "it does not get mixed into item one's question. <bookmark mark='a'/> Now cross-attention gathers "
            "values from the heritage into each query. Shared parameters can produce different reads because "
            "the local context differs. These routes illustrate the mechanism; they are not measured attention weights."
        ) as vo:
            self.play(FadeIn(head), FadeIn(tag))
            vo.wait_until("m")
            self.play(FadeIn(mem, lag_ratio=0.15), FadeIn(ml))
            vo.wait_until("q")
            self.play(FadeIn(qs), FadeIn(positions), FadeIn(counts), FadeIn(note_))
            vo.wait_until("s")
            self.play(FadeIn(siblings))
            for sibling, q in zip(siblings, qs):
                transmit(self, [sibling], q, C.NUMBER, run_time=0.55)
            self.play(
                *[q.sq.animate.set_fill(interpolate_color(ManimColor(C.QUERY), ManimColor(C.NUMBER), 0.3)) for q in qs]
            )
            vo.wait_until("a")
            self.play(Create(routes))
            for q, keys in zip(qs, reads):
                transmit(self, keys, q, C.ATTN, run_time=0.85)
        self.clear_scene()

    # ------------------------------------------------------------------
    def heads(self):
        pooled = VGroup(*[tile(C.QUERY, 0.32) for _ in range(1)]).move_to(LEFT * 5 + UP * 0.3)
        pl = label(r"pooled\\vector", font_size=22, color=GREY_A).next_to(pooled, DOWN, buff=0.2)
        state = VGroup(*[state_tile(k, 0.34) for k in range(5)]).arrange(RIGHT, buff=0.08)
        rows = VGroup(
            VGroup(label(r"every type:", font_size=24), state, label(r"5-way state", font_size=22, color=GREY_A)),
            VGroup(
                label(r"Number:", font_size=24, color=C.NUMBER),
                mtex(r"\hat x", font_size=34),
                label(r"a value in source units", font_size=22, color=GREY_A),
            ),
            VGroup(
                label(r"Boolean:", font_size=24, color=C.BOOLEAN),
                mtex(r"p(\text{true})", font_size=30),
                label(r"a probability", font_size=22, color=GREY_A),
            ),
            VGroup(
                label(r"Category:", font_size=24, color=C.CATEGORY),
                mtex(r"\operatorname{softmax}", font_size=30),
                label(r"over the labels it has discovered", font_size=22, color=GREY_A),
            ),
        )
        for r in rows:
            r.arrange(RIGHT, buff=0.3)
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to(RIGHT * 0.6 + UP * 0.3)
        arrows = VGroup(
            *[
                Arrow(
                    pooled.get_right(),
                    r.get_left(),
                    buff=0.15,
                    color=GREY_C,
                    stroke_width=2,
                    max_tip_length_to_length_ratio=0.08,
                )
                for r in rows
            ]
        )
        loss = mtex(
            r"\mathcal{L} = \sum_{\text{fields}} w_f \Big[\, \mathrm{CE}(\text{state}) + "
            r"\mathcal{L}_{\text{content}} \,\Big]_{\text{selected coordinates}}",
            font_size=34,
        ).to_edge(DOWN, buff=0.7)
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
            for r in rows:
                transmit(self, [pooled], r[1], C.RF_TARGET, run_time=0.4)
            vo.wait_until("l")
            self.play(Write(loss), run_time=1.6)
        self.play(FadeOut(VGroup(pooled, pl, rows, arrows, loss)))
