from __future__ import annotations

from explainer import *  # noqa: F403
from videos.relflow.common import (
    BRAND_INK, D, Card, Mono, Tree, chip, code, fmt_int, label, logo, note, real_tag, wordmark,
)

CUSTOMER_TREE = (
    "customer", "root", "Branch", [
        ("tenure_days", "field", "Number", []),
        ("orders", "branch", "Branch", [
            ("age_days", "field", "Number", []),
            ("line_items", "branch", "Branch", [
                ("sku", "field", "Category", []),
                ("quantity", "field", "Number", []),
            ]),
        ]),
        ("churned", "target", "Boolean", []),
        ("next_month_spend", "target", "Number", []),
    ],
)

YAML = [
    ("tenure_days: 420", "field"),
    ("orders:", "branch"),
    ("  - age_days: 3", "field"),
    ("    line_items:", "branch"),
    ("      - sku: A12", "field"),
    ("        quantity: 2", "field"),
    ("      - sku: B07", "field"),
    ("        quantity: 1", "field"),
    ("  - age_days: 41", "field"),
    ("    line_items:", "branch"),
    ("      - sku: C33", "field"),
    ("        quantity: 5", "field"),
    ("churned: false", "target"),
    ("next_month_spend: 75.0", "target"),
]

SHAPES = [
    ("a support ticket", [
        ("subject: Delayed transfer", "field"), ("messages:", "branch"), ("  - author: customer", "field"),
        ("    body: Not arrived", "field"), ("  - author: agent", "field"), ("    body: In review", "field"),
        ("escalated: true", "target"),
    ]),
    ("a transfer", [
        ("amount: 120.00", "field"), ("parties:", "branch"), ("  - account: acct-17", "field"),
        ("    role: payer", "field"), ("  - account: acct-83", "field"), ("    role: payee", "field"),
        ("flagged: false", "target"),
    ]),
    ("a sensor", [
        ("site: north-7", "field"), ("readings:", "branch"), ("  - hour: 0", "field"), ("    temp_c: 4.1", "field"),
        ("  - hour: 1", "field"), ("    temp_c: 3.8", "field"), ("fault: false", "target"),
    ]),
]

CODE = """
model = rf.Model.md(
    tenure_days=rf.Number,
    orders=rf.Branch(
        length=64,
        age_days=rf.Number,
        line_items=rf.Branch(
            length=16,
            sku=rf.Category,
            quantity=rf.Number,
        ),
    ),
    churned=rf.Boolean(mask=True),
    next_month_spend=rf.Number(mask=True),
)
"""


def yaml_block(font_size=24, lines=YAML) -> VGroup:
    """A record as YAML: keys colored by their role in the schema, indentation kept exactly."""
    col = {"field": C.RF_FIELD, "branch": C.RF_BRANCH, "target": C.RF_TARGET}
    char_w = Mono("M" * 20, font_size=font_size).width / 20
    rows = VGroup()
    for text, kind in lines:
        indent = len(text) - len(text.lstrip(" "))
        body = text.strip()
        parts = VGroup()
        if body.startswith("- "):
            parts.add(Mono("-", font_size=font_size, color=GREY_B))
            body = body[2:]
        key, _, val = body.partition(":")
        parts.add(Mono(key + ":", font_size=font_size, color=col[kind]))
        if val.strip():
            parts.add(Mono(val.strip(), font_size=font_size, color=WHITE))
        parts.arrange(RIGHT, buff=char_w * 0.9)
        parts.indent = indent
        rows.add(parts)
    rows.arrange(DOWN, aligned_edge=LEFT, buff=0.07)
    x0 = rows.get_left()[0]
    for r in rows:
        r.shift(RIGHT * (x0 + r.indent * char_w - r.get_left()[0]))
    return rows


class Hook(VoiceoverScene):
    def construct(self):
        self.title()
        self.shapes()
        self.record()
        self.flatten()
        self.schema()
        self.network()
        self.roadmap()

    # ------------------------------------------------------------------
    def title(self):
        mark = logo(2.2)
        word = wordmark(110)
        head = VGroup(mark, word).arrange(RIGHT, buff=0.5).move_to(UP * 0.5)
        sub = label(r"A model factory for structured data", font_size=38, color=GREY_A).next_to(head, DOWN, buff=0.6)
        self.play(FadeIn(mark, scale=0.9), run_time=1.4)
        self.play(Write(word), run_time=1.4)
        self.play(FadeIn(sub, shift=UP * 0.15), run_time=1.0)
        self.wait(1.4)
        with self.voiceover(
            "This is relflow: a model factory for structured data. You describe the shape of one record, and it "
            "builds the PyTorch model that reads it. In this series, we'll take it apart, one short episode at a "
            "time, and watch every piece work."
        ):
            pass
        self.play(FadeOut(VGroup(mark, word, sub), shift=UP * 0.2), run_time=0.8)

    # ------------------------------------------------------------------
    def shapes(self):
        blocks = VGroup()
        for name, lines in SHAPES:
            y = yaml_block(20, lines)
            cap = label(name, font_size=26, color=GREY_A).next_to(y, UP, buff=0.3).align_to(y, LEFT)
            blocks.add(VGroup(cap, y))
        blocks.arrange(RIGHT, buff=0.75, aligned_edge=UP).move_to(UP * 0.2)
        lists = VGroup(*[SurroundingRectangle(VGroup(*b[1][2:6]), buff=0.08, color=C.RF_BRANCH, stroke_width=1.8,
                                              corner_radius=0.06) for b in blocks])
        answers = VGroup(*[SurroundingRectangle(b[1][6], buff=0.06, color=C.RF_TARGET, stroke_width=1.8,
                                                corner_radius=0.06) for b in blocks])
        with self.voiceover(
            "Most data worth learning from has shape. <bookmark mark='a'/> A support ticket, with its thread of "
            "messages. <bookmark mark='b'/> A transfer, between two parties. <bookmark mark='c'/> A sensor, with its "
            "readings. <bookmark mark='l'/> Each is a record with a list inside it, <bookmark mark='q'/> and a value "
            "we'd like to know."
        ) as vo:
            for mark, b in zip("abc", blocks):
                vo.wait_until(mark)
                self.play(FadeIn(b, shift=UP * 0.15, lag_ratio=0.05), run_time=0.9)
            vo.wait_until("l")
            self.play(LaggedStart(*[Create(r) for r in lists], lag_ratio=0.2), run_time=1.0)
            vo.wait_until("q")
            self.play(LaggedStart(*[Create(r) for r in answers], lag_ratio=0.2), run_time=1.0)
        self.wait(0.4)
        self.play(FadeOut(VGroup(blocks, lists, answers)), run_time=0.8)

    # ------------------------------------------------------------------
    def record(self):
        y = yaml_block(26).move_to(LEFT * 2.6)
        head = note(r"one observation, written as YAML", color=GREY_A, font_size=24).next_to(y, UP, buff=0.35)
        head.align_to(y, LEFT)
        orders = VGroup(*y[2:12])
        items = [VGroup(*y[4:6]), VGroup(*y[6:8]), VGroup(*y[10:12])]
        b_orders = SurroundingRectangle(orders, buff=0.1, color=C.RF_BRANCH, stroke_width=2, corner_radius=0.08)
        b_items = VGroup(*[SurroundingRectangle(it, buff=0.06, color=C.RF_BRANCH, stroke_width=1.5, corner_radius=0.06)
                           .set_stroke(opacity=0.7) for it in items])
        b_targets = SurroundingRectangle(VGroup(*y[12:]), buff=0.1, color=C.RF_TARGET, stroke_width=2, corner_radius=0.08)
        t_orders = label(r"a list of orders\\each with its own list of items", font_size=28, color=C.RF_BRANCH)
        t_orders.next_to(b_orders, RIGHT, buff=0.6)
        t_targets = label(r"what happens next month:\\the answers we want to predict", font_size=28, color=C.RF_TARGET)
        t_targets.next_to(b_targets, RIGHT, buff=0.6)
        with self.voiceover(
            "We'll follow one example through the series: a customer of an online store, written as YAML. "
            "<bookmark mark='t'/> They've been a customer for "
            "420 days. <bookmark mark='o'/> They've placed orders, and every order has its own list of line items: "
            "a product SKU, and how many were bought. <bookmark mark='a'/> The last two fields are what we'd like to "
            "predict: did this customer leave the next month, and how much did they spend?"
        ) as vo:
            self.play(FadeIn(head), LaggedStart(*[FadeIn(r, shift=RIGHT * 0.1) for r in y], lag_ratio=0.06), run_time=2.4)
            vo.wait_until("t")
            self.play(Indicate(y[0], color=C.RF_FIELD, scale_factor=1.05))
            vo.wait_until("o")
            self.play(Create(b_orders), FadeIn(t_orders, shift=LEFT * 0.2))
            self.play(LaggedStart(*[Create(b) for b in b_items], lag_ratio=0.3), run_time=1.5)
            vo.wait_until("a")
            self.play(Create(b_targets), FadeIn(t_targets, shift=LEFT * 0.2))
        self.yaml = y
        self.play(FadeOut(VGroup(b_orders, b_items, b_targets, t_orders, t_targets, head)))

    # ------------------------------------------------------------------
    def flatten(self):
        y = self.yaml
        cols = [("tenure", "420"), ("number of orders", "2"), ("number of items", "3"), ("mean quantity", "2.67"),
                ("distinct SKUs", "3"), ("days since last order", "3")]
        rows = VGroup()
        for h, v in cols:
            hc = label(h, font_size=28, color=GREY_A)
            vc = label(v, font_size=30)
            rows.add(VGroup(hc, vc))
        for r in rows:
            r[0].move_to(ORIGIN, aligned_edge=LEFT)
            r[1].move_to(RIGHT * 4.3, aligned_edge=RIGHT)
        rows.arrange(DOWN, buff=0.32, aligned_edge=LEFT)
        frame = SurroundingRectangle(rows, buff=0.3, color=GREY_B, stroke_width=1.5, corner_radius=0.06)
        seps = VGroup(*[Line(frame.get_left(), frame.get_right(), color=GREY_D, stroke_width=1)
                        .set_y((a.get_bottom()[1] + b.get_top()[1]) / 2) for a, b in zip(rows[:-1], rows[1:])])
        table = VGroup(frame, seps, rows).move_to(RIGHT * 3.2 + DOWN * 0.35)
        cap = label(r"the usual move: one row per customer,\\one hand-made feature per column", font_size=28)
        cap.next_to(table, UP, buff=0.35)
        q = label(r"which product was bought\\five at a time?", font_size=32, color=C.RF_TARGET)
        q.next_to(table, UP, buff=0.35)
        with self.voiceover(
            "The usual move is to flatten it: one row per customer, one column per hand-made feature. <bookmark mark='c'/> "
            "Count the orders. Count the items. Average the quantities. Count the distinct products. <bookmark mark='l'/> "
            "But every column throws something away. Once the quantities are averaged, you can no longer ask which "
            "product was bought five at a time, or whether it was in the newest order. Which value belongs to which "
            "item, and which item to which order: that structure is gone."
        ) as vo:
            self.play(y.animate.scale(0.85).to_edge(LEFT, buff=0.8), run_time=1.0)
            self.play(Create(frame), FadeIn(cap), run_time=0.8)
            vo.wait_until("c")
            for row, src in zip(rows, [y[0], y[1], VGroup(y[4], y[6], y[10]), VGroup(y[5], y[7], y[11]),
                                       VGroup(y[4], y[6], y[10]), y[2]]):
                ghost = src.copy().set_opacity(0.6)
                self.play(ReplacementTransform(ghost, row), run_time=0.55)
            self.play(Create(seps), run_time=0.4)
            vo.wait_until("l")
            self.play(FadeOut(cap), FadeIn(q), Indicate(y[11], color=C.RF_TARGET), run_time=1.0)
            cross = Cross(rows[3], stroke_color=RED, stroke_width=4).scale(0.9)
            self.play(Create(cross), rows[3].animate.set_opacity(0.45))
        self.play(FadeOut(VGroup(table, q, cross, y)))

    # ------------------------------------------------------------------
    def schema(self):
        c = code(CODE, font_size=19).to_edge(LEFT, buff=0.35)
        tree = Tree(CUSTOMER_TREE, font_size=19, x_gap=0.42).to_edge(RIGHT, buff=0.35)
        rows = {"/tenure_days": 1, "/orders": 2, "/orders/age_days": 4, "/orders/line_items": 5,
                "/orders/line_items/sku": 7, "/orders/line_items/quantity": 8, "/churned": 11,
                "/next_month_spend": 12}
        root_card = tree.cards["/"]
        with self.voiceover(
            "relflow takes the opposite road. You describe the record's shape once, in Python, <bookmark mark='s'/> "
            "and that description is the architecture. <bookmark mark='k'/> Keywords name the fields, and a type says "
            "what each value means. <bookmark mark='b'/> A Branch marks a repeated collection, with room for 64 orders, "
            "each with up to 16 line items. <bookmark mark='m'/> And mask equals True marks an answer: a value the "
            "model will learn to predict, but never gets to see."
        ) as vo:
            self.play(FadeIn(c, shift=RIGHT * 0.2), run_time=1.0)
            vo.wait_until("s")
            self.play(FadeIn(root_card), Create(SurroundingRectangle(c.code_lines[0], buff=0.05, color=C.RF_ROOT,
                                                                     stroke_width=1.5).set_stroke(opacity=0)), run_time=0.6)
            for addr in ("/tenure_days", "/orders", "/orders/age_days", "/orders/line_items", "/orders/line_items/sku",
                         "/orders/line_items/quantity", "/churned", "/next_month_spend"):
                line = c.code_lines[rows[addr]]
                card = tree.cards[addr]
                hl = SurroundingRectangle(line, buff=0.04, color=card.role_color, stroke_width=1.6, corner_radius=0.04)
                self.play(Create(tree.edge_to[addr]), TransformFromCopy(line, card), Create(hl), run_time=0.5)
                self.remove(hl)
                if addr == "/orders/line_items/quantity":
                    vo.wait_until("b")
                    self.play(Indicate(tree.cards["/orders"], color=C.RF_BRANCH),
                              Indicate(tree.cards["/orders/line_items"], color=C.RF_BRANCH), run_time=1.2)
            vo.wait_until("m")
            self.play(*[Indicate(tree.cards[a], color=C.RF_TARGET) for a in ("/churned", "/next_month_spend")],
                      run_time=1.2)
        self.code_block, self.tree = c, tree

    # ------------------------------------------------------------------
    def network(self):
        tree = self.tree
        self.play(FadeOut(self.code_block), tree.animate.scale(1.05).move_to(LEFT * 0.9 + DOWN * 0.1), run_time=1.0)
        chips = VGroup()
        leaf_chips, branch_chips, target_chips = [], [], []
        for addr, card in tree.cards.items():
            kind = card.kind
            if kind == "field":
                ch = chip(r"embedder", color=C.EMBED, font_size=20).next_to(card, RIGHT, buff=0.15)
                leaf_chips.append(ch)
            elif kind == "target":
                ch = chip(r"decoder", color=C.RF_TARGET, font_size=20).next_to(card, RIGHT, buff=0.15)
                target_chips.append(ch)
            else:
                ch = chip(r"encoder", color=card.role_color, font_size=20).next_to(card, DOWN, buff=0.08)
                branch_chips.append(ch)
            chips.add(ch)
        s = D("schema")["customer"]
        total = s["total_params"]
        count = label(rf"this model: {fmt_int(total)} parameters, before it has seen one label", font_size=26,
                      color=GREY_A).to_edge(DOWN, buff=0.45)
        tag = real_tag(r"counted from the real model, relflow 0.2.3", corner=UR)
        with self.voiceover(
            "From that tree, relflow builds the network. <bookmark mark='e'/> Every input field gets a small embedding "
            "module that turns its values into vectors. <bookmark mark='b'/> Every branch, and the root, gets its own "
            "transformer encoder, which reads a collection and passes a summary up to its parent. "
            "<bookmark mark='d'/> And every target gets a decoder, which reads the summaries back down the tree to "
            "make its prediction. <bookmark mark='n'/> With the medium preset, this one schema becomes a network of "
            "twelve million parameters."
        ) as vo:
            vo.wait_until("e")
            self.play(LaggedStart(*[FadeIn(c, shift=LEFT * 0.1) for c in leaf_chips], lag_ratio=0.15))
            vo.wait_until("b")
            self.play(LaggedStart(*[FadeIn(c, shift=UP * 0.1) for c in branch_chips], lag_ratio=0.2))
            vo.wait_until("d")
            self.play(LaggedStart(*[FadeIn(c, shift=LEFT * 0.1) for c in target_chips], lag_ratio=0.2))
            vo.wait_until("n")
            self.play(FadeIn(count), FadeIn(tag))
        self.wait(0.5)
        self.play(FadeOut(VGroup(tree, chips, count, tag)))

    # ------------------------------------------------------------------
    def roadmap(self):
        stages = [
            (r"Arrow\\records", C.ARROW_BUF), (r"dense\\tensors", C.VALUED), (r"embed\\each field", C.EMBED),
            (r"encode\\each branch", C.RF_BRANCH), (r"reduce\\and route up", C.MASS), (r"decode\\the targets", C.RF_TARGET),
        ]
        boxes = VGroup()
        for text, col in stages:
            r = RoundedRectangle(width=1.85, height=1.2, corner_radius=0.12, stroke_color=col, stroke_width=2.2,
                                 fill_color=col, fill_opacity=0.12)
            t = label(text, font_size=24).move_to(r)
            boxes.add(VGroup(r, t))
        boxes.arrange(RIGHT, buff=0.38).move_to(UP * 1.2)
        arrows = VGroup(*[Arrow(a.get_right(), b.get_left(), buff=0.06, stroke_width=3, color=GREY_B,
                                max_tip_length_to_length_ratio=0.4) for a, b in zip(boxes[:-1], boxes[1:])])
        first = label(r"next: the whole thing in three lines of code", font_size=28, color=WHITE)
        first.next_to(boxes, UP, buff=0.5)
        topics = VGroup(
            label(r"records $\rightarrow$ rectangles \;$\cdot$\; null is not zero", font_size=25, color=GREY_A),
            label(r"numbers, labels, identifiers, ranks, time, vectors, text", font_size=25, color=GREY_A),
            label(r"inside a branch \;$\cdot$\; padding costs nothing \;$\cdot$\; why softmax can't count", font_size=25,
                  color=GREY_A),
            label(r"decoders \;$\cdot$\; masks \;$\cdot$\; learning without labels", font_size=25, color=GREY_A),
            label(r"preprocessing \;$\cdot$\; a real run \;$\cdot$\; time and memory \;$\cdot$\; the proofs",
                  font_size=25, color=GREY_A),
        ).arrange(DOWN, buff=0.22).next_to(boxes, DOWN, buff=0.55)
        tag = real_tag()
        with self.voiceover(
            "Each episode stands on its own, and together they follow one batch of records through that whole "
            "machine. <bookmark mark='n'/> First, the short version: the whole thing in three lines of code. "
            "<bookmark mark='a'/> Then the pieces. How nested Arrow data becomes rectangles of tensors, and why a null "
            "is not a zero. <bookmark mark='b'/> How numbers, labels, identifiers, time and text each become vectors. "
            "<bookmark mark='c'/> What happens inside a branch, why padding costs nothing, and why softmax attention "
            "can't count. <bookmark mark='d'/> How a decoder asks the tree a question, how one mask mechanism makes "
            "targets, dropout and pretraining, <bookmark mark='e'/> and what happens before and after the model, what "
            "it costs in time and memory, and what relflow's own proofs show. <bookmark mark='f'/> Every grid, curve "
            "and number in the series comes from running relflow itself, on the computer that rendered it."
        ) as vo:
            self.play(LaggedStart(*[FadeIn(b, shift=UP * 0.15) for b in boxes], lag_ratio=0.12),
                      LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.12), run_time=2.0)
            vo.wait_until("n")
            self.play(FadeIn(first, shift=UP * 0.1))
            for mark, i, box_idx in (("a", 0, (0, 1)), ("b", 1, (2,)), ("c", 2, (3, 4)), ("d", 3, (5,)), ("e", 4, ())):
                vo.wait_until(mark)
                self.play(FadeIn(topics[i], shift=UP * 0.1),
                          *[Indicate(boxes[j], color=WHITE, scale_factor=1.06) for j in box_idx], run_time=0.9)
            vo.wait_until("f")
            self.play(FadeIn(tag))
        self.wait(0.6)
        self.play(FadeOut(VGroup(boxes, arrows, first, topics, tag)))
