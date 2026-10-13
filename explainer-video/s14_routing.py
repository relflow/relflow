from __future__ import annotations

from explainer import *  # noqa: F403
from videos.relflow.common import (
    D, Mono, label, mtex, note, real_tag, show_episode_card, tile,
)


def tiles(n, color, size=0.2, gap=0.04, fill=0.85, direction=RIGHT, dashed=False) -> VGroup:
    g = VGroup()
    for _ in range(n):
        t = tile(color, size, fill=0.0 if dashed else fill, stroke=color if dashed else None)
        if dashed:
            t.sq.set_stroke(color, 1.2, opacity=0.8)
        g.add(t)
    return g.arrange(direction, buff=gap)


class Routing(VoiceoverScene):
    def construct(self):
        show_episode_card(self)
        nodes = {n["address"]: n for n in D("schema")["customer"]["nodes"]}
        li, od, rt = nodes["/orders/line_items"], nodes["/orders"], nodes["/"]
        assert li["input_width"] == 32 and li["outputs"] == 4
        assert od["input_width"] == 320 and od["outputs"] == 4
        assert rt["input_width"] == 7 and rt["outputs"] == 4
        assert D("schema")["customer"]["depthwise"] == [["/"], ["/orders"], ["/orders/line_items"]]
        self.level_items()
        self.level_orders()
        self.level_root()
        self.plan()

    # ------------------------------------------------------------------
    def level_items(self):
        sku = tiles(16, C.CATEGORY)
        qty = tiles(16, C.NUMBER)
        grid = VGroup(sku, qty).arrange(DOWN, buff=0.06)
        names = VGroup(Mono("sku", font_size=18, color=C.CATEGORY).next_to(sku, LEFT, buff=0.2),
                       Mono("quantity", font_size=18, color=C.NUMBER).next_to(qty, LEFT, buff=0.2))
        block = VGroup(grid, names).move_to(DOWN * 2.6 + LEFT * 2.0)
        frame = SurroundingRectangle(grid, buff=0.12, color=C.RF_BRANCH, stroke_width=1.8, corner_radius=0.08)
        lab = label(r"\texttt{line\_items}, for one order: 32 tokens", font_size=24, color=C.RF_BRANCH)
        lab.next_to(frame, UP, buff=0.12).align_to(frame, LEFT)
        out = tiles(4, C.MASS, size=0.26).next_to(frame, RIGHT, buff=1.4)
        arr = Arrow(frame.get_right(), out.get_left(), buff=0.12, color=GREY_B, stroke_width=3)
        al = label(r"encode, reduce", font_size=20, color=GREY_A).next_to(arr, UP, buff=0.05)
        ol = label(r"4 summaries", font_size=22, color=C.MASS).next_to(out, DOWN, buff=0.15)
        shape = mtex(r"B \times 64 \times 32 \times 256 \;\to\; B \times 64 \times 4 \times 256", font_size=28, color=GREY_A)
        shape.to_corner(DR, buff=0.4).shift(UP * 0.2)
        with self.voiceover(
            "Let's run a whole model, the customer model with its orders and line items, and follow the "
            "shapes from the leaves up. <bookmark mark='l'/> "
            "Every field has been embedded into vectors of width 256. <bookmark mark='d'/> The deepest branch runs "
            "first. For each of a customer's 64 orders, the line items encoder sees 16 SKU slots and 16 quantity "
            "slots: 32 tokens. <bookmark mark='r'/> It encodes them, and reduces them to four summary vectors, the "
            "medium preset's number of outputs. All 64 orders are processed at once, as one big batch."
        ) as vo:
            self.play(FadeIn(block, lag_ratio=0.02), run_time=1.2)
            vo.wait_until("l")
            self.play(Indicate(grid, color=WHITE, scale_factor=1.03))
            vo.wait_until("d")
            self.play(Create(frame), FadeIn(lab))
            vo.wait_until("r")
            self.play(GrowArrow(arr), FadeIn(al), FadeIn(out, lag_ratio=0.2), FadeIn(ol), FadeIn(shape))
        self.items = VGroup(block, frame, lab, arr, al, ol, shape)
        self.item_out = out

    # ------------------------------------------------------------------
    def level_orders(self):
        cols = VGroup()
        shown = 9
        for k in range(shown):
            age = tile(C.NUMBER, 0.2)
            summ = tiles(4, C.MASS, size=0.2, gap=0.035, direction=DOWN)
            cols.add(VGroup(age, summ).arrange(DOWN, buff=0.06))
        cols.arrange(RIGHT, buff=0.14)
        dots = Mono("· · ·", font_size=24, color=GREY_B)
        last = cols[-1].copy()
        row = VGroup(*cols, dots, last).arrange(RIGHT, buff=0.14).move_to(UP * 0.35 + LEFT * 2.2)
        frame = SurroundingRectangle(row, buff=0.12, color=C.RF_BRANCH, stroke_width=1.8, corner_radius=0.08)
        lab = label(r"\texttt{orders}: 64 orders $\times$ (age + 4 summaries) = 320 tokens", font_size=24, color=C.RF_BRANCH)
        lab.next_to(frame, UP, buff=0.12).align_to(frame, LEFT)
        names = VGroup(Mono("age_days", font_size=16, color=C.NUMBER).next_to(cols[0][0], LEFT, buff=0.2),
                       label(r"line items", font_size=18, color=C.MASS).next_to(cols[0][1], LEFT, buff=0.2))
        out = tiles(4, C.MASS, size=0.26).next_to(frame, RIGHT, buff=1.2)
        arr = Arrow(frame.get_right(), out.get_left(), buff=0.12, color=GREY_B, stroke_width=3)
        ol = label(r"4 summaries", font_size=22, color=C.MASS).next_to(out, DOWN, buff=0.15)
        shape = mtex(r"B \times 320 \times 256 \;\to\; B \times 4 \times 256", font_size=28, color=GREY_A)
        shape.next_to(self.items[-1], UP, buff=1.6).align_to(self.items[-1], RIGHT)
        route = Arrow(self.item_out.get_top(), cols[0][1].get_bottom(), buff=0.1, color=C.MASS, stroke_width=3,
                      max_tip_length_to_length_ratio=0.12)
        with self.voiceover(
            "Those four vectors are routed up into the orders branch, <bookmark mark='a'/> as if they were four more "
            "fields of each order. <bookmark mark='c'/> So every order contributes its age, plus four line-item "
            "summaries: five tokens, times 64 orders, is 320 tokens. <bookmark mark='e'/> The orders encoder runs "
            "across all of them, so an order can be compared with every other order, and reduces them to four vectors "
            "per customer."
        ) as vo:
            self.play(GrowArrow(route))
            vo.wait_until("a")
            self.play(FadeIn(row, lag_ratio=0.02), FadeIn(names), run_time=1.4)
            vo.wait_until("c")
            self.play(Create(frame), FadeIn(lab))
            vo.wait_until("e")
            self.play(GrowArrow(arr), FadeIn(out, lag_ratio=0.2), FadeIn(ol), FadeIn(shape))
        self.orders = VGroup(row, frame, lab, names, arr, ol, shape, route)
        self.order_out = out

    # ------------------------------------------------------------------
    def level_root(self):
        ten = tile(C.NUMBER, 0.26)
        summ = tiles(4, C.MASS, size=0.26)
        tg = tiles(2, C.RF_TARGET, size=0.26, dashed=True)
        row = VGroup(ten, summ, tg).arrange(RIGHT, buff=0.12).move_to(UP * 2.75 + LEFT * 2.2)
        frame = SurroundingRectangle(row, buff=0.12, color=C.RF_ROOT, stroke_width=1.8, corner_radius=0.08)
        lab = label(r"root: tenure + 4 order summaries + 2 masked targets = 7 slots", font_size=24, color=C.RF_ROOT)
        lab.next_to(frame, LEFT, buff=0.3)
        if lab.get_left()[0] < -7.0:
            lab.next_to(frame, UP, buff=0.1)
        tgl = note(r"absent: masked", font_size=18, color=C.RF_TARGET).next_to(tg, DOWN, buff=0.1)
        out = tiles(4, C.RF_ROOT, size=0.3).next_to(frame, RIGHT, buff=1.2)
        arr = Arrow(frame.get_right(), out.get_left(), buff=0.12, color=GREY_B, stroke_width=3)
        ol = label(r"the customer's 4 root vectors", font_size=22, color=C.RF_ROOT).next_to(out, RIGHT, buff=0.2)
        route = Arrow(self.order_out.get_top(), summ.get_bottom(), buff=0.1, color=C.MASS, stroke_width=3,
                      max_tip_length_to_length_ratio=0.12)
        with self.voiceover(
            "Finally, the root. <bookmark mark='s'/> Its sequence is the tenure, the four order summaries, and a slot "
            "for each of the two targets, which stay empty, because they're masked: seven slots in all. "
            "<bookmark mark='r'/> Three layers of attention, the deepest stack in the model, and a last reduction "
            "gives the customer's four root vectors."
        ) as vo:
            self.play(GrowArrow(route))
            vo.wait_until("s")
            self.play(FadeIn(row, lag_ratio=0.1), Create(frame), FadeIn(lab), FadeIn(tgl))
            vo.wait_until("r")
            self.play(GrowArrow(arr), FadeIn(out, lag_ratio=0.2), FadeIn(ol))
        self.root = VGroup(row, frame, lab, tgl, arr, out, ol, route)

    # ------------------------------------------------------------------
    def plan(self):
        shapes = VGroup(self.items[-1], self.orders[6])
        self.items.remove(shapes[0])
        self.orders.remove(shapes[1])
        everything = VGroup(self.items, self.item_out, self.orders, self.order_out, self.root)
        plan = VGroup(
            label(r"the execution plan:", font_size=28),
            Mono("depth 2: /orders/line_items", font_size=22, color=C.RF_BRANCH),
            Mono("depth 1: /orders", font_size=22, color=C.RF_BRANCH),
            Mono("depth 0: /", font_size=22, color=C.RF_ROOT),
            label(r"deepest first; each branch waits\\only for its own children", font_size=22, color=GREY_A),
            label(r"each hand-off is a reshape:\\the item axis folds into the order axis", font_size=22, color=GREY_A),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.16).to_corner(UR, buff=0.5).shift(DOWN * 1.2)
        tag = real_tag(r"widths and plan from \texttt{model.schema}, relflow 0.2.3", corner=DL)
        with self.voiceover(
            "That's relflow's execution plan: <bookmark mark='p'/> branches grouped by depth and run deepest first, "
            "each one waiting only for its own children. <bookmark mark='r'/> And every hand-off between levels is just "
            "a reshape: the item axis folds into the order axis, and the order axis into the customer. Nothing is "
            "copied out to Python along the way."
        ) as vo:
            self.play(FadeOut(shapes), run_time=0.5)
            self.play(everything.animate.scale(0.8).to_edge(LEFT, buff=0.3), FadeIn(tag), run_time=1.0)
            vo.wait_until("p")
            self.play(FadeIn(plan[:4], lag_ratio=0.2), FadeIn(plan[4]))
            vo.wait_until("r")
            self.play(FadeIn(plan[5]))
        self.wait(0.5)
        self.play(*[FadeOut(m) for m in self.mobjects])
