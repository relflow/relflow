from __future__ import annotations

from explainer import *  # noqa: F403
from videos.relflow.common import (
    D, Mono, label, mtex, note, real_tag, show_episode_card, state_tile,
)

ORDER_HUES = ["#8ECAE6", "#F4A261", "#CDB4DB", "#90BE6D"]


def item_chip(sku, qty, hue, font_size=20) -> VGroup:
    q = "null" if qty is None else str(qty)
    t = Mono(f"{sku}·{q}", font_size=font_size, color=WHITE)
    r = RoundedRectangle(width=t.width + 0.22, height=t.height + 0.16, corner_radius=0.07, stroke_color=hue,
                         stroke_width=1.6, fill_color=hue, fill_opacity=0.18)
    t.move_to(r)
    g = VGroup(r, t)
    g.box, g.text = r, t
    return g


def cell(text, color=C.ARROW_BUF, w=0.78, h=0.5, font_size=20, fill=0.12, text_color=WHITE) -> VGroup:
    r = Rectangle(width=w, height=h, stroke_color=color, stroke_width=1.4, fill_color=color, fill_opacity=fill)
    t = Mono(text, font_size=font_size, color=text_color).move_to(r)
    if t.width > w - 0.08:
        t.width = w - 0.08
    g = VGroup(r, t)
    g.box, g.text = r, t
    return g


class RaggedToDense(VoiceoverScene):
    def construct(self):
        show_episode_card(self)
        e = D("encode")
        self.e = e
        self.records()
        self.arrow_layout()
        self.placement()
        self.overflow()
        self.nested()
        self.states()

    # ------------------------------------------------------------------
    def records(self):
        rows = VGroup()
        self.chips = []
        for i, r in enumerate(self.e["records"]):
            name = Mono(f"order {i}", font_size=22, color=ORDER_HUES[i])
            items = VGroup(*[item_chip(it["sku"], it["quantity"], ORDER_HUES[i]) for it in r["line_items"]])
            if len(items):
                items.arrange(RIGHT, buff=0.12)
            else:
                items = VGroup(Mono("[ ]  (no items)", font_size=20, color=GREY_B))
            row = VGroup(name, items).arrange(RIGHT, buff=0.35)
            rows.add(row)
            self.chips.append(items)
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.22).to_corner(UL, buff=0.6)
        hint = note(r"each chip: SKU $\cdot$ quantity", font_size=22).next_to(rows, DOWN, buff=0.25).align_to(rows, LEFT)
        with self.voiceover(
            "How does a batch of records become a tensor? Let's follow a real batch. Here are four orders: two items, one item with a missing quantity, no items at "
            "all, and five items."
        ):
            self.play(LaggedStart(*[FadeIn(r, shift=RIGHT * 0.15) for r in rows], lag_ratio=0.25), FadeIn(hint),
                      run_time=2.2)
        self.rows, self.hint = rows, hint

    # ------------------------------------------------------------------
    def arrow_layout(self):
        e = self.e
        offs = VGroup(*[cell(str(o), color=GOLD, w=0.62) for o in e["offsets"]]).arrange(RIGHT, buff=0)
        flat = []
        for i, r in enumerate(e["records"]):
            for it in r["line_items"]:
                flat.append((it, i))
        skus = VGroup(*[cell(it["sku"], color=ORDER_HUES[i]) for it, i in flat]).arrange(RIGHT, buff=0)
        qtys = VGroup(*[cell("null" if it["quantity"] is None else str(it["quantity"]), color=ORDER_HUES[i],
                             text_color=C.NULL_STATE if it["quantity"] is None else WHITE) for it, i in flat])
        qtys.arrange(RIGHT, buff=0)
        lab_o = label(r"offsets", font_size=26, color=GOLD)
        lab_s = label(r"sku", font_size=26, color=C.ARROW_BUF)
        lab_q = label(r"quantity", font_size=26, color=C.ARROW_BUF)
        grid = VGroup(VGroup(lab_o, offs), VGroup(lab_s, skus), VGroup(lab_q, qtys))
        for g in grid:
            g[1].next_to(g[0], RIGHT, buff=0.3)
        grid.arrange(DOWN, buff=0.35, aligned_edge=LEFT)
        for g in grid:
            g[1].align_to(grid[0][1], LEFT)
        for g in grid[1:]:
            g[0].next_to(g[1], LEFT, buff=0.3)
        grid[0][0].next_to(grid[0][1], LEFT, buff=0.3)
        grid.next_to(self.hint, DOWN, buff=0.55).align_to(self.rows, LEFT).shift(RIGHT * 0.3)
        grid[0].shift(UP * 0.35)
        grid[0][1].match_x(skus)
        grid[0][0].next_to(grid[0][1], LEFT, buff=0.3)
        idx = VGroup(*[Mono(str(k), font_size=16, color=GREY_B).next_to(c, UP, buff=0.05) for k, c in enumerate(skus)])
        spans = VGroup()
        for i in range(4):
            a, b = e["offsets"][i], e["offsets"][i + 1]
            if b > a:
                br = Brace(VGroup(*skus[a:b]), DOWN, buff=0.05, color=ORDER_HUES[i])
                br.next_to(qtys, DOWN, buff=0.08).match_x(VGroup(*qtys[a:b]))
                br.stretch_to_fit_width(VGroup(*qtys[a:b]).width)
                spans.add(VGroup(br, Mono(f"order {i}", font_size=18, color=ORDER_HUES[i]).next_to(br, DOWN, buff=0.06)))
        ptr = VGroup()
        for i, o in enumerate(e["offsets"]):
            x = skus[o].get_left()[0] if o < len(skus) else skus[-1].get_right()[0]
            ptr.add(Arrow(offs[i].get_bottom(), [x, skus.get_top()[1] + 0.25, 0], buff=0.06, stroke_width=2, color=GOLD,
                          max_tip_length_to_length_ratio=0.2, tip_length=0.12))
        with self.voiceover(
            "In memory, Apache Arrow doesn't store lists the way Python does. <bookmark mark='c'/> Every item of every "
            "order goes into one long column per field: all the SKUs, back to back, and all the quantities. "
            "<bookmark mark='o'/> A separate offsets array says where each order's items start and stop. Order zero "
            "owns items zero and one, order one owns item two, order two starts and stops at three, so it's empty, "
            "and order three owns the last five."
        ) as vo:
            vo.wait_until("c")
            src = VGroup(*[c for ch in self.chips for c in ch if hasattr(c, "box")])
            self.play(FadeIn(lab_s), FadeIn(lab_q),
                      *[TransformFromCopy(s_, VGroup(skus[k], qtys[k])) for k, s_ in enumerate(src)], run_time=1.8)
            self.play(FadeIn(idx))
            vo.wait_until("o")
            self.play(FadeIn(lab_o), FadeIn(offs, lag_ratio=0.1))
            self.play(LaggedStart(*[GrowArrow(p) for p in ptr], lag_ratio=0.15), run_time=1.2)
            self.play(LaggedStart(*[FadeIn(s_) for s_ in spans], lag_ratio=0.3), run_time=1.5)
        self.buffers = VGroup(grid, idx, spans, ptr)
        self.skus, self.qtys, self.flat = skus, qtys, flat

    # ------------------------------------------------------------------
    def dense_grid(self, retained):
        cells = VGroup()
        for r in range(4):
            for s in range(4):
                cells.add(Rectangle(width=0.95, height=0.5, stroke_color=GREY_C, stroke_width=1.2, fill_color=C.PADDED,
                                    fill_opacity=0.25))
        cells.arrange_in_grid(4, 4, buff=0.06)
        rl = VGroup(*[Mono(f"order {r}", font_size=18, color=ORDER_HUES[r]).next_to(cells[4 * r], LEFT, buff=0.2)
                      for r in range(4)])
        cl = VGroup(*[Mono(f"slot {s}", font_size=16, color=GREY_B).next_to(cells[s], UP, buff=0.12) for s in range(4)])
        g = VGroup(cells, rl, cl)
        g.cells = cells
        return g

    def placement(self):
        e = self.e
        head = e["head"]
        assert head["placement"] == [0, 1, 4, 12, 13, 14, 15]
        grid = self.dense_grid(head["retained_skus"]).to_edge(RIGHT, buff=0.6).shift(UP * 1.4)
        title = label(r"the model wants a rectangle: 4 slots per order", font_size=26, color=GREY_A)
        title.next_to(grid, UP, buff=0.55)
        formula = mtex(r"\text{placement} = \text{parent} \times \text{length} + \text{slot}", font_size=34)
        formula.next_to(grid, DOWN, buff=0.45)
        nums = VGroup()
        movers = []
        parents = [0, 0, 1, 3, 3, 3, 3]
        slots = [0, 1, 0, 0, 1, 2, 3]
        for k, (p, sl) in enumerate(zip(head["placement"], slots)):
            tgt = grid.cells[p]
            nums.add(Mono(str(p), font_size=15, color=GOLD).move_to(tgt.get_corner(UR) + DL * 0.12))
        with self.voiceover(
            "The model, though, wants rectangles: in this small example, a grid of four slots per order. "
            "<bookmark mark='p'/> So for every item it keeps, relflow computes a single number, its placement in the "
            "grid: the parent's position, times the branch length, plus the item's slot within its own list. "
            "<bookmark mark='m'/> Item four, for instance, is order three's second item: three times four, plus one, "
            "is thirteen. <bookmark mark='a'/> Then every value is scattered to its placement in one vectorized step."
        ) as vo:
            self.play(FadeIn(grid), FadeIn(title))
            vo.wait_until("p")
            self.play(Write(formula))
            vo.wait_until("m")
            hl = SurroundingRectangle(self.skus[4], buff=0.03, color=YELLOW, stroke_width=3)
            ex = mtex(r"3 \times 4 + 1 = 13", font_size=34, color=YELLOW).next_to(formula, DOWN, buff=0.25)
            self.play(Create(hl), Write(ex))
            vo.wait_until("a")
            anims = []
            for k, p in enumerate(head["placement"]):
                tgt = grid.cells[p]
                c = self.skus[k].copy()
                movers.append(c)
                anims.append(c.animate.move_to(tgt).set_opacity(1))
            self.play(LaggedStart(*anims, lag_ratio=0.12), run_time=2.0)
            self.play(FadeIn(nums), *[grid.cells[p].animate.set_fill(ORDER_HUES[parents[k]], 0.25)
                                      for k, p in enumerate(head["placement"])])
            self.play(FadeOut(hl), FadeOut(ex))
        self.grid, self.movers, self.formula, self.gtitle, self.nums = grid, movers, formula, title, nums

    # ------------------------------------------------------------------
    def overflow(self):
        e = self.e
        f11 = self.skus[7]
        drop = VGroup(Cross(f11, stroke_color=RED, stroke_width=3).scale(0.7))
        head_t = VGroup(Mono('overflow="head"', font_size=22, color=C.RF_BRANCH),
                        label(r"(default): keep the first 4", font_size=24, color=GREY_A)).arrange(RIGHT, buff=0.15)
        tail_t = VGroup(Mono('overflow="tail"', font_size=22, color=C.RF_BRANCH),
                        label(r": keep the last 4", font_size=24, color=GREY_A)).arrange(RIGHT, buff=0.1)
        head_t.next_to(self.formula, DOWN, buff=0.35)
        tail_t.move_to(head_t)
        row3 = [self.grid.cells[12 + s] for s in range(4)]
        tail_skus = e["tail"]["retained_skus"][-4:]
        assert tail_skus == ["A12", "D01", "E90", "F11"]
        with self.voiceover(
            "Order two is empty, so its whole row stays padding. <bookmark mark='o'/> Order three has five items but only "
            "four slots. By default relflow keeps the head, the first four, so F11 is dropped. <bookmark mark='t'/> With "
            "overflow equals tail it keeps the last four instead, which is what you want for a history that's sorted "
            "oldest to newest."
        ) as vo:
            self.play(Indicate(VGroup(*[self.grid.cells[8 + s] for s in range(4)]), color=GREY_A))
            vo.wait_until("o")
            self.play(FadeIn(head_t), Create(drop))
            vo.wait_until("t")
            new = VGroup(*[cell(s_, color=ORDER_HUES[3]).move_to(row3[k]) for k, s_ in enumerate(tail_skus)])
            old = VGroup(*self.movers[3:7])
            self.play(FadeOut(drop), ReplacementTransform(head_t, tail_t),
                      old.animate.shift(LEFT * 1.0).set_opacity(0), FadeIn(new, shift=LEFT * 1.0), run_time=1.4)
            self.wait(0.8)
            back = VGroup(*[cell(s_, color=ORDER_HUES[3]).move_to(row3[k]) for k, s_ in
                            enumerate(e["head"]["retained_skus"][-4:])])
            self.play(FadeOut(new), FadeIn(back), FadeOut(tail_t))
        self.remove(*old)
        self.movers = self.movers[:3] + list(back)

    # ------------------------------------------------------------------
    def nested(self):
        f = mtex(r"\text{line item placement} = (\underbrace{\text{customer} \times 64 + \text{order}}_{\text{the order's own placement}})"
                 r" \times 16 + \text{item}", font_size=32)
        f.to_edge(DOWN, buff=0.55)
        vec = label(r"all of it NumPy over Arrow buffers: no Python loop walks the records", font_size=24, color=GREY_A)
        vec.next_to(f, UP, buff=0.3)
        with self.voiceover(
            "Nested branches simply repeat the rule. <bookmark mark='n'/> In the customer model, an order's own "
            "placement becomes the parent position for its line items, so every item lands somewhere in a grid of 64 "
            "orders by 16 items. <bookmark mark='v'/> And all of it is NumPy over Arrow's buffers: no Python loop ever "
            "walks the records one at a time."
        ) as vo:
            self.play(FadeOut(self.buffers), FadeOut(self.rows), FadeOut(self.hint))
            vo.wait_until("n")
            self.play(Write(f), run_time=1.6)
            vo.wait_until("v")
            self.play(FadeIn(vec))
        self.play(FadeOut(f), FadeOut(vec))

    # ------------------------------------------------------------------
    def states(self):
        e = self.e["head"]["fields"]
        grid = self.grid
        q_state = e["/line_items/quantity"]["state"]
        sku_ids = e["/line_items/sku"]["content"]
        sku_state = e["/line_items/sku"]["state"]
        assert q_state[1][0] == 1 and q_state[2] == [2, 2, 2, 2] and q_state[3] == [0, 0, 0, 0]
        tiles = VGroup()
        for r in range(4):
            for s in range(4):
                t = state_tile(q_state[r][s], size=0.44).move_to(grid.cells[4 * r + s])
                tiles.add(t)
        head = label(r"state of \texttt{quantity}, as relflow encoded it", font_size=26).move_to(self.gtitle)
        legend = VGroup(*[VGroup(state_tile(k, 0.34), label(t, font_size=24, color=GREY_A)).arrange(RIGHT, buff=0.15)
                          for k, t in ((0, r"valued"), (1, r"null"), (2, r"padded"))]).arrange(RIGHT, buff=0.5)
        legend.next_to(self.formula, DOWN, buff=0.2).match_x(grid)
        ids = VGroup()
        for r in range(4):
            for s in range(4):
                if sku_state[r][s] == 0:
                    ids.add(Mono(str(sku_ids[r][s]), font_size=22, color=C.CATEGORY).move_to(grid.cells[4 * r + s]))
        dictionary = label(r"SKUs became small integers from a batch-local dictionary:\\"
                           r"\texttt{A12}$\to$0, \texttt{B07}$\to$1, \texttt{C33}$\to$2, \texttt{D01}$\to$3, \texttt{E90}$\to$4",
                           font_size=26, color=C.CATEGORY).to_corner(UL, buff=0.6)
        tag = real_tag(r"real \texttt{model.encode(...)} output, relflow 0.2.3")
        with self.voiceover(
            "Alongside the values, every slot also gets a state. <bookmark mark='g'/> Here's the real state grid relflow "
            "produced for the quantities of these four orders. <bookmark mark='v'/> V for a real value, "
            "<bookmark mark='n'/> N for the quantity that was explicitly null, <bookmark mark='p'/> and P for padding, "
            "where there was no item at all. <bookmark mark='s'/> And the SKUs came out as small integers, looked up "
            "in a dictionary that's local to this batch. We'll see later how those meet the model's own vocabulary."
        ) as vo:
            self.play(FadeOut(VGroup(*self.movers)), FadeOut(self.nums), ReplacementTransform(self.gtitle, head),
                      FadeIn(tag))
            vo.wait_until("g")
            self.play(LaggedStart(*[FadeIn(t, scale=0.6) for t in tiles], lag_ratio=0.04), run_time=1.4)
            for mark, k in (("v", 0), ("n", 1), ("p", 2)):
                vo.wait_until(mark)
                self.play(FadeIn(legend[k]), *[Indicate(t, color=WHITE, scale_factor=1.15) for t, st in
                                               zip(tiles, [x for row in q_state for x in row]) if st == k], run_time=0.8)
            vo.wait_until("s")
            self.play(FadeOut(tiles), FadeIn(ids), FadeIn(dictionary))
        self.wait(0.5)
        self.play(*[FadeOut(m) for m in self.mobjects])
