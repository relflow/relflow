from __future__ import annotations

from explainer import *  # noqa: F403
from videos.relflow.common import (
    D, Mono, Plot, Tree, chip, fmt_int, label, mtex, note, real_tag, show_episode_card,
)
from videos.relflow.s01_hook import CUSTOMER_TREE

ADDRESSES = {
    "/": "/", "/tenure_days": "/tenure_days", "/orders": "/orders", "/orders/age_days": "/orders/age_days",
    "/orders/line_items": "/orders/line_items", "/orders/line_items/sku": "/orders/line_items/sku",
    "/orders/line_items/quantity": "/orders/line_items/quantity", "/churned": "/churned",
    "/next_month_spend": "/next_month_spend",
}


class SchemaIsArchitecture(VoiceoverScene):
    def construct(self):
        show_episode_card(self)
        self.addresses()
        self.shapes()
        self.modules()
        self.parameters()
        self.presets()

    # ------------------------------------------------------------------
    def addresses(self):
        tree = Tree(CUSTOMER_TREE, font_size=22, x_gap=0.55, y_gap=0.42).move_to(LEFT * 1.6 + DOWN * 0.1)
        addr = {}
        for a, card in tree.cards.items():
            t = Mono(ADDRESSES[a], font_size=15, color=GREY_A).next_to(card, DOWN, buff=0.06).align_to(card, LEFT)
            addr[a] = t
        root_note = label(r"the root: anonymous,\\always exactly\\one observation", font_size=24, color=C.RF_ROOT)
        root_note.next_to(tree.cards["/"], UP, buff=0.5).align_to(tree.cards["/"], LEFT)
        uses = VGroup(*[label(t, font_size=26, color=GREY_A) for t in
                        (r"addresses name:", r"$\cdot$ predictions", r"$\cdot$ metrics", r"$\cdot$ exported embeddings",
                         r"$\cdot$ edits to a trained model")]).arrange(DOWN, aligned_edge=LEFT, buff=0.14)
        uses.to_corner(UR, buff=0.5)
        sku = tree.cards["/orders/line_items/sku"]
        path_nodes = ["/", "/orders", "/orders/line_items", "/orders/line_items/sku"]
        with self.voiceover(
            "Here's the schema of a customer model: a customer with orders, each order with its line items, "
            "and two values to predict. Let's look closely at that tree. <bookmark mark='a'/> Every node has an address: a path that starts at "
            "the root, which is written as a single slash. <bookmark mark='r'/> The root is anonymous, and it always "
            "holds exactly one observation. Anything repeated lives in a branch beneath it. <bookmark mark='p'/> "
            "So the SKU of a line item lives at slash orders, slash line items, slash SKU. <bookmark mark='u'/> "
            "Addresses are how relflow names everything: predictions, metrics, exported embeddings, and even edits "
            "to a model that's already trained."
        ) as vo:
            self.play(FadeIn(tree), run_time=1.0)
            vo.wait_until("a")
            self.play(LaggedStart(*[FadeIn(t, shift=UP * 0.05) for t in addr.values()], lag_ratio=0.1), run_time=1.6)
            vo.wait_until("r")
            self.play(FadeIn(root_note), Indicate(tree.cards["/"], color=C.RF_ROOT))
            vo.wait_until("p")
            self.play(FadeOut(root_note))
            self.play(LaggedStart(*[Indicate(tree.cards[a], color=WHITE, scale_factor=1.08) for a in path_nodes],
                                  lag_ratio=0.35), Indicate(addr["/orders/line_items/sku"], color=YELLOW), run_time=2.0)
            vo.wait_until("u")
            self.play(FadeIn(uses, shift=LEFT * 0.2))
        self.tree, self.addr = tree, addr
        self.play(FadeOut(uses), FadeOut(VGroup(*addr.values())))

    # ------------------------------------------------------------------
    def shapes(self):
        tree = self.tree
        s = {n["address"]: n for n in D("schema")["customer"]["nodes"]}
        shape_tex = {
            "/tenure_days": r"B \times 1", "/orders/age_days": r"B \times 1 \times 64",
            "/orders/line_items/sku": r"B \times 1 \times 64 \times 16",
            "/orders/line_items/quantity": r"B \times 1 \times 64 \times 16",
            "/churned": r"B \times 1", "/next_month_spend": r"B \times 1",
        }
        assert s["/orders/line_items/sku"]["shape"] == [1, 64, 16] and s["/orders/age_days"]["shape"] == [1, 64]
        shapes = VGroup()
        for a, tx in shape_tex.items():
            m = mtex(tx, font_size=26, color=GREY_A).next_to(tree.cards[a], RIGHT, buff=0.2)
            shapes.add(m)
        lens = VGroup(
            chip(r"length = 64", color=C.RF_BRANCH, font_size=20).next_to(tree.cards["/orders"], DOWN, buff=0.08),
            chip(r"length = 16", color=C.RF_BRANCH, font_size=20).next_to(tree.cards["/orders/line_items"], DOWN, buff=0.08),
        )
        b = note(r"B = observations in the batch", font_size=24, color=GREY_A).to_corner(DR, buff=0.4)
        cut = VGroup(
            label(r"shorter lists are padded", font_size=26),
            label(r"longer lists are cut: keep the first (\texttt{head})\\or the last (\texttt{tail})", font_size=26),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.25).to_corner(UR, buff=0.5)
        with self.voiceover(
            "Each branch also fixes a shape. <bookmark mark='l'/> Orders holds up to 64 entries, and each order up to "
            "16 line items. <bookmark mark='s'/> So the tensor for SKU always has the shape: batch, by one root, by 64, "
            "by 16, however many items a real customer happens to have. <bookmark mark='c'/> Short lists get padded. "
            "Long lists get cut, keeping either the first entries, or the last ones, which is what you want for a "
            "history sorted oldest to newest. <bookmark mark='f'/> The shapes are fixed by the schema before a single "
            "record arrives."
        ) as vo:
            self.play(tree.animate.shift(LEFT * 0.6), run_time=0.6)
            for s_ in shapes:
                s_.shift(LEFT * 0.6)
            for l_ in lens:
                l_.shift(LEFT * 0.6)
            vo.wait_until("l")
            self.play(FadeIn(lens, shift=UP * 0.1))
            vo.wait_until("s")
            self.play(LaggedStart(*[FadeIn(m, shift=LEFT * 0.1) for m in shapes], lag_ratio=0.12), FadeIn(b))
            self.play(Indicate(shapes[2], color=YELLOW))
            vo.wait_until("c")
            self.play(FadeIn(cut[0]))
            self.play(FadeIn(cut[1]))
            vo.wait_until("f")
        self.play(FadeOut(VGroup(shapes, lens, b, cut, tree)))

    # ------------------------------------------------------------------
    def modules(self):
        # a leaf
        leaf = VGroup()
        lc = Tree(("quantity", "field", "Number", []), font_size=24)
        emb = chip(r"embedder: values $\rightarrow$ vectors of width $d_{\mathrm{model}}$", color=C.EMBED, font_size=28)
        dec = chip(r"decoder (targets, or \texttt{embed=True})", color=C.RF_TARGET, font_size=28)
        col = VGroup(emb, dec).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        leaf.add(lc, col.next_to(lc, RIGHT, buff=0.5))
        leaf.move_to(UP * 2.0)

        # a branch
        bc = Tree(("line_items", "branch", "Branch", []), font_size=24)
        parts = [
            (r"coordinate attention: the fields of one item talk", C.RF_FIELD),
            (r"$n_{\mathrm{layers}}$ transformer layers across the collection", C.RF_BRANCH),
            (r"reduction: summarize the collection for the parent", C.MASS),
        ]
        stack = VGroup()
        for text, colr in parts:
            r = RoundedRectangle(width=8.2, height=0.7, corner_radius=0.1, stroke_color=colr, stroke_width=2,
                                 fill_color=colr, fill_opacity=0.12)
            t = label(text, font_size=27).move_to(r)
            stack.add(VGroup(r, t))
        stack.arrange(UP, buff=0.12)
        arrows = VGroup(*[Arrow(a.get_top(), b.get_bottom(), buff=0.02, stroke_width=2.5, color=GREY_B,
                                max_tip_length_to_length_ratio=0.5, tip_length=0.12) for a, b in zip(stack[:-1], stack[1:])])
        enc = VGroup(stack, arrows)
        branch = VGroup(bc, enc.next_to(bc, RIGHT, buff=0.5)).move_to(DOWN * 1.2)
        with self.voiceover(
            "Now the modules. <bookmark mark='e'/> A leaf owns an embedder, which turns its values into vectors of one "
            "shared width, called d model. <bookmark mark='d'/> If the leaf is a target, or if you ask for its "
            "embedding, it also owns a decoder. <bookmark mark='b'/> A branch owns an encoder in three parts. "
            "<bookmark mark='c'/> First, a layer that lets the fields of the same item talk to each other. "
            "<bookmark mark='s'/> Then a stack of transformer layers that runs across the whole collection. "
            "<bookmark mark='r'/> And finally a reduction, which boils the collection down to a few vectors for its "
            "parent."
        ) as vo:
            self.play(FadeIn(lc))
            vo.wait_until("e")
            self.play(FadeIn(emb, shift=LEFT * 0.1))
            vo.wait_until("d")
            self.play(FadeIn(dec, shift=LEFT * 0.1))
            vo.wait_until("b")
            self.play(FadeIn(bc))
            for mark, i in (("c", 0), ("s", 1), ("r", 2)):
                vo.wait_until(mark)
                self.play(FadeIn(stack[i], shift=UP * 0.1), *([GrowArrow(arrows[i - 1])] if i else []))
        self.play(FadeOut(VGroup(leaf, branch)))

    # ------------------------------------------------------------------
    def parameters(self):
        s = D("schema")["customer"]
        nodes = {n["address"]: n for n in s["nodes"]}
        total = s["total_params"]
        rows = [
            ("root encoder", sum(nodes["/"]["params"].values()), C.RF_ROOT),
            ("line\\_items encoder", sum(nodes["/orders/line_items"]["params"].values()), C.RF_BRANCH),
            ("orders encoder", sum(nodes["/orders"]["params"].values()), C.RF_BRANCH),
            ("next\\_month\\_spend decoder", nodes["/next_month_spend"]["params"]["decoder"], C.RF_TARGET),
            ("churned decoder", nodes["/churned"]["params"]["decoder"], C.RF_TARGET),
            ("five embedders", sum(n["params"]["embedder"] for n in s["nodes"] if n["kind"] == "leaf"), C.EMBED),
        ]
        assert sum(r[1] for r in rows) == total
        sku = nodes["/orders/line_items/sku"]["params"]["embedder"]
        assert sku == 6 * 256  # five value-state rows plus a single, still empty, vocabulary row
        scale = 6.2 / rows[0][1]
        bars = VGroup()
        for name, n, colr in rows:
            nm = label(name, font_size=26, color=GREY_A)
            r = Rectangle(width=max(0.03, n * scale), height=0.42, stroke_width=0, fill_color=colr, fill_opacity=0.85)
            v = label(fmt_int(n), font_size=24)
            bars.add(VGroup(nm, r, v))
        for g in bars:
            g[0].move_to(ORIGIN, aligned_edge=RIGHT)
            g[1].next_to(g[0], RIGHT, buff=0.3, aligned_edge=ORIGIN).align_to(RIGHT * 0.3, LEFT)
            g[2].next_to(g[1], RIGHT, buff=0.2)
        bars.arrange(DOWN, buff=0.22, aligned_edge=LEFT)
        for g in bars:
            g[1].align_to(bars[0][1], LEFT)
            g[0].next_to(g[1], LEFT, buff=0.3)
            g[2].next_to(g[1], RIGHT, buff=0.2)
        bars.move_to(DOWN * 0.2 + RIGHT * 0.6)
        head = label(rf"where the customer model's {fmt_int(total)} parameters live", font_size=34).to_edge(UP, buff=0.6)
        sku_note = label(rf"the SKU embedder: {fmt_int(sku)} parameters, its vocabulary still empty", font_size=26,
                         color=C.CATEGORY).next_to(bars, DOWN, buff=0.5)
        tag = real_tag(r"counted from the real model (preset md, $d_{\mathrm{model}}=256$), relflow 0.2.3")
        with self.voiceover(
            "Here's where the twelve million parameters of our customer model actually live. <bookmark mark='b'/> "
            "Almost all of them are in the three encoders, the root's being the deepest, and in the two decoders. "
            "<bookmark mark='e'/> The embedders are tiny. <bookmark mark='s'/> And the SKU embedder is the smallest of "
            "all, because its vocabulary is still empty. It will grow as labels arrive, while it trains."
        ) as vo:
            self.play(FadeIn(head), FadeIn(tag))
            vo.wait_until("b")
            self.play(LaggedStart(*[AnimationGroup(FadeIn(g[0]), GrowFromEdge(g[1], LEFT), FadeIn(g[2]))
                                    for g in bars[:5]], lag_ratio=0.25), run_time=2.4)
            vo.wait_until("e")
            self.play(FadeIn(bars[5][0]), GrowFromEdge(bars[5][1], LEFT), FadeIn(bars[5][2]))
            vo.wait_until("s")
            self.play(FadeIn(sku_note, shift=UP * 0.1), Indicate(bars[5], color=C.CATEGORY))
        self.play(FadeOut(VGroup(head, bars, sku_note, tag)))

    # ------------------------------------------------------------------
    def presets(self):
        ps = D("schema")["presets"]
        head = [r"preset", r"$d_{\mathrm{model}}$", r"root layers", r"branch layers", r"heads", r"root outputs",
                r"branch outputs"]
        cells = [[label(h, font_size=26, color=GREY_A) for h in head]]
        for p in ps:
            cells.append([Mono(f"rf.Model.{p['name']}", font_size=22, color=C.RF_ROOT if p["name"] == "md" else WHITE)] +
                         [label(str(p[k]), font_size=28) for k in ("d_model", "root_layers", "branch_layers", "heads",
                                                                    "root_outputs", "branch_outputs")])
        table = MobjectTable(cells, include_outer_lines=False, h_buff=0.45, v_buff=0.28,
                             line_config={"stroke_color": GREY_D, "stroke_width": 1})
        table.scale_to_fit_width(12.4).move_to(UP * 0.2)
        md_row = table.get_rows()[3]
        hl = SurroundingRectangle(md_row, buff=0.06, color=C.RF_ROOT, stroke_width=2, corner_radius=0.06)
        override = Mono("rf.Model.md(n_layers=4, ...)  # only the root gets 4 layers", font_size=22, color=GREY_A)
        override.next_to(table, DOWN, buff=0.55)
        tag = real_tag(r"read from \texttt{rf.presets}, relflow 0.2.3")
        with self.voiceover(
            "Those sizes come from a preset. <bookmark mark='t'/> relflow ships five, from extra small to extra large. "
            "Each one sets the width, the depth of the root and of every branch, the number of attention heads, and "
            "how many summary vectors each branch passes to its parent. <bookmark mark='m'/> Our customer model used "
            "medium. <bookmark mark='o'/> Anything you set yourself wins, but only at the node where you set it."
        ) as vo:
            vo.wait_until("t")
            self.play(FadeIn(table, lag_ratio=0.02), FadeIn(tag), run_time=1.5)
            vo.wait_until("m")
            self.play(Create(hl))
            vo.wait_until("o")
            self.play(FadeIn(override, shift=UP * 0.1))
        self.wait(0.4)
        self.play(FadeOut(VGroup(table, hl, override, tag)))
