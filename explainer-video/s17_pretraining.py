from __future__ import annotations

import math

import numpy as np

from explainer import *  # noqa: F403
from videos.relflow.common import (
    D, Mono, Plot, code, heat_grid, label, note, proof_tag, real_tag, show_episode_card, tile,
)

PRETRAIN = '''
pretrain = rf.Model(
    d_model=64, n_layers=2, n_heads=4, batch_size=128,
    line_items=rf.Branch(
        length=8,
        sku=rf.Category(mask=rf.Mask(rate=0.15, reconstruct=True)),
        quantity=rf.Number,
        price=rf.Number(mask=rf.Mask(rate=0.15, reconstruct=True)),
    ),
)
'''

FINETUNE = '''
sku, price = rf.Address("line_items", "sku"), rf.Address("line_items", "price")
model = rf.Model.load("pretrained.rf")
model.update(rf.where("address").is_in([sku, price]), mask=False)
model.extend(rf.where("address") == rf.Address(), returned=rf.Boolean(mask=True))
model.optimizer = rf.adamw(learning_rate=1e-3)
'''

GROUP_COLORS = [C.RF_TARGET, C.NUMBER, C.CATEGORY, C.HASH, C.BOOLEAN, C.VALUED, C.MASS, C.RF_BRANCH]


class Pretraining(VoiceoverScene):
    def construct(self):
        show_episode_card(self)
        self.d = D("pretraining")
        self.idea()
        self.data()
        self.learned()
        self.finetune()
        self.results()
        self.caveats()

    # ------------------------------------------------------------------
    def idea(self):
        c = code(PRETRAIN, font_size=20).move_to(UP * 0.4)
        hl = [SurroundingRectangle(c.code_lines[i], buff=0.04, color=C.MASKED, stroke_width=2, corner_radius=0.04)
              for i in (4, 6)]
        none = label(r"no target at all: every objective comes from the records themselves", font_size=26,
                     color=GREY_A).next_to(c, DOWN, buff=0.5)
        with self.voiceover(
            "Labels are expensive. Records usually aren't: for every order someone labeled, there may be a hundred "
            "that nobody did. <bookmark mark='m'/> Masks let a model learn from those too. Hide a sample of the values "
            "that are already there, and ask the model to reconstruct them from everything else. "
            "<bookmark mark='c'/> Here, fifteen percent of SKUs and prices, picked fresh every epoch. "
            "<bookmark mark='n'/> There's no label in this schema at all."
        ) as vo:
            vo.wait_until("m")
            self.play(FadeIn(c, shift=UP * 0.1))
            vo.wait_until("c")
            self.play(*[Create(h) for h in hl])
            vo.wait_until("n")
            self.play(FadeIn(none))
        self.play(FadeOut(VGroup(c, *hl, none)))

    # ------------------------------------------------------------------
    def data(self):
        d = self.d
        groups = VGroup()
        for g in range(d["groups"]):
            dots = VGroup(*[tile(GROUP_COLORS[g], 0.16, fill=0.9) for _ in range(25)]).arrange_in_grid(5, 5, buff=0.04)
            lab = label("ABCDEFGH"[g], font_size=22, color=GROUP_COLORS[g]).next_to(dots, UP, buff=0.1)
            groups.add(VGroup(lab, dots))
        groups.arrange(RIGHT, buff=0.42).move_to(UP * 1.5)
        head = label(rf"{d['catalog']} products in {d['groups']} hidden groups", font_size=28).next_to(groups, UP, buff=0.35)
        facts = VGroup(
            label(r"an order mostly buys from one group", font_size=26, color=GREY_A),
            label(r"returned $\iff$ a group-A product is bought 3 or more at a time", font_size=26, color=C.RF_TARGET),
            label(r"group A costs the same as group B, so price alone can't tell", font_size=26, color=GREY_A),
            label(rf"{d['unlabeled_rows']:,} orders without labels; a few dozen to a thousand with them".replace(",", "{,}", 1),
                  font_size=26),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.22).next_to(groups, DOWN, buff=0.6)
        with self.voiceover(
            f"Here's a synthetic shop with {d['catalog']} products, secretly arranged in {d['groups']} groups. "
            "<bookmark mark='o'/> Each order buys mostly from one group. <bookmark mark='r'/> An order comes back if "
            "it holds a group-A product bought three or more at a time. <bookmark mark='p'/> And group A costs about "
            "the same as group B, so the price can't give it away. To predict returns, a model has to learn which "
            f"products are in group A. <bookmark mark='u'/> We have {d['unlabeled_rows']:,} orders without labels, "
            "and only a few with them."
        ) as vo:
            self.play(FadeIn(head), FadeIn(groups, lag_ratio=0.05), run_time=1.5)
            for mark, f in zip("orpu", facts):
                vo.wait_until(mark)
                self.play(FadeIn(f, shift=RIGHT * 0.1))
        self.play(FadeOut(VGroup(head, groups, facts)))

    # ------------------------------------------------------------------
    def learned(self):
        d = self.d
        e = d["embedding"]
        w = np.array(e["rows"])
        unit = w / np.linalg.norm(w, axis=1, keepdims=True)
        groups = np.array([s[0] for s in e["labels"]])
        letters = "ABCDEFGH"
        m = np.array([[np.mean(unit[groups == a] @ unit[groups == b].T) for b in letters] for a in letters])
        grid = heat_grid(np.clip(m, 0, None), C.EMBED, cell=0.55, vmin=0.0, vmax=m.max()).move_to(LEFT * 3.2 + DOWN * 0.4)
        rl = VGroup(*[label(letters[i], font_size=20, color=GROUP_COLORS[i]).next_to(grid[i], LEFT, buff=0.15)
                      for i in range(8)])
        cl = VGroup(*[label(letters[j], font_size=20, color=GROUP_COLORS[j]).next_to(grid[0][j], UP, buff=0.12)
                      for j in range(8)])
        gt = label(r"mean cosine similarity of the\\learned SKU vectors, by group", font_size=22, color=GREY_A)
        gt.next_to(VGroup(grid, cl), UP, buff=0.25)
        curve = d["pretrain_curve"][-1]
        acc = curve[".line_items.sku/validate.accuracy.content"]
        chance = 1 / d["catalog"]
        ceiling = 0.8 / 25 + 0.2 / d["catalog"]
        side = VGroup(
            label(r"reconstructing a hidden SKU:", font_size=26),
            label(rf"random guess: {chance * 100:.1f}\%", font_size=24, color=GREY_A),
            label(rf"the model, after pretraining: {acc * 100:.1f}\%", font_size=24, color=C.EMBED),
            label(rf"knowing the group but not the item: about {ceiling * 100:.1f}\%", font_size=24, color=GREY_A),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to(RIGHT * 3.0 + UP * 0.6)
        diag = float(np.mean(np.diag(m)))
        off = float(np.mean(m[~np.eye(8, dtype=bool)]))
        verdict = label(rf"same group: {diag:.2f} $\quad$ different groups: {off:+.2f}", font_size=24, color=YELLOW)
        verdict.next_to(side, DOWN, buff=0.6).align_to(side, LEFT)
        tag = real_tag(rf"real pretraining run: 6 epochs over {d['unlabeled_rows']:,} unlabeled orders".replace(",", "{,}"))
        assert diag > 5 * abs(off) and acc > 4 * chance
        with self.voiceover(
            "After six epochs of pretraining on those unlabeled orders, what has it learned? <bookmark mark='a'/> "
            f"Asked to name a hidden SKU, it gets {acc * 100:.1f} percent right. That sounds poor, but a random guess "
            f"gets {chance * 100:.1f}, and even a model that knew each product's group perfectly, but not which item, "
            f"would get about {ceiling * 100:.0f}. <bookmark mark='g'/> It has learned the groups. "
            "<bookmark mark='s'/> You can see it in the SKU embedding table: products from the same group point in "
            "similar directions, and products from different groups don't. Nobody told it the groups existed."
        ) as vo:
            self.play(FadeIn(tag))
            vo.wait_until("a")
            self.play(FadeIn(side, lag_ratio=0.25), run_time=1.6)
            vo.wait_until("g")
            self.play(Indicate(side[2], color=WHITE))
            vo.wait_until("s")
            self.play(FadeIn(grid, lag_ratio=0.01), FadeIn(rl), FadeIn(cl), FadeIn(gt))
            self.play(FadeIn(verdict))
        self.play(FadeOut(VGroup(grid, rl, cl, gt, side, verdict, tag)))

    # ------------------------------------------------------------------
    def finetune(self):
        c = code(FINETUNE, font_size=21).move_to(UP * 0.6)
        notes = VGroup(
            label(r"load the pretrained weights, vocabularies and normalizers", font_size=24, color=GREY_A),
            label(r"SKU and price become ordinary inputs again", font_size=24, color=GREY_A),
            label(r"add the label as a new target at the root", font_size=24, color=C.RF_TARGET),
            label(r"a fresh optimizer; then fit as usual", font_size=24, color=GREY_A),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18).next_to(c, DOWN, buff=0.5)
        with self.voiceover(
            "Fine-tuning takes a few lines. <bookmark mark='l'/> Load the pretrained model, <bookmark mark='u'/> turn "
            "the reconstructed fields back into ordinary inputs, <bookmark mark='e'/> add the label you care about as a "
            "new target, <bookmark mark='o'/> give it an optimizer, and fit, exactly as before. Weights whose names "
            "and shapes still match carry over; the new decoder starts fresh."
        ) as vo:
            self.play(FadeIn(c))
            for mark, n in zip("lueo", notes):
                vo.wait_until(mark)
                self.play(FadeIn(n, shift=RIGHT * 0.1))
        self.play(FadeOut(VGroup(c, notes)))

    # ------------------------------------------------------------------
    def results(self):
        d = self.d
        budgets = d["budgets"]
        res = {a: [np.array(d["results"][a][str(b)]) for b in budgets] for a in ("scratch", "pretrained")}
        pl = Plot((24, 1400), (0.7, 1.0), width=8.6, height=4.6, log_x=True, x_ticks=budgets,
                  y_ticks=[0.7, 0.8, 0.9, 1.0], x_fmt=lambda v: f"{int(v)}", font_size=22).move_to(LEFT * 1.6 + DOWN * 0.3)
        xt = pl.x_title(r"labeled orders used for fine-tuning", font_size=22)
        yt = label(r"test AUC, 2{,}000 new orders", font_size=22, color=GREY_A).next_to(pl, UP, buff=0.15).align_to(pl, LEFT)
        series = {}
        for arm, col in (("scratch", GREY_B), ("pretrained", C.EMBED)):
            means = [r.mean() for r in res[arm]]
            line = pl.line(budgets, means, col, 3.5)
            dots = VGroup(*[pl.dot(b, v, color=col, radius=0.035, opacity=0.7) for b, r in zip(budgets, res[arm]) for v in r])
            series[arm] = VGroup(line, dots)
        leg = VGroup(
            VGroup(Dot(color=GREY_B), label(r"from scratch", font_size=24, color=GREY_B)).arrange(RIGHT, buff=0.15),
            VGroup(Dot(color=C.EMBED), label(r"pretrained, then fine-tuned", font_size=24, color=C.EMBED))
            .arrange(RIGHT, buff=0.15),
            label(r"3 seeds each (dots), mean (line), 300 steps", font_size=20, color=GREY_A),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15).to_corner(UR, buff=0.4).shift(DOWN * 0.6)
        i64 = budgets.index(64)
        s64, p64 = res["scratch"][i64].mean(), res["pretrained"][i64].mean()
        last = len(budgets) - 1
        s_last, p_last = res["scratch"][last].mean(), res["pretrained"][last].mean()
        seen32 = d["skus_seen"]["32"]
        note1 = label(rf"64 labels: {s64:.2f} $\to$ {p64:.2f}", font_size=26, color=YELLOW).next_to(leg, DOWN, buff=0.5)
        note1.align_to(leg, LEFT)
        note2 = label(rf"32 labeled orders show only {seen32:.0f}\\of the {d['catalog']} products", font_size=22,
                      color=GREY_A).next_to(note1, DOWN, buff=0.35).align_to(leg, LEFT)
        note3 = label(rf"{budgets[-1]:,} labels: {s_last:.2f} vs {p_last:.2f}".replace(",", "{,}"), font_size=24,
                      color=GREY_A).next_to(note2, DOWN, buff=0.35).align_to(leg, LEFT)
        tag = real_tag(r"real fine-tuning runs, relflow 0.2.3")
        assert p64 - s64 > 0.05 and abs(p_last - s_last) < 0.02
        assert all(p.mean() >= s.mean() - 0.005 for p, s in zip(res["pretrained"], res["scratch"]))
        ahead = [b for b, p, s in zip(budgets, res["pretrained"], res["scratch"]) if p.mean() > s.mean()]
        ahead_until = max(b for b in ahead if all(x in ahead for x in budgets[: budgets.index(b) + 1]))
        with self.voiceover(
            "Now the real comparison. Same architecture, same test orders, same three hundred training steps; the "
            "only difference is where the weights start. <bookmark mark='s'/> From scratch, a model trained on a "
            "handful of labeled orders does all right, and improves as labels grow. <bookmark mark='p'/> The "
            f"pretrained model is ahead at every budget up to {ahead_until}. With 64 labeled orders, test AUC goes from {s64:.2f} to "
            f"{p64:.2f}. <bookmark mark='w'/> Part of the reason: with so few labels, many products never appear at "
            "all, so a model trained from scratch has never seen them. The pretrained one has, in thousands of orders. "
            f"<bookmark mark='c'/> And by {budgets[-1]:,} labels, the two meet. Pretraining was a head start, not a "
            "higher ceiling."
        ) as vo:
            self.play(Create(pl), FadeIn(xt), FadeIn(yt), FadeIn(tag))
            vo.wait_until("s")
            self.play(Create(series["scratch"][0]), FadeIn(series["scratch"][1]), FadeIn(leg[0]), FadeIn(leg[2]))
            vo.wait_until("p")
            self.play(Create(series["pretrained"][0]), FadeIn(series["pretrained"][1]), FadeIn(leg[1]))
            self.play(FadeIn(note1))
            vo.wait_until("w")
            self.play(FadeIn(note2))
            vo.wait_until("c")
            self.play(FadeIn(note3))
        self.play(FadeOut(VGroup(pl, xt, yt, series["scratch"], series["pretrained"], leg, note1, note2, note3, tag)))

    # ------------------------------------------------------------------
    def caveats(self):
        rows = VGroup(
            label(r"compare against the same task trained from scratch, on the same held-out data", font_size=25),
            label(r"keep pretraining data on the right side of your evaluation boundary", font_size=25),
            label(r"vocabularies and normalizers keep learning while you fine-tune", font_size=25),
            label(r"for large vocabularies, watch coverage: how many targets the model can even name (P062)",
                  font_size=25),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        tag = proof_tag(r"P062")
        with self.voiceover(
            "A few habits keep this honest. <bookmark mark='a'/> Always compare against the same task trained from "
            "scratch, as we did. <bookmark mark='b'/> Don't pretrain on the customers or the time period you'll test "
            "on. <bookmark mark='c'/> Remember that vocabularies and normalizers keep learning while you fine-tune. "
            "<bookmark mark='d'/> And with large vocabularies, measure coverage, how many targets the model can even "
            "name, alongside accuracy."
        ) as vo:
            for mark, r in zip("abcd", rows):
                vo.wait_until(mark)
                self.play(FadeIn(r, shift=RIGHT * 0.1))
            self.play(FadeIn(tag))
        self.wait(0.5)
        self.play(FadeOut(VGroup(rows, tag)))
