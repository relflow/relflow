from __future__ import annotations

from collections import Counter

from explainer import *  # noqa: F403
from videos.relflow.common import (
    D, Mono, label, logo, mtex, note, proof_tag, show_episode_card, wordmark,
)

STATUS_COLOR = {"Passing": "#3FBF8F", "Limited": "#FFD166", "Partial": "#FF8FA3"}


class Evidence(VoiceoverScene):
    def construct(self):
        show_episode_card(self)
        self.p = D("proofs")
        self.catalog()
        self.highlights()
        self.limits()
        self.recap()

    # ------------------------------------------------------------------
    def catalog(self):
        cat = self.p["catalog"]
        st = Counter(c["status"] for c in cat)
        runs = Counter(c["latest_full"] for c in cat)
        assert len(cat) == 78 and st == {"Passing": 55, "Limited": 13, "Partial": 10} and runs["not_met"] == 7
        fams = []
        for c in cat:
            if c["family"] not in fams:
                fams.append(c["family"])
        rows = VGroup()
        for f in fams:
            items = [c for c in cat if c["family"] == f]
            sq = VGroup(*[Square(0.3, stroke_width=1.5 if c["latest_full"] == "not_met" else 0, stroke_color=WHITE,
                                 fill_color=STATUS_COLOR[c["status"]], fill_opacity=0.9) for c in items])
            sq.arrange(RIGHT, buff=0.06)
            name = label(f.replace("_", " "), font_size=20, color=GREY_A)
            rows.add(VGroup(name, sq))
        for r in rows:
            r[0].move_to(ORIGIN, aligned_edge=RIGHT)
            r[1].next_to(r[0], RIGHT, buff=0.25)
        rows.arrange(DOWN, buff=0.07, aligned_edge=LEFT)
        for r in rows:
            r[1].align_to(rows[0][1], LEFT)
            r[0].next_to(r[1], LEFT, buff=0.25)
        rows.move_to(LEFT * 2.0 + DOWN * 0.15)
        if rows.height > 6.6:
            rows.scale_to_fit_height(6.6)
        leg = VGroup(*[VGroup(Square(0.28, stroke_width=0, fill_color=c, fill_opacity=0.9),
                              label(rf"{k}: {st[k]}", font_size=26)).arrange(RIGHT, buff=0.2)
                       for k, c in STATUS_COLOR.items()])
        leg.add(VGroup(Square(0.28, stroke_width=1.5, stroke_color=WHITE, fill_opacity=0),
                       label(rf"missed its own gates\\in the latest run: {runs['not_met']}", font_size=24)).arrange(RIGHT, buff=0.2))
        leg.arrange(DOWN, aligned_edge=LEFT, buff=0.3).to_edge(RIGHT, buff=0.6)
        tag = proof_tag(r"\texttt{proofs/results.yaml}, all 78 entries")
        with self.voiceover(
            "How do we know any of this works, beyond a few demos? <bookmark mark='c'/> relflow ships its own evidence: "
            "a catalog of 78 proofs, small controlled experiments that each test one learning claim, with controls "
            "designed to fail when the claim is false. They're empirical tests, not mathematical proofs, and the "
            "catalog says so. <bookmark mark='s'/> 55 are marked passing, <bookmark mark='l'/> 13 limited, "
            "<bookmark mark='p'/> and 10 partial. <bookmark mark='m'/> In the latest recorded runs, seven of them "
            "missed their own gates, and that's recorded too."
        ) as vo:
            vo.wait_until("c")
            self.play(LaggedStart(*[FadeIn(r, shift=RIGHT * 0.1) for r in rows], lag_ratio=0.06), FadeIn(tag), run_time=2.4)
            for mark, i in (("s", 0), ("l", 1), ("p", 2), ("m", 3)):
                vo.wait_until(mark)
                self.play(FadeIn(leg[i], shift=LEFT * 0.1))
        self.play(FadeOut(VGroup(rows, leg, tag)))

    # ------------------------------------------------------------------
    def highlights(self):
        m = self.p["metrics"]
        p40, p76, p44 = m["P040"], m["P076"], m["P044"]
        assert abs(p40["recovered_pair_delta_fraction"] - 0.906) < 0.01
        assert abs(p76["informative"]["anomaly_auc"] - 0.962) < 0.01 and p76["marginal_auc"] == 0.5
        cards = VGroup(
            VGroup(label(r"P040 \; counts survive nesting", font_size=31, color=STATUS_COLOR["Passing"]),
                   label(r"the same six amounts, grouped into different sessions:", font_size=27, color=GREY_A),
                   label(rf"{100 * p40['recovered_pair_delta_fraction']:.0f}\% of the true difference between paired records "
                         r"recovered", font_size=27)),
            VGroup(label(r"P076 \; spotting contradictions", font_size=31, color=STATUS_COLOR["Passing"]),
                   label(r"every field normal on its own, the combination wrong:", font_size=27, color=GREY_A),
                   label(rf"reconstruction error flags it with AUC {p76['informative']['anomaly_auc']:.2f}; "
                         rf"any single field: {p76['marginal_auc']:.1f}", font_size=27)),
            VGroup(label(r"P044 \; calendar phase", font_size=31, color=STATUS_COLOR["Passing"]),
                   label(r"the month from the day of the year:", font_size=27, color=GREY_A),
                   label(rf"accuracy {p44['periodic_accuracy']:.2f}, against {p44['permuted_accuracy']:.2f} with the "
                         r"days shuffled", font_size=27)),
        )
        for c in cards:
            c.arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        cards.arrange(DOWN, aligned_edge=LEFT, buff=0.55).move_to(ORIGIN)
        tag = proof_tag(r"P040, P076, P044")
        with self.voiceover(
            "A few highlights. <bookmark mark='a'/> Counts can survive nesting: six equal amounts, grouped into "
            "sessions in different ways, and the model recovers ninety-one percent of the true difference between "
            "the groupings. <bookmark mark='b'/> Reconstruction can spot contradictions: records where every field "
            "looks normal on its own, but the combination is wrong, get flagged with an AUC of 0.96, where any single "
            "field gives chance. <bookmark mark='c'/> And calendar features pick up phase: the month, from the day of "
            "the year, at 92 percent, against ten percent with the days shuffled."
        ) as vo:
            self.play(FadeIn(tag))
            for mark, c in zip("abc", cards):
                vo.wait_until(mark)
                self.play(FadeIn(c, shift=UP * 0.1))
        self.play(FadeOut(VGroup(cards, tag)))

    # ------------------------------------------------------------------
    def limits(self):
        cards = VGroup(
            VGroup(label(r"P028 \; identity across summarized collections", font_size=31, color=STATUS_COLOR["Limited"]),
                   label(r"equality is lost once each collection has been reduced: chance", font_size=27, color=GREY_A)),
            VGroup(label(r"P066 \; admitting a new label mid-training", font_size=31, color=STATUS_COLOR["Partial"]),
                   label(r"its fresh row is random, and can jolt predictions before it has learned anything;", font_size=27,
                         color=GREY_A),
                   label(r"the obvious fixes regressed other proofs, so they weren't adopted", font_size=27, color=GREY_A)),
            VGroup(label(r"P071 \; familiar instructions, unseen combinations", font_size=31, color=STATUS_COLOR["Partial"]),
                   label(r"one of three seeds failed a withheld group--operation pair:", font_size=27, color=GREY_A),
                   label(r"reliable transfer to every unseen pair remains unproven", font_size=27, color=GREY_A)),
        )
        for c in cards:
            c.arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        cards.arrange(DOWN, aligned_edge=LEFT, buff=0.55).move_to(ORIGIN)
        tag = proof_tag(r"P028, P066, P071")
        with self.voiceover(
            "And the limits are recorded just as carefully. <bookmark mark='a'/> Comparing identities across two "
            "collections that have each been summarized fails, as the hashing episode showed. <bookmark mark='b'/> Admitting a brand-new "
            "label in the middle of training activates a random row, which can jolt predictions before it has learned "
            "anything, and the obvious fixes made other proofs worse, so they weren't adopted. <bookmark mark='c'/> And "
            "combining two familiar instructions in a pairing never seen in training works on some seeds, but not "
            "reliably yet."
        ) as vo:
            self.play(FadeIn(tag))
            for mark, c in zip("abc", cards):
                vo.wait_until(mark)
                self.play(FadeIn(c, shift=UP * 0.1))
        self.play(FadeOut(VGroup(cards, tag)))

    # ------------------------------------------------------------------
    def recap(self):
        stages = [
            (r"Arrow records", r"one placement rule;\\a state beside every value", C.ARROW_BUF),
            (r"embedders", r"Fourier + ruler, growing\\vocabularies, salted hashes", C.EMBED),
            (r"branch encoders", r"fields talk, then items talk;\\padding packed away", C.RF_BRANCH),
            (r"reductions", r"summaries that can\\still count", C.MASS),
            (r"decoders", r"read back down the tree,\\fill in what masks hide", C.RF_TARGET),
        ]
        boxes = VGroup()
        for t, sub, col in stages:
            r = RoundedRectangle(width=2.45, height=2.1, corner_radius=0.14, stroke_color=col, stroke_width=2.2,
                                 fill_color=col, fill_opacity=0.1)
            tt = label(t, font_size=26, color=col).move_to(r.get_top() + DOWN * 0.4)
            ss = label(sub, font_size=19, color=GREY_A).move_to(r.get_center() + DOWN * 0.3)
            boxes.add(VGroup(r, tt, ss))
        boxes.arrange(RIGHT, buff=0.22).move_to(UP * 0.3)
        arrows = VGroup(*[Arrow(a.get_right(), b.get_left(), buff=0.03, stroke_width=2.5, color=GREY_B,
                                max_tip_length_to_length_ratio=0.5, tip_length=0.12) for a, b in zip(boxes[:-1], boxes[1:])])
        with self.voiceover(
            "So, the journey of one batch. <bookmark mark='a'/> Nested Arrow records are laid into fixed rectangles by "
            "a single placement rule, with a state beside every value. <bookmark mark='b'/> Each field becomes vectors "
            "in its own way: Fourier features and a ruler for numbers, a growing table for labels, a salted hash for "
            "identities. <bookmark mark='c'/> Each branch lets an item's fields talk, then lets its items talk, with "
            "the padding packed away, <bookmark mark='d'/> and reports to its parent with summaries that can still "
            "count. <bookmark mark='e'/> And decoders reach back down the tree, to fill in whatever the masks have "
            "hidden."
        ) as vo:
            for mark, i in zip("abcde", range(5)):
                vo.wait_until(mark)
                anims = [FadeIn(boxes[i], shift=UP * 0.15)]
                if i:
                    anims.append(GrowArrow(arrows[i - 1]))
                self.play(*anims)
        mark_ = logo(1.4)
        word = wordmark(80)
        brand = VGroup(mark_, word).arrange(RIGHT, buff=0.35).move_to(DOWN * 2.4)
        line = label(r"the schema you write is the network you get", font_size=32, color=GREY_A)
        with self.voiceover(
            "The schema you write is the network you get. Thanks for watching."
        ):
            self.play(boxes.animate.shift(UP * 0.6), arrows.animate.shift(UP * 0.6), FadeIn(brand, shift=UP * 0.2))
        line.next_to(boxes, DOWN, buff=0.45)
        self.play(FadeIn(line))
        self.wait(3.0)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1.5)
        self.wait(1.0)
