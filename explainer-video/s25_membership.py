from __future__ import annotations

from explainer import *  # noqa: F403
from videos.ml.relflow.common import C, Mono, box, label, show_episode_card
from videos.ml.relflow.current_visuals import heading, trace_tag
from videos.ml.relflow.story import transmit


class Membership(VoiceoverScene):
    def construct(self):
        show_episode_card(self)
        self.sets()
        self.clusters()

    def sets(self):
        head = heading("One label, several labels, or structured items?")
        options = VGroup(
            *[
                box(t, color=c, width=3.7, height=1.0, font_size=25)
                for t, c in [
                    ("Category: one merchant", C.CATEGORY),
                    ("Set: merchant tags", C.HASH),
                    ("Branch: transactions", C.RF_BRANCH),
                ]
            ]
        )
        options.arrange(RIGHT, buff=0.35).move_to(UP * 1.6)
        tags = (
            VGroup(
                *[
                    box(t, color=C.HASH, width=2.6, height=0.7, font_size=23)
                    for t in ["grocery", "essential", "grocery"]
                ]
            )
            .arrange(RIGHT, buff=0.3)
            .move_to(UP * 0.1)
        )
        bits = (
            VGroup(
                box("grocery = 1", color=C.HASH, width=3.1, height=0.7, font_size=25),
                box("essential = 1", color=C.HASH, width=3.1, height=0.7, font_size=25),
            )
            .arrange(RIGHT, buff=0.4)
            .move_to(DOWN * 1.4)
        )
        foot = label(r"empty set $\ne$ null set $\ne$ unavailable membership", font_size=27).move_to(DOWN * 2.4)
        tag = trace_tag("schematic; Set and Cluster semantics verified against current source")
        with self.voiceover(
            "Choose a field for what the value means. Category is one label. Set is unordered membership. "
            "Branch is a collection whose items have their own fields or positions. "
            '<bookmark mark="a"/> Grocery, essential, grocery becomes two memberships. Reordering and duplicates do not change the representation. '
            '<bookmark mark="b"/> An empty set is a valued observation with no positive members. A null set is a different state. '
            "Unknown members use an unavailable contribution; they do not acquire invented identities. "
            "For reconstruction, inspect membership F one, complete set accuracy and coverage together. Sparse bit accuracy can look good while missing the useful labels."
        ) as vo:
            self.play(FadeIn(head), FadeIn(options), FadeIn(tags), FadeIn(tag))
            vo.wait_until("a")
            self.play(Swap(tags[0], tags[1]))
            self.play(TransformFromCopy(tags[0], bits[0]), TransformFromCopy(tags[1], bits[1]), run_time=1.3)
            transmit(self, [tags[2]], bits[0], C.HASH, run_time=0.9)
            self.play(FadeOut(tags[2]))
            vo.wait_until("b")
            self.play(FadeIn(foot))
        self.clear_scene()

    def clusters(self):
        head = heading("A learned group needs a learning objective")
        labels = VGroup(
            *[Mono(t, font_size=24, color=C.CATEGORY) for t in ["Cafe A", "Cafe B", "Market C", "Market D"]]
        )
        labels.arrange(DOWN, buff=0.48).move_to(LEFT * 4.5)
        groups = VGroup(*[Circle(radius=0.6, color=c, fill_opacity=0.15) for c in [C.RF_BRANCH, C.MASS]])
        groups.arrange(DOWN, buff=1).move_to(RIGHT * 0.4)
        routes = VGroup(
            *[Arrow(l.get_right(), groups[i // 2].get_left(), buff=0.1, color=C.CATEGORY) for i, l in enumerate(labels)]
        )
        objective = box(
            r"hide merchant\\reconstruct its label", color=C.MASKED, width=3.5, height=1.2, font_size=24
        ).move_to(RIGHT * 4.6)
        warning = label(
            r"downstream dispute supervision alone does not train cluster assignments", font_size=26, color=C.RF_TARGET
        ).move_to(DOWN * 2.6)
        tag = trace_tag("schematic assignments; no learned groups or recovery scores claimed")
        with self.voiceover(
            "Cluster adds shared learned groups to persistent label identities. The lines here are a schematic, not recovered merchant categories. "
            '<bookmark mark="a"/> A reconstructing mask is required to train the cluster assignments and adaptive selection. '
            'Dispute supervision alone does not do that job. <bookmark mark="b"/> Fixed bounds choose a group count; ranged bounds allow revival and merging within an allocated width. '
            "Neither guarantees human categories. Measure agreement with a known grouping and held out predictive quality separately. "
            "Group confidence and label confidence also answer different questions."
        ) as vo:
            self.play(FadeIn(head), FadeIn(labels), FadeIn(groups), FadeIn(tag))
            vo.wait_until("a")
            self.play(FadeIn(objective), *[GrowArrow(a) for a in routes], run_time=1.5)
            for i, g in enumerate(groups):
                transmit(self, list(labels[2 * i : 2 * i + 2]), g, C.CATEGORY, run_time=0.65)
            transmit(self, list(groups), objective, C.MASKED, run_time=0.9)
            vo.wait_until("b")
            self.play(FadeIn(warning))
        self.clear_scene()
