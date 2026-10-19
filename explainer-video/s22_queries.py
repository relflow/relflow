from __future__ import annotations

from explainer import *  # noqa: F403
from videos.ml.relflow.common import C, D, Mono, box, label, show_episode_card
from videos.ml.relflow.current_visuals import heading, trace_tag
from videos.ml.relflow.story import stream


class QueryPaths(VoiceoverScene):
    def construct(self):
        show_episode_card(self)
        head = heading("Follow the structure. Keep your own names.")
        source = VGroup(
            *[
                Mono(t, font_size=24, color=c)
                for t, c in [
                    ("payload", C.RF_ROOT),
                    ("  transactions", C.RF_BRANCH),
                    ("    gross_amount: 32.00", C.NUMBER),
                    ("    gross_amount: 56.20", C.NUMBER),
                    ("    gross_amount: 84.50", C.NUMBER),
                ]
            ]
        )
        source.arrange(DOWN, aligned_edge=LEFT, buff=0.24).move_to(LEFT * 3.5 + UP * 0.5)
        nodes = VGroup(
            *[
                box(t, color=c, width=3.8, height=0.8, font_size=24)
                for t, c in [
                    ("root: payload", C.RF_ROOT),
                    ("trxns: transactions[-2:]", C.RF_BRANCH),
                    (r"amount: gross\_amount", C.NUMBER),
                ]
            ]
        )
        nodes.arrange(DOWN, buff=0.5).move_to(RIGHT * 3.3 + UP * 0.5)
        routes = VGroup(
            Arrow(source[0].get_right(), nodes[0].get_left(), buff=0.15, color=C.RF_ROOT),
            Arrow(source[1].get_right(), nodes[1].get_left(), buff=0.15, color=C.RF_BRANCH),
        )
        selected = SurroundingRectangle(source[3:], color=C.NUMBER, buff=0.13)
        values = D("current")["query"]["retained_values"]
        assert len(values) == 2
        output = Mono("amount = [56.20, 84.50]", font_size=28, color=C.NUMBER).move_to(DOWN * 2.2)
        tag = trace_tag("real structural query selection; relflow 0.3.2")
        with self.voiceover(
            "What if the source calls a field something else? A query lets the schema keep a useful name. "
            '<bookmark mark="r"/> The root selects payload. <bookmark mark="b"/> The branch selects the last two transaction records. '
            '<bookmark mark="f"/> Each amount then reads gross amount relative to its own selected transaction. '
            "The real encoder retains fifty six twenty and eighty four fifty. The first transaction is outside the selected slice. "
            "The schema address is slash trxns, slash amount. Source paths and schema addresses answer different questions."
        ) as vo:
            self.play(FadeIn(head), FadeIn(source), FadeIn(tag))
            vo.wait_until("r")
            self.play(FadeIn(nodes[0]), GrowArrow(routes[0]))
            stream(self, [routes[0]], C.RF_ROOT, run_time=0.65)
            vo.wait_until("b")
            self.play(FadeIn(nodes[1]), GrowArrow(routes[1]), Create(selected))
            stream(self, [routes[1]], C.RF_BRANCH, run_time=0.65)
            vo.wait_until("f")
            self.play(FadeIn(nodes[2]), TransformFromCopy(source[3:], output), run_time=1.5)
        self.clear_scene()
        head = heading("Select whole records. Preserve alignment.")
        rows = VGroup(
            *[
                VGroup(
                    *[
                        box(t, color=c, width=2.5, height=0.7, font_size=23)
                        for t, c in [(amount, C.NUMBER), (merchant, C.CATEGORY), (time, C.RF_BRANCH)]
                    ]
                ).arrange(RIGHT, buff=0.15)
                for amount, merchant, time in [
                    ("32.00", "Station Cafe", "May 21"),
                    ("56.20", "North Market", "May 27"),
                    ("84.50", "North Market", "May 28"),
                ]
            ]
        )
        rows.arrange(DOWN, buff=0.3).move_to(UP * 0.6)
        boundary = Line([-5, -1.4, 0], [5, -1.4, 0], color=C.RF_TARGET)
        rules = label(
            r"queries: members, indices, slices, map keys\\preprocessors: filters, joins, sorting, derived values",
            font_size=26,
        ).move_to(DOWN * 2.2)
        with self.voiceover(
            "Slices select whole records before branch geometry is resolved. Amount, merchant and timestamp stay together. "
            '<bookmark mark="a"/> Queries navigate structure. Put filtering, joins, sorting and derived values in a preprocessor. '
            "A directly bound hidden target can be omitted at prediction time, but its containing branch must remain to define output positions. "
            "For an explicitly queried target, preserve the queried source field, using null labels at prediction time."
        ) as vo:
            self.play(FadeIn(head), FadeIn(rows, lag_ratio=0.2))
            self.play(rows[0].animate.set_opacity(0.2), rows[1:].animate.shift(UP * 0.35))
            vo.wait_until("a")
            self.play(Create(boundary), FadeIn(rules))
        self.clear_scene()
