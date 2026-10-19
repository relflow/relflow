from __future__ import annotations

from explainer import *  # noqa: F403
from videos.ml.relflow.common import C, D, Mono, box, label, show_episode_card
from videos.ml.relflow.current_visuals import heading, trace_tag


class Output(VoiceoverScene):
    def construct(self):
        show_episode_card(self)
        self.structure()
        self.coordinates()
        self.batching()

    def structure(self):
        d = D("current")
        rows = d["output"]
        assert rows[0]["predictions"]["trxns"] is None and rows[1]["predictions"]["trxns"] == []
        assert rows[2]["predictions"]["trxns"][1] is None
        head = heading("A prediction keeps its place")
        left = VGroup(
            *[
                Mono(t, font_size=25, color=c)
                for t, c in [("null", C.NULL_STATE), ("[]", C.RF_BRANCH), ("[32.00, hole, 84.50]", C.NUMBER)]
            ]
        )
        left.arrange(DOWN, aligned_edge=LEFT, buff=0.8).move_to(LEFT * 3.3 + UP * 0.4)
        right = VGroup(
            *[
                Mono(t, font_size=25, color=c)
                for t, c in [("null", C.NULL_STATE), ("[]", C.RF_BRANCH), ("[result, null, result]", C.RF_TARGET)]
            ]
        )
        right.arrange(DOWN, aligned_edge=LEFT, buff=0.8).move_to(RIGHT * 3.1 + UP * 0.4)
        arrows = VGroup(
            *[
                Arrow(a.get_right(), b.get_left(), buff=0.3, color=c)
                for a, b, c in zip(left, right, [C.NULL_STATE, C.RF_BRANCH, C.RF_TARGET])
            ]
        )
        captions = VGroup(
            Mono("input: trxns", font_size=24, color=C.ARROW_BUF).next_to(left, UP, buff=0.6),
            Mono("predictions.trxns", font_size=24, color=C.RF_TARGET).next_to(right, UP, buff=0.6),
        )
        foot = label(r"variable Arrow lists of records; no cap padding in public output", font_size=27).move_to(
            DOWN * 2.3
        )
        tag = trace_tag("real output structure; probabilities omitted; untrained model, relflow 0.3.2")
        with self.voiceover(
            "A prediction belongs to a record, not just to a tensor slot. "
            '<bookmark mark="a"/> A null transaction collection stays null. An empty collection stays empty. '
            '<bookmark mark="b"/> A missing item inside a collection keeps its place. Two valid records around it receive two predictions. '
            "Sibling predictions share one record and one set of list boundaries. "
            '<bookmark mark="c"/> Even a branch with a finite cap writes variable Arrow lists. The output does not expose unused cap padding. '
            "These structures come from an actual untrained model. They demonstrate the contract, without claiming learned accuracy."
        ) as vo:
            self.play(FadeIn(head), FadeIn(left), FadeIn(captions), FadeIn(tag))
            vo.wait_until("a")
            self.play(FadeIn(right[:2]), *[GrowArrow(a) for a in arrows[:2]])
            vo.wait_until("b")
            self.play(GrowArrow(arrows[2]), FadeIn(right[2]))
            vo.wait_until("c")
            self.play(FadeIn(foot))
        self.clear_scene()

    def coordinates(self):
        head = heading("One record, three selected output planes")
        planes = VGroup(
            *[
                box(t, color=c, width=3.6, height=0.95, font_size=26)
                for t, c in [("inputs", C.ARROW_BUF), ("predictions", C.RF_TARGET), ("representations", C.EMBED)]
            ]
        )
        planes.arrange(RIGHT, buff=0.45).move_to(UP * 1.7)
        path = VGroup(
            *[
                Mono(t, font_size=24, color=c)
                for t, c in [
                    ("path: [trxns, disputed]", C.RF_BRANCH),
                    ("parents: [2]", C.RF_TARGET),
                    ("output_index: 0", C.MASS),
                    ("embedding: [ ... 16 floats ... ]", C.EMBED),
                ]
            ]
        )
        path.arrange(DOWN, aligned_edge=LEFT, buff=0.24).move_to(DOWN * 0.6 + RIGHT * 1.7)
        slots = (
            VGroup(
                *[
                    box(str(i), color=C.RF_BRANCH if i != 1 else C.PADDED, width=0.7, height=0.7, font_size=24)
                    for i in range(3)
                ]
            )
            .arrange(RIGHT, buff=0.16)
            .move_to(LEFT * 3.7 + DOWN * 0.5)
        )
        hl = SurroundingRectangle(slots[2], color=C.RF_TARGET, buff=0.08)
        link = Arrow(slots[2].get_right(), path[1].get_left(), color=C.RF_TARGET, buff=0.2)
        norms = label(r"\texttt{embed=True} exports context; it creates no training objective", font_size=26).move_to(
            DOWN * 2.6
        )
        reps = D("current")["output"][2]["representations"]
        assert any(v["path"] == ["trxns", "disputed"] and v["parents"] == [2] for v in reps)
        with self.voiceover(
            "Retention selects inputs. Hidden targets select decoded predictions. Embed true selects representations. "
            '<bookmark mark="a"/> Each embedding names its site with a path, repeated ancestor ordinals, and an intrinsic output index. '
            '<bookmark mark="b"/> This vector belongs to the dispute field on transaction two. The null item at position one has not shifted that identity. '
            "The anonymous root uses an empty path. A branch summary belongs to the branch, rather than being duplicated for every child. "
            '<bookmark mark="c"/> Embeddings are normalized float vectors. Exporting one creates no objective. Inferred means requested reconstruction, not confidence.'
        ) as vo:
            self.play(FadeIn(head), FadeIn(planes, lag_ratio=0.2))
            vo.wait_until("a")
            self.play(FadeIn(path, lag_ratio=0.2), FadeIn(slots))
            vo.wait_until("b")
            self.play(Create(hl), GrowArrow(link))
            vo.wait_until("c")
            self.play(FadeIn(norms))
        self.clear_scene()

    def batching(self):
        head = heading("Batch computation. Keep one table contract.")
        steps = VGroup(
            *[
                box(t, color=c, width=2.3, height=1.1, font_size=23)
                for t, c in [
                    (r"preprocess\\once", C.ARROW_BUF),
                    (r"chunk rows\\before encode", C.RF_BRANCH),
                    (r"combine\\Arrow outputs", C.RF_TARGET),
                    (r"postprocess\\once", C.VALUED),
                ]
            ]
        )
        steps.arrange(RIGHT, buff=0.65)
        arrows = VGroup(*[Arrow(a.get_right(), b.get_left(), buff=0.1) for a, b in zip(steps, steps[1:])])
        metadata = Mono("relflow.prediction_format = 2", font_size=26, color=C.RF_TARGET).move_to(DOWN * 1.8)
        with self.voiceover(
            "Explicit batch size on direct prediction preprocesses the full input once, chunks processed rows before encoding, "
            'combines Arrow outputs, and postprocesses once. <bookmark mark="a"/> Omitting it preserves whole input direct execution. '
            "Use data modules and a Writer for streaming file jobs. The table records prediction format two in its metadata. "
            "For fixed width vectors beneath null containers, use Arrow IPC: the Parquet writer rejects that unsupported round trip."
        ) as vo:
            self.play(
                FadeIn(head), LaggedStart(*[FadeIn(s) for s in steps], lag_ratio=0.2), *[GrowArrow(a) for a in arrows]
            )
            vo.wait_until("a")
            self.play(FadeIn(metadata))
        self.clear_scene()
