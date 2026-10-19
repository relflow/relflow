from __future__ import annotations

from explainer import *  # noqa: F403
from videos.ml.relflow.common import C, D, Mono, box, code, label, show_episode_card
from videos.ml.relflow.current_visuals import heading, trace_tag

THREE = """
model = rf.Model.xs(amount=rf.Number, merchant=rf.Category, disputed=rf.Boolean(mask=True))
data = rf.ArrowDataModule(model=model, train="payments/train.parquet", validate="payments/validate.parquet")
model.fit(data, batch_size=128, learning_rate=1e-4, weight_decay=0.01, max_epochs=30)
"""


class ThreeLines(VoiceoverScene):
    def construct(self):
        show_episode_card(self)
        self.payment()
        self.history()
        self.fit()
        self.predict()

    def payment(self):
        head = heading("One payment. One question.")
        fields = VGroup(
            *[
                box(t, color=c, width=3.3, height=1.0, font_size=26)
                for t, c in [
                    ("amount: 84.50", C.NUMBER),
                    ("merchant: North Market", C.CATEGORY),
                    ("disputed: false", C.RF_TARGET),
                ]
            ]
        )
        fields.arrange(RIGHT, buff=0.5).move_to(UP * 0.8)
        hidden = box("disputed: ?", color=C.MASKED, width=3.3, height=1.0, font_size=26).move_to(fields[2])
        model = box("learn the dispute risk", color=C.RF_ROOT, width=5.4, height=1.0).move_to(DOWN * 1.3)
        arrows = VGroup(
            *[
                Arrow(f.get_bottom(), model.get_top(), buff=0.15, color=c)
                for f, c in zip(fields[:2], [C.NUMBER, C.CATEGORY])
            ]
        )
        caption = label(
            r"\texttt{mask=True}: supervise this field; exclude its label from input", font_size=26
        ).move_to(DOWN * 2.7)
        with self.voiceover(
            "Start where the current documentation starts: one payment, one target, and a small model. "
            "<bookmark mark='a'/> The amount is a number. The merchant is a category. The question is whether a dispute follows. "
            "<bookmark mark='b'/> Mask true keeps the answer out of the inputs while making it a supervised target. "
            "During training the model sees the payment and is scored against its settled label. At prediction time, omit that label."
        ) as vo:
            self.play(FadeIn(head))
            vo.wait_until("a")
            self.play(FadeIn(fields, lag_ratio=0.25))
            vo.wait_until("b")
            self.play(Transform(fields[2], hidden), FadeIn(model), FadeIn(caption), *[GrowArrow(a) for a in arrows])
        self.clear_scene()

    def history(self):
        head = heading("Add history at one review cutoff")
        axis = NumberLine(x_range=[-8, 6, 1], length=11, include_ticks=False).move_to(DOWN * 0.5)
        cutoff = DashedLine(axis.n2p(0) + DOWN * 0.7, axis.n2p(0) + UP * 2.1, color=C.RF_TARGET)
        before = Line(axis.n2p(-8), axis.n2p(0), stroke_width=18, color=C.RF_BRANCH, stroke_opacity=0.25)
        future = Line(axis.n2p(0), axis.n2p(6), stroke_width=18, color=C.RF_TARGET, stroke_opacity=0.25)
        dots = VGroup(*[Dot(axis.n2p(t), radius=0.1, color=C.RF_BRANCH) for t in [-6, -3, -1]])
        cards = VGroup(
            *[
                box(text, color=C.RF_BRANCH, width=1.4, height=0.7, font_size=23).next_to(dot, UP, buff=0.5)
                for dot, text in zip(dots, ["32.00", "56.20", "84.50"])
            ]
        )
        label = Mono("review cutoff", font_size=23, color=C.RF_TARGET).next_to(cutoff, UP, buff=0.2)
        known = Mono("inputs known by cutoff", font_size=23, color=C.RF_BRANCH).move_to(LEFT * 3 + DOWN * 1.4)
        window = Mono("next 30 days: labels", font_size=23, color=C.RF_TARGET).move_to(RIGHT * 3 + DOWN * 1.4)
        labels = VGroup(
            *[
                box("?", color=C.MASKED, width=0.65, height=0.65, font_size=27).next_to(card, UP, buff=0.18)
                for card in cards
            ]
        )
        code_line = Mono("trxns: booked_at, amount, merchant, disputed=Boolean(mask=True)", font_size=21).move_to(
            DOWN * 2.6
        )
        with self.voiceover(
            "When a payment needs context, put earlier transactions and the latest payment in one history, exactly once. "
            '<bookmark mark="a"/> Order them oldest to newest; a finite cap with tail overflow keeps the latest records. '
            '<bookmark mark="b"/> At one account review cutoff, every input must already be known. Each transaction label asks whether a new dispute follows in the next thirty days. '
            '<bookmark mark="c"/> Mask true inside the branch hides every dispute label and supervises every retained transaction. '
            "Training needs settled labels for all of them. At prediction, keep the transaction records and omit their labels. "
            "Calendar features describe cycles; derive a Number if exact elapsed time matters."
        ) as vo:
            self.play(FadeIn(head), Create(axis), Create(before), Create(future))
            vo.wait_until("a")
            self.play(FadeIn(dots, lag_ratio=0.2), FadeIn(cards, lag_ratio=0.2))
            vo.wait_until("b")
            self.play(Create(cutoff), FadeIn(label), FadeIn(known), FadeIn(window))
            vo.wait_until("c")
            self.play(FadeIn(labels), FadeIn(code_line))
        self.clear_scene()

    def fit(self):
        c = code(THREE, font_size=18).to_edge(UP, buff=0.6)
        stages = VGroup(
            *[
                box(t, color=col, width=2.1, height=0.9, font_size=23)
                for t, col in [
                    ("Arrow rows", C.ARROW_BUF),
                    ("encode", C.EMBED),
                    ("tree forward", C.RF_BRANCH),
                    ("loss", C.RF_TARGET),
                    ("AdamW step", C.MASS),
                ]
            ]
        )
        stages.arrange(RIGHT, buff=0.45).move_to(DOWN * 0.1)
        arrows = VGroup(
            *[Arrow(a.get_right(), b.get_left(), buff=0.08, stroke_width=2) for a, b in zip(stages, stages[1:])]
        )
        settings = label(
            r"\texttt{fit}: batch size, optimizer options, hardware and Lightning settings", font_size=26
        ).move_to(DOWN * 1.6)
        policy = label(
            r"validate each epoch $\rightarrow$ stop early $\rightarrow$ restore the best model",
            font_size=26,
            color=C.VALUED,
        ).move_to(DOWN * 2.3)
        d = D("current")["fit"]
        assert d["prediction_rows"] == 2
        tag = trace_tag("current API smoke-tested on Arrow data; file paths above are application inputs")
        with self.voiceover(
            'The workflow is still three lines. <bookmark mark="a"/> Define the model, wrap application supplied Parquet splits, '
            "and call model fit. Batch size and AdamW settings belong on fit. Lightning still runs the loops underneath. "
            '<bookmark mark="b"/> Arrow rows become tensors, flow up the tree, produce a target loss, and update the weights. '
            '<bookmark mark="c"/> By default, validation supports early stopping and restoration of the best validation model. '
            "Each new fit starts fresh optimization state from the current parameters. Use a checkpoint path to resume full training state."
        ) as vo:
            vo.wait_until("a")
            self.play(FadeIn(c), FadeIn(tag))
            vo.wait_until("b")
            self.play(
                LaggedStart(*[FadeIn(s) for s in stages], lag_ratio=0.18), *[GrowArrow(a) for a in arrows], run_time=2
            )
            self.play(FadeIn(settings))
            vo.wait_until("c")
            self.play(FadeIn(policy))
        self.clear_scene()

    def predict(self):
        c = code(
            """
requests = rf.ArrowDataModule(model=model, predict="payments/requests.parquet", retain=("payment_id",))
result = model.predict(datamodule=requests, batch_size=256, devices=1)
model.save("artifacts/payment.rf")
""",
            font_size=18,
        ).to_edge(UP, buff=0.6)
        columns = VGroup(
            *[
                box(t, color=col, width=3.7, height=1.3, font_size=25)
                for t, col in [
                    (r"inputs\\payment identity", C.ARROW_BUF),
                    (r"predictions\\disputed probability", C.RF_TARGET),
                    (r"representations\\only if requested", C.EMBED),
                ]
            ]
        )
        columns.arrange(RIGHT, buff=0.35).move_to(DOWN * 0.5)
        with self.voiceover(
            "For files, put the request path on a data module. For an in memory Arrow table, use direct prediction. "
            '<bookmark mark="a"/> The result is an Arrow table. Retention adds payment identity under inputs. '
            "Decoded values live under predictions, with ordinary schema field names. "
            '<bookmark mark="b"/> Ask for embeddings explicitly with embed true, and they appear in a separate representations list. '
            "An unrequested output column is omitted. Save the model to preserve its schema, weights, vocabularies and normalizers."
        ) as vo:
            self.play(FadeIn(c))
            vo.wait_until("a")
            self.play(FadeIn(columns[:2]))
            vo.wait_until("b")
            self.play(FadeIn(columns[2]))
        self.clear_scene()
