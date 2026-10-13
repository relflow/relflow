from __future__ import annotations

from explainer import *  # noqa: F403
from videos.relflow.common import (
    D, Mono, box, code, label, note, real_tag, show_episode_card,
)

PRE = '''
@rf.preprocess
def prepare(frame: pl.DataFrame, *, scale: float = 100.0) -> pl.DataFrame:
    accounts = pl.col("accounts")
    return frame.with_columns(
        amount=pl.col("amount_minor") / scale,
        parties=pl.concat_list(
            pl.struct(account=accounts.struct.field("payer"), role=pl.lit("payer")),
            pl.struct(account=accounts.struct.field("payee"), role=pl.lit("payee")),
        ),
    ).drop("accounts", "amount_minor")
'''

MODEL = '''
model = rf.Model.sm(
    parties=rf.Branch(length=2, overflow="error", account=rf.Category, role=rf.Category),
    amount=rf.Number,
    flagged=rf.Boolean(mask=True),
)
data = rf.ArrowDataModule(model=model, train=train, validate=validate, preprocessor=prepare)
'''

POST = '''
@rf.postprocess
def decision(frame: pl.DataFrame, *, threshold: float) -> pl.DataFrame:
    flagged = pl.col("predictions").struct.field(rf.Address("flagged"))
    probability = flagged.struct.field("content").struct.field("probability")
    return frame.select(
        transfer_id=pl.col("inputs").struct.field("transfer_id"),
        flag_probability=probability,
        review=probability >= threshold,
    )

model.predict(requests, preprocess=prepare, retain=("transfer_id",),
              postprocess=decision.partial(threshold=0.5))
'''


def yaml_lines(rows, font_size=20, color=WHITE) -> VGroup:
    g = VGroup(*[Mono(r, font_size=font_size, color=color) for r in rows]).arrange(DOWN, aligned_edge=LEFT, buff=0.06)
    char_w = Mono("M" * 20, font_size=font_size).width / 20
    x0 = g.get_left()[0]
    for m, r in zip(g, rows):
        m.shift(RIGHT * (x0 + (len(r) - len(r.lstrip())) * char_w - m.get_left()[0]))
    return g


class Processors(VoiceoverScene):
    def construct(self):
        show_episode_card(self)
        self.d = D("processors")
        self.pipeline()
        self.reshape()
        self.train()
        self.post()
        self.attach()

    # ------------------------------------------------------------------
    def pipeline(self):
        stages = [(r"your\\data", C.ARROW_BUF), (r"preprocess\\Polars", C.NUMBER),
                  (r"bind to\\the schema", C.RF_FIELD), (r"the model", C.RF_BRANCH), (r"prediction\\table", C.RF_TARGET),
                  (r"postprocess\\Polars", C.NUMBER)]
        boxes = VGroup(*[box(t, color=c, width=2.0, height=1.25, font_size=21) for t, c in stages]).arrange(RIGHT, buff=0.3)
        boxes.move_to(UP * 0.4)
        arrows = VGroup(*[Arrow(a.get_right(), b.get_left(), buff=0.05, stroke_width=3, color=GREY_B,
                                max_tip_length_to_length_ratio=0.4) for a, b in zip(boxes[:-1], boxes[1:])])
        under = VGroup(
            label(r"rename, derive, filter, join, reshape", font_size=22, color=C.NUMBER).next_to(boxes[1], DOWN, buff=0.4),
            label(r"thresholds, business rules, API shapes", font_size=22, color=C.NUMBER).next_to(boxes[5], DOWN, buff=0.4),
        )
        under[0].align_to(boxes[1], LEFT)
        under[1].align_to(boxes[5], RIGHT)
        with self.voiceover(
            "A model only sees what its schema describes, and data rarely arrives in exactly that shape. relflow "
            "gives you two hooks. <bookmark mark='p'/> A preprocessor runs before the schema is applied: an ordinary "
            "Polars function that can rename, derive, filter, join, or reshape. <bookmark mark='o'/> A postprocessor "
            "runs after prediction, and turns relflow's prediction table into whatever your application needs."
        ) as vo:
            self.play(LaggedStart(*[FadeIn(b, shift=RIGHT * 0.1) for b in boxes], lag_ratio=0.12),
                      LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.12), run_time=2.0)
            vo.wait_until("p")
            self.play(Indicate(boxes[1], color=WHITE, scale_factor=1.06), FadeIn(under[0]))
            vo.wait_until("o")
            self.play(Indicate(boxes[5], color=WHITE, scale_factor=1.06), FadeIn(under[1]))
        self.play(FadeOut(VGroup(boxes, arrows, under)))

    # ------------------------------------------------------------------
    def reshape(self):
        d = self.d
        s, pr = d["source"], d["processed"]
        src = yaml_lines([f"transfer_id: {s['transfer_id']}", "accounts:", f"  payer: {s['accounts']['payer']}",
                          f"  payee: {s['accounts']['payee']}", f"amount_minor: {s['amount_minor']}"])
        out = yaml_lines([f"transfer_id: {pr['transfer_id']}", f"amount: {pr['amount']:.2f}", "parties:",
                          *[ln for p in pr["parties"] for ln in (f"  - account: {p['account']}", f"    role: {p['role']}")]],
                         color=C.NUMBER)
        src_t = label(r"as it arrives", font_size=24, color=GREY_A)
        out_t = label(r"after \texttt{prepare}", font_size=24, color=C.NUMBER)
        left = VGroup(src_t, src).arrange(DOWN, aligned_edge=LEFT, buff=0.25)
        right = VGroup(out_t, out).arrange(DOWN, aligned_edge=LEFT, buff=0.25)
        pair = VGroup(left, right).arrange(RIGHT, buff=2.0).to_edge(DOWN, buff=0.6)
        arr = Arrow(left.get_right(), right.get_left(), buff=0.3, color=C.NUMBER)
        c = code(PRE, font_size=16).to_edge(UP, buff=0.3)
        tag = real_tag(r"real record, processed by the function shown", corner=DR)
        with self.voiceover(
            "Here's a real transfer as it arrives. <bookmark mark='s'/> The two accounts sit in a struct, under payer "
            "and payee, and the amount is in cents. <bookmark mark='c'/> The preprocessor is ten lines of Polars. "
            "<bookmark mark='o'/> It converts cents to an amount, and turns the two named accounts into a repeated "
            "list of parties, each with an account and a role. <bookmark mark='w'/> Why bother? Now both accounts "
            "share one account field, so the model learns about account-004 once, whichever side it's on, and the "
            "role tells it which side that was."
        ) as vo:
            vo.wait_until("s")
            self.play(FadeIn(left), FadeIn(tag))
            vo.wait_until("c")
            self.play(FadeIn(c, shift=DOWN * 0.1))
            vo.wait_until("o")
            self.play(GrowArrow(arr), FadeIn(right, lag_ratio=0.1))
            vo.wait_until("w")
            self.play(Indicate(VGroup(*out[3:]), color=WHITE, scale_factor=1.03))
        self.play(FadeOut(VGroup(left, right, arr, c, tag)))

    # ------------------------------------------------------------------
    def train(self):
        d = self.d
        st = d["stats"]
        c = code(MODEL, font_size=19).to_edge(UP, buff=0.5)
        rule = label(rf"flagged $\iff$ the payee is on an {d['watchlist_size']}-account watchlist, or the amount is over"
                     r" 2{,}500", font_size=24, color=C.RF_TARGET).next_to(c, DOWN, buff=0.45)
        rows = VGroup(
            label(rf"{d['train_rows']:,} training transfers, {d['seconds']:.0f} seconds; {d['test_rows']:,} test"
                  .replace(",", "{,}"), font_size=24, color=GREY_A),
            label(rf"accuracy {st['accuracy'] * 100:.2f}\%", font_size=28),
            label(rf"watchlisted \emph{{payees}} flagged: {st['watched_payee_recall'] * 100:.0f}\%", font_size=28,
                  color=C.RF_TARGET),
            label(rf"the same accounts as \emph{{payers}} flagged: {st['watched_payer_flag_rate'] * 100:.0f}\%",
                  font_size=28, color=C.VALUED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.22).next_to(rule, DOWN, buff=0.45)
        tag = real_tag(r"real training run with the preprocessor attached")
        assert st["watched_payee_recall"] == 1.0 and st["watched_payer_flag_rate"] == 0.0
        with self.voiceover(
            "The schema describes the processed shape: a repeated branch of exactly two parties, an amount, and a "
            "hidden flag. <bookmark mark='p'/> The preprocessor goes on the data module, and runs on every batch, in "
            "every split. <bookmark mark='r'/> The rule to learn: flag a transfer if the money goes to one of eight "
            "watched accounts, or if it's over two and a half thousand. <bookmark mark='a'/> On two thousand new "
            f"transfers, it's right {st['accuracy'] * 100:.2f} percent of the time. <bookmark mark='w'/> Every transfer to a watched account "
            "is flagged, <bookmark mark='f'/> and none of the transfers merely sent from one. The role came through "
            "the reshape."
        ) as vo:
            self.play(FadeIn(c))
            vo.wait_until("p")
            self.play(Indicate(c.code_lines[5], color=WHITE, scale_factor=1.02))
            vo.wait_until("r")
            self.play(FadeIn(rule))
            vo.wait_until("a")
            self.play(FadeIn(rows[:2], lag_ratio=0.3), FadeIn(tag))
            vo.wait_until("w")
            self.play(FadeIn(rows[2]))
            vo.wait_until("f")
            self.play(FadeIn(rows[3]))
        self.play(FadeOut(VGroup(c, rule, rows, tag)))

    # ------------------------------------------------------------------
    def post(self):
        d = self.d
        c = code(POST, font_size=16).to_edge(UP, buff=0.3)
        head = VGroup(*[Mono(t, font_size=19, color=GREY_A) for t in ("transfer_id", "flag_probability", "review")])
        xs = [-4.6, -0.6, 2.8]
        table = VGroup()
        for t, x in zip(head, xs):
            t.move_to([x, -1.2, 0], aligned_edge=LEFT)
        table.add(head)
        for i, r in enumerate(d["result"]):
            y = -1.65 - 0.42 * i
            table.add(VGroup(Mono(r["transfer_id"], font_size=19).move_to([xs[0], y, 0], aligned_edge=LEFT),
                             Mono(f"{r['flag_probability']:.4f}", font_size=19).move_to([xs[1], y, 0], aligned_edge=LEFT),
                             Mono(str(r["review"]).lower(), font_size=19,
                                  color=C.RF_TARGET if r["review"] else GREY_B).move_to([xs[2], y, 0], aligned_edge=LEFT)))
        tag = real_tag(r"real output of \texttt{model.predict} with both processors")
        assert [r["review"] for r in d["result"]] == d["truth"]
        with self.voiceover(
            "On the way out, relflow returns a prediction table: the columns you chose to keep, and a predictions "
            "struct with a state, a value and an inferred flag for every target. <bookmark mark='c'/> A postprocessor "
            "reshapes that for your application. This one keeps the transfer ID, pulls out the probability, and "
            "applies a review threshold, a business rule that belongs to you, not to the model. "
            "<bookmark mark='o'/> Here's the real output for four new transfers: an ordinary one, one to a watched "
            "account, one from a watched account, and a large one."
        ) as vo:
            vo.wait_until("c")
            self.play(FadeIn(c))
            vo.wait_until("o")
            self.play(FadeIn(table, lag_ratio=0.2), FadeIn(tag), run_time=1.6)
        self.play(FadeOut(VGroup(c, table, tag)))

    # ------------------------------------------------------------------
    def attach(self):
        rows = [
            (r"training", r"\texttt{rf.ArrowDataModule(..., preprocessor=prepare)}"),
            (r"direct prediction", r"\texttt{model.predict(..., preprocess=prepare, postprocess=decision)}"),
            (r"batch jobs", r'\texttt{rf.Writer("predictions", postprocessor=decision)}'),
            (r"serving", r"\texttt{rf.Deployment(model=model).postprocess(decision)}"),
        ]
        g = VGroup(*[VGroup(label(a, font_size=24, color=GREY_A), label(b, font_size=22)).arrange(RIGHT, buff=0.4)
                     for a, b in rows]).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to(UP * 0.9)
        for r in g:
            r[1].set_x(-2.6, LEFT)
            r[0].set_x(-2.9, RIGHT)
        facts = VGroup(
            label(r"processors are not saved in the model: pass them again when you predict", font_size=24),
            label(r'by default each runs per partition, streamed; \texttt{scope="dataset"} gathers a whole split',
                  font_size=24, color=GREY_A),
            label(r"keep a key with \texttt{retain}: relflow won't restore the original rows or order", font_size=24,
                  color=GREY_A),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).to_edge(DOWN, buff=0.6)
        with self.voiceover(
            "The same two functions plug into every path: <bookmark mark='t'/> training, <bookmark mark='p'/> direct "
            "prediction, <bookmark mark='b'/> batch jobs that write Parquet, <bookmark mark='s'/> and an HTTP "
            "deployment. <bookmark mark='f'/> Three things to remember. Processors aren't stored in the model file, "
            "so pass them again when you predict. By default they run on each partition as it streams; ask for the "
            "dataset scope only when a transform needs the whole split, which then has to fit in memory. And keep a "
            "key, like the transfer ID, because a filter or a join can change which rows come out."
        ) as vo:
            for mark, r in zip("tpbs", g):
                vo.wait_until(mark)
                self.play(FadeIn(r, shift=RIGHT * 0.1), run_time=0.7)
            vo.wait_until("f")
            self.play(FadeIn(facts, lag_ratio=0.3), run_time=1.5)
        self.wait(0.5)
        self.play(FadeOut(VGroup(g, facts)))
