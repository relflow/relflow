from __future__ import annotations

import numpy as np

from explainer import *  # noqa: F403
from videos.relflow.common import (
    D, Mono, Plot, box, code, label, note, real_tag, show_episode_card,
)

SCHEMA = '''
schema = dict(
    line_items=rf.Branch(length=8, sku=rf.Category, quantity=rf.Number, price=rf.Number),
    returned=rf.Boolean(mask=True),
)
'''

THREE = '''
model = rf.Model.sm(**schema, batch_size=128, optimizer=rf.adamw(learning_rate=2e-3))
data = rf.ArrowDataModule(model=model, train=train, validate=validate, test=test)
lit.Trainer(max_epochs=6).fit(model, datamodule=data)
'''

AFTER = '''
model.save("orders.rf")
restored = rf.Model.load("orders.rf")
predictions = restored.predict(requests, retain=("order_id",))
'''


class ThreeLines(VoiceoverScene):
    def construct(self):
        show_episode_card(self)
        self.d = D("happy")
        self.lines()
        self.run()
        self.loop()
        self.stages()
        self.after()

    # ------------------------------------------------------------------
    def lines(self):
        d = self.d
        s = code(SCHEMA, font_size=20).to_edge(UP, buff=0.55)
        t = code(THREE, font_size=20).next_to(s, DOWN, buff=0.45)
        hls = [SurroundingRectangle(t.code_lines[i], buff=0.05, color=c, stroke_width=2, corner_radius=0.04)
               for i, c in enumerate((C.RF_ROOT, C.ARROW_BUF, C.RF_TARGET))]
        nums = VGroup(*[label(str(i + 1), font_size=30, color=h.get_color()).next_to(h, LEFT, buff=0.2)
                        for i, h in enumerate(hls)])
        notes = VGroup(
            label(r"the schema becomes a model; \texttt{sm} is the small preset", font_size=24, color=C.RF_ROOT),
            label(r"Arrow tables, Parquet datasets or Polars frames, streamed in batches", font_size=24, color=C.ARROW_BUF),
            label(r"Lightning runs the loop", font_size=24, color=C.RF_TARGET),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.22).next_to(t, DOWN, buff=0.55).align_to(t, LEFT)
        mac = note(r"on this M1 the Trainer is pinned to the CPU, which is faster here: see the benchmarks episode",
                   font_size=18).to_corner(DR, buff=0.25)
        assert d["preset"] == "sm" and d["batch_size"] == 128
        with self.voiceover(
            "Here's the whole thing, start to finish. <bookmark mark='s'/> First, the shape of one record: an order "
            "with up to eight line items, each with a SKU, a quantity and a price, and one value to predict. Was the "
            "order returned? <bookmark mark='a'/> Line one loads that schema into a model. S M is the small preset: "
            "it chooses the widths and depths. The batch size and the optimizer travel with the model. "
            "<bookmark mark='b'/> Line two wraps the data. A data module takes Arrow tables, Parquet datasets on disk, "
            "or Polars frames, and streams them in batches. <bookmark mark='c'/> And line three hands both to "
            "Lightning, which runs the training loop. That's it."
        ) as vo:
            vo.wait_until("s")
            self.play(FadeIn(s, shift=UP * 0.1))
            vo.wait_until("a")
            self.play(FadeIn(t, shift=UP * 0.1), Create(hls[0]), FadeIn(nums[0]), FadeIn(notes[0]))
            vo.wait_until("b")
            self.play(Create(hls[1]), FadeIn(nums[1]), FadeIn(notes[1]))
            vo.wait_until("c")
            self.play(Create(hls[2]), FadeIn(nums[2]), FadeIn(notes[2]), FadeIn(mac))
        self.wait(0.5)
        self.play(FadeOut(VGroup(s, t, *hls, nums, notes, mac)))

    # ------------------------------------------------------------------
    def run(self):
        d = self.d
        loss = np.array(d["loss"])
        steps = np.arange(1, len(loss) + 1)
        smooth = np.convolve(loss, np.ones(4) / 4, mode="valid")
        pl = Plot((0, 200), (0, 1.0), width=7.4, height=4.0, x_ticks=[0, 64, 128, 192], y_ticks=[0, 0.5, 1.0],
                  font_size=22).move_to(LEFT * 2.3 + DOWN * 0.3)
        curve = pl.line(steps[3:], smooth, ORANGE, 3)
        xt = pl.x_title(r"training step (batches of 128 orders)", font_size=22)
        yt = label(r"training loss", font_size=22, color=ORANGE).next_to(pl, UP, buff=0.15).align_to(pl, LEFT)
        marks = VGroup()
        for e in d["epochs"]:
            marks.add(VGroup(pl.vline(e["steps"], color=GREY_D, stroke_width=1.5),
                             label(f"{e['auc']:.2f}", font_size=19, color=C.VALUED)
                             .move_to(pl.c2p(e["steps"], 0.92)).shift(LEFT * 0.3)))
        side = VGroup(
            label(rf"{d['params'] / 1e6:.2f} million parameters", font_size=26),
            label(rf"{d['train_rows']:,} orders, {len(d['epochs'])} epochs".replace(",", "{,}"), font_size=26),
            label(rf"{d['seconds']:.0f} seconds on this CPU", font_size=26),
            label(r"validation AUC per epoch:", font_size=24, color=GREY_A),
            label(", ".join(f"{e['auc']:.2f}" for e in d["epochs"]), font_size=24, color=C.VALUED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).to_edge(RIGHT, buff=0.45)
        tag = real_tag(r"the three lines above, run for real: relflow 0.2.3 + Lightning")
        e0, e1 = d["epochs"][0]["auc"], d["epochs"][1]["auc"]
        assert e1 > 0.999 and e0 < 0.9
        with self.voiceover(
            f"Here are those three lines, run for real, on {d['train_rows']:,} synthetic orders. An order comes back "
            "if it holds a fragile product bought three or more at a time, a rule the model has to discover. "
            f"<bookmark mark='p'/> The small preset is about {d['params'] / 1e6:.1f} million parameters. "
            f"<bookmark mark='l'/> Six epochs take {d['seconds']:.0f} seconds. <bookmark mark='a'/> After the first "
            f"epoch, the validation AUC is {e0:.2f}. After the second, it's one."
        ) as vo:
            self.play(Create(pl), FadeIn(xt), FadeIn(yt), FadeIn(tag))
            vo.wait_until("p")
            self.play(FadeIn(side[:2], lag_ratio=0.3))
            vo.wait_until("l")
            self.play(Create(curve), FadeIn(side[2]), run_time=2.4, rate_func=linear)
            vo.wait_until("a")
            self.play(FadeIn(marks, lag_ratio=0.3), FadeIn(side[3:], lag_ratio=0.3), run_time=1.5)
        self.wait(0.5)
        self.play(FadeOut(VGroup(pl, curve, xt, yt, marks, side, tag)))

    # ------------------------------------------------------------------
    def loop(self):
        steps = [
            (r"Arrow batch", C.ARROW_BUF), (r"encode\\to tensors", C.VALUED), (r"forward\\through the tree", C.RF_BRANCH),
            (r"one loss\\per target", C.RF_TARGET), (r"backward,\\AdamW step", C.EMBED),
        ]
        boxes = VGroup(*[box(t, color=c, width=2.15, height=1.2, font_size=23) for t, c in steps])
        boxes.arrange(RIGHT, buff=0.42).move_to(UP * 1.0)
        arrows = VGroup(*[Arrow(a.get_right(), b.get_left(), buff=0.06, stroke_width=3, color=GREY_B,
                                max_tip_length_to_length_ratio=0.4) for a, b in zip(boxes[:-1], boxes[1:])])
        back = CurvedArrow(boxes[-1].get_bottom() + DOWN * 0.1, boxes[0].get_bottom() + DOWN * 0.1, angle=-TAU / 6,
                           color=GREY_B, stroke_width=2.5)
        learn = VGroup(
            label(r"while training, fields learn too:", font_size=26),
            label(r"$\cdot$ Category vocabularies admit new labels", font_size=24, color=GREY_A),
            label(r"$\cdot$ Number normalizers update their running mean and variance", font_size=24, color=GREY_A),
            label(r"validation, test and prediction only read them", font_size=24, color=GREY_A),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.16).to_edge(DOWN, buff=0.5)
        with self.voiceover(
            "What does that third line do? Every step is the same short loop. <bookmark mark='a'/> A batch of Arrow "
            "records is encoded into tensors, <bookmark mark='b'/> run forward through the tree, <bookmark mark='c'/> "
            "scored with one loss per target, summed, <bookmark mark='d'/> and the optimizer takes a step. "
            "<bookmark mark='l'/> Some learning happens outside the gradients, too. While training, vocabularies admit "
            "new labels and number fields update their statistics. Validation, testing and prediction only read them, "
            "so a score never depends on the data it's scoring."
        ) as vo:
            self.play(FadeIn(boxes[0]))
            for mark, i in zip("abcd", range(1, 5)):
                vo.wait_until(mark)
                self.play(GrowArrow(arrows[i - 1]), FadeIn(boxes[i], shift=RIGHT * 0.1), run_time=0.7)
            self.play(Create(back))
            vo.wait_until("l")
            self.play(FadeIn(learn, lag_ratio=0.2))
        self.play(FadeOut(VGroup(boxes, arrows, back, learn)))

    # ------------------------------------------------------------------
    def stages(self):
        d = self.d
        rows = [
            (r"\texttt{trainer.fit}", r"\texttt{train}, \texttt{validate}", r"optimize; validate each epoch"),
            (r"\texttt{trainer.validate}", r"\texttt{validate}", r"score the validation split"),
            (r"\texttt{trainer.test}", r"\texttt{test}", r"score held-out data, once"),
            (r"\texttt{trainer.predict}", r"\texttt{predict}", r"write predictions"),
        ]
        xs = [-6.3, -2.4, 1.0]
        head = VGroup(*[label(t, font_size=24, color=GREY_A).move_to([x, 2.2, 0], aligned_edge=LEFT)
                        for t, x in zip((r"call", r"split it reads", r"what it does"), xs)])
        g = VGroup()
        for i, r in enumerate(rows):
            g.add(VGroup(*[label(t, font_size=26).move_to([x, 1.4 - 0.7 * i, 0], aligned_edge=LEFT)
                           for t, x in zip(r, xs)]))
        line = Line([-6.4, 1.85, 0], [6.6, 1.85, 0], color=GREY_D, stroke_width=1.5)
        test = d["test"]
        res = label(rf"\texttt{{trainer.test}} on {d['test_rows']:,} new orders: AUC {test['.returned/test.auc.content']:.2f},"
                    rf" accuracy {test['.returned/test.accuracy@0.5.content']:.2f}".replace(",", "{,}", 1), font_size=26,
                    color=C.VALUED).to_edge(DOWN, buff=1.1)
        tag = real_tag(r"real test run, relflow 0.2.3")
        with self.voiceover(
            "Lightning has four loops, and each reads its own split. <bookmark mark='f'/> Fit trains, and validates "
            "after every epoch. <bookmark mark='v'/> Validate and <bookmark mark='t'/> test only score. "
            "<bookmark mark='p'/> Predict writes predictions. The stage matters to relflow too: it decides which masks "
            "are active and whether fields may learn. <bookmark mark='r'/> Testing our model on two thousand new orders: "
            "an AUC of one."
        ) as vo:
            self.play(FadeIn(head), Create(line))
            for mark, r in zip("fvtp", g):
                vo.wait_until(mark)
                self.play(FadeIn(r, shift=RIGHT * 0.1), run_time=0.7)
            vo.wait_until("r")
            self.play(FadeIn(res), FadeIn(tag))
        self.play(FadeOut(VGroup(head, line, g, res, tag)))

    # ------------------------------------------------------------------
    def after(self):
        d = self.d
        c = code(AFTER, font_size=21).to_edge(UP, buff=0.6)
        size = label(rf"\texttt{{orders.rf}}: {d['artifact_bytes'] / 1e6:.1f} MB, the schema, the weights,"
                     r" the vocabularies and the normalizers", font_size=24, color=GREY_A).next_to(c, DOWN, buff=0.35)
        p = next(x for x in d["predictions"] if x["returned"])
        q = next(x for x in d["predictions"] if not x["returned"])
        rows = VGroup()
        for x in (q, p):
            pr = x["prediction"]
            rows.add(VGroup(
                Mono(f"{x['order_id']}", font_size=22, color=GREY_A),
                Mono(f"/returned: probability {pr['content']['probability']:.4f}", font_size=22,
                     color=C.RF_TARGET),
                Mono(f"state: valued {pr['state']['valued']:.4f}", font_size=20, color=C.VALUED),
                Mono(f"inferred: {str(pr['inferred']).lower()}", font_size=20, color=GREY_B),
            ).arrange(RIGHT, buff=0.45))
        truth = VGroup(*[label(r"(kept)" if not x["returned"] else r"(returned)", font_size=22, color=GREY_B)
                         .next_to(r, RIGHT, buff=0.3) for x, r in zip((q, p), rows)])
        for r, tr in zip(rows, truth):
            r.add(tr)
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        rows.set(width=min(rows.width, 12.8)).next_to(size, DOWN, buff=0.6)
        gotcha = VGroup(
            label(r"two things to set yourself:", font_size=24),
            label(rf"a batch size (the presets default to {d['defaults']['batch_size']}) and an optimizer"
                  r" (there's no default)", font_size=24, color=GREY_A),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12).to_edge(DOWN, buff=0.5)
        assert d["defaults"]["batch_size"] == 1 and not d["defaults"]["optimizer_set"]
        tag = real_tag(r"real artifact and predictions", corner=UR)
        with self.voiceover(
            "When you're done, <bookmark mark='s'/> save the model to a single file. It holds the schema, the weights, "
            "and everything the fields learned: vocabularies and normalizers. <bookmark mark='l'/> Load it anywhere, "
            "<bookmark mark='p'/> and predict. Predictions come back as an Arrow table, one row per request, with the "
            "columns you asked to keep. <bookmark mark='r'/> For each target, a probability, the predicted state, and "
            "a flag saying it was inferred. <bookmark mark='g'/> Two things the presets leave to you: a batch size, "
            "since they default to one, and an optimizer."
        ) as vo:
            vo.wait_until("s")
            self.play(FadeIn(c), FadeIn(tag))
            self.play(FadeIn(size))
            vo.wait_until("l")
            self.play(Indicate(c.code_lines[1], color=WHITE, scale_factor=1.03))
            vo.wait_until("p")
            self.play(Indicate(c.code_lines[2], color=WHITE, scale_factor=1.03))
            vo.wait_until("r")
            self.play(FadeIn(rows, lag_ratio=0.3))
            vo.wait_until("g")
            self.play(FadeIn(gotcha, shift=UP * 0.1))
        self.wait(0.6)
        self.play(*[FadeOut(m) for m in self.mobjects])
