from __future__ import annotations

import numpy as np

from explainer import *  # noqa: F403
from videos.relflow.common import (
    D, Mono, Plot, code, label, note, real_tag, show_episode_card,
)

PRESETS = ["xs", "sm", "md", "lg", "xl"]


def gib(mib: float) -> str:
    return f"{mib / 1024:.1f} GB" if mib >= 1024 else f"{mib:.0f} MB"


def params_text(n: int) -> str:
    return f"{n / 1e6:.1f}M" if n >= 1e6 else f"{n / 1e3:.0f}K"


class Benchmarks(VoiceoverScene):
    def construct(self):
        show_episode_card(self)
        self.d = D("benchmarks")
        self.runs = self.d["runs"]
        self.workload()
        self.presets()
        self.devices()
        self.context()
        self.advice()

    def run(self, preset, mode, device="cpu", length=64):
        return next(r for r in self.runs if r["preset"] == preset and r["mode"] == mode and r["device"] == device
                    and r["length"] == length)

    # ------------------------------------------------------------------
    def workload(self):
        d = self.d
        m = d["machine"]
        c = code('''model = rf.Model.md(            # and .xs, .sm, .lg, .xl
    tenure_days=rf.Number,
    orders=rf.Branch(length=64, overflow="tail", age_days=rf.Number,
                     line_items=rf.Branch(length=16, sku=rf.Category, quantity=rf.Number)),
    churned=rf.Boolean(mask=True),
    next_month_spend=rf.Number(mask=True),
)''', font_size=19).to_edge(UP, buff=0.5)
        facts = VGroup(
            label(rf"{m['chip']}, {m['memory_gib']:.0f} GB, {m['threads']} CPU threads, PyTorch {m['torch']}",
                  font_size=26),
            label(rf"batches of {d['batch']} customers, about 10 orders of 1 to 6 items each", font_size=24,
                  color=GREY_A),
            label(rf"median of {d['steps'] - d['warmup']} batches after {d['warmup']} warm-up; peak memory of a fresh"
                  r" process", font_size=24, color=GREY_A),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).next_to(c, DOWN, buff=0.55)
        with self.voiceover(
            "What does a relflow model cost? Here's a measurement on the computer that rendered this video: "
            f"<bookmark mark='m'/> an Apple {m['chip'].replace('Apple ', '')} laptop with {m['memory_gib']:.0f} "
            "gigabytes of memory. <bookmark mark='c'/> The model is the customer model from the getting started guide: "
            "orders with line items, and two targets, built at each of the five sizes. <bookmark mark='b'/> Each batch "
            f"is {d['batch']} synthetic customers with about ten orders each. Every number is the median of several "
            "batches after a warm-up, and memory is the peak of a fresh process."
        ) as vo:
            vo.wait_until("m")
            self.play(FadeIn(facts[0]))
            vo.wait_until("c")
            self.play(FadeIn(c))
            vo.wait_until("b")
            self.play(FadeIn(facts[1:], lag_ratio=0.3))
        self.play(FadeOut(VGroup(c, facts)))

    # ------------------------------------------------------------------
    def presets(self):
        cols = [r"preset", r"parameters", r"train, rows/s", r"predict, rows/s", r"peak memory, training"]
        xs = [-6.2, -4.0, -1.0, 2.4, 5.4]
        head = VGroup(*[label(t, font_size=22, color=GREY_A).move_to([x, 2.4, 0]) for t, x in zip(cols, xs)])
        line = Line([-6.8, 2.1, 0], [6.9, 2.1, 0], color=GREY_D, stroke_width=1.5)
        rows = VGroup()
        tmax = max(self.run(p, "train")["rows_per_s"] for p in PRESETS)
        pmax = max(self.run(p, "predict")["rows_per_s"] for p in PRESETS)
        for i, p in enumerate(PRESETS):
            tr, pr = self.run(p, "train"), self.run(p, "predict")
            y = 1.55 - 0.78 * i
            bt = Rectangle(width=2.4 * tr["rows_per_s"] / tmax, height=0.26, stroke_width=0, fill_color=C.EMBED,
                           fill_opacity=0.5).move_to([xs[2] - 1.2, y - 0.24, 0], aligned_edge=LEFT)
            bp = Rectangle(width=2.4 * pr["rows_per_s"] / pmax, height=0.26, stroke_width=0, fill_color=C.RF_TARGET,
                           fill_opacity=0.5).move_to([xs[3] - 1.2, y - 0.24, 0], aligned_edge=LEFT)
            rows.add(VGroup(
                Mono(p, font_size=24, color=WHITE).move_to([xs[0], y, 0]),
                label(params_text(tr["params"]), font_size=26).move_to([xs[1], y, 0]),
                bt, label(f"{tr['rows_per_s']:,.0f}".replace(",", "{,}"), font_size=26).move_to([xs[2], y + 0.06, 0]),
                bp, label(f"{pr['rows_per_s']:,.0f}".replace(",", "{,}"), font_size=26).move_to([xs[3], y + 0.06, 0]),
                label(gib(tr["peak_mib"]), font_size=26).move_to([xs[4], y, 0]),
            ))
        base = np.median([r["baseline_mib"] for r in self.runs])
        foot = label(rf"about {gib(base)} of that is Python, PyTorch and relflow before any model exists",
                     font_size=22, color=GREY_A).to_edge(DOWN, buff=0.45)
        tag = real_tag(r"measured on this computer; CPU only", corner=UR)
        xs_, md_, xl_ = (self.run(p, "train") for p in ("xs", "md", "xl"))
        mdp = self.run("md", "predict")
        ratio = xs_["rows_per_s"] / xl_["rows_per_s"]
        assert xs_["rows_per_s"] > md_["rows_per_s"] > xl_["rows_per_s"]
        with self.voiceover(
            "Here are all five presets. <bookmark mark='x'/> The extra-small one has about "
            f"{xs_['params'] / 1e3:.0f} thousand parameters and trains on {xs_['rows_per_s']:,.0f} customers a second. "
            f"<bookmark mark='m'/> The medium preset, {md_['params'] / 1e6:.0f} million parameters, trains on "
            f"{md_['rows_per_s']:,.0f} customers a second, predicts for {mdp['rows_per_s']:,.0f}, and peaks at "
            f"{gib(md_['peak_mib']).replace('GB', 'gigabytes').replace('MB', 'megabytes')} while training. "
            f"<bookmark mark='l'/> The extra-large one, {xl_['params'] / 1e6:.0f} million parameters, is about "
            f"{ratio:.0f} times slower than extra-small. <bookmark mark='f'/> Note the floor: around "
            f"{base:.0f} megabytes is just Python, PyTorch and relflow, before any model exists."
        ) as vo:
            self.play(FadeIn(head), Create(line), FadeIn(tag))
            vo.wait_until("x")
            self.play(FadeIn(rows[0], shift=RIGHT * 0.1))
            vo.wait_until("m")
            self.play(FadeIn(rows[1:3], shift=RIGHT * 0.1, lag_ratio=0.3))
            vo.wait_until("l")
            self.play(FadeIn(rows[3:], shift=RIGHT * 0.1, lag_ratio=0.3))
            vo.wait_until("f")
            self.play(FadeIn(foot))
        self.play(FadeOut(VGroup(head, line, rows, foot, tag)))

    # ------------------------------------------------------------------
    def devices(self):
        mps = [r for r in self.runs if r["device"] == "mps"]
        if not mps:
            return
        presets = [p for p in PRESETS if any(r["preset"] == p for r in mps)]
        groups = [(p, mode) for p in presets for mode in ("train", "predict")]
        ymax = max(max(self.run(p, m)["rows_per_s"], self.run(p, m, "mps")["rows_per_s"]) for p, m in groups) * 1.15
        pl = Plot((0, len(groups)), (0, ymax), width=10.5, height=4.0, x_ticks=[], y_ticks=[], font_size=20,
                  grid=False).move_to(DOWN * 0.4)
        bars = VGroup()
        for i, (p, m) in enumerate(groups):
            cpu, gpu = self.run(p, m), self.run(p, m, "mps")
            b1 = pl.bar(i + 0.32, cpu["rows_per_s"], 0.34, C.EMBED)
            b2 = pl.bar(i + 0.68, gpu["rows_per_s"], 0.34, C.HASH)
            t1 = label(f"{cpu['rows_per_s']:.0f}", font_size=18).next_to(b1, UP, buff=0.05)
            t2 = label(f"{gpu['rows_per_s']:.0f}", font_size=18, color=GREY_A).next_to(b2, UP, buff=0.05)
            xl = label(rf"\texttt{{{p}}}, {m}", font_size=20, color=GREY_A).next_to(pl.c2p(i + 0.5, 0), DOWN, buff=0.15)
            bars.add(VGroup(b1, b2, t1, t2, xl))
        yt = label(r"rows per second", font_size=22, color=GREY_A).next_to(pl, UP, buff=0.1).align_to(pl, LEFT)
        leg = VGroup(VGroup(Square(0.22, stroke_width=0, fill_color=C.EMBED, fill_opacity=0.85),
                            label(r"CPU", font_size=22)).arrange(RIGHT, buff=0.12),
                     VGroup(Square(0.22, stroke_width=0, fill_color=C.HASH, fill_opacity=0.85),
                            label(r"M1 GPU (MPS)", font_size=22)).arrange(RIGHT, buff=0.12)).arrange(RIGHT, buff=0.5)
        leg.to_corner(UR, buff=0.5)
        cpu_wins = [(p, m) for p, m in groups if self.run(p, m)["rows_per_s"] > self.run(p, m, "mps")["rows_per_s"]]
        md_t = self.run("md", "train")["rows_per_s"] / self.run("md", "train", "mps")["rows_per_s"]
        tip = code('lit.Trainer(accelerator="cpu")', font_size=20).next_to(leg, DOWN, buff=0.35).align_to(leg, RIGHT)
        tag = real_tag(r"same models, same batches", corner=DL)
        gpu_wins = [(p, m) for p, m in groups if (p, m) not in cpu_wins]
        names = {"xs": "extra-small", "sm": "small", "md": "medium", "lg": "large", "xl": "extra-large"}
        if not gpu_wins:
            verdict = ("On every one of them, the CPU wins. For the medium model it trains "
                       f"{md_t:.1f} times faster. These models do many small operations, and on this machine the "
                       "overhead of launching each one on the GPU outweighs its speed.")
        else:
            wins = " and ".join(f"{m}ing the {names[p]} model" for p, m in gpu_wins)
            verdict = (f"The CPU wins {len(cpu_wins)} of the {len(groups)}. The medium model trains {md_t:.1f} times "
                       f"faster on the CPU. The GPU only edges ahead {wins}: it pays off once each operation is big "
                       "enough to cover the cost of launching it.")
        assert len(cpu_wins) >= len(groups) - 1
        with self.voiceover(
            "Lightning picks the GPU automatically when there is one, and on a Mac that's the M1's GPU. Is it "
            "faster? <bookmark mark='b'/> Here are three of the presets, training and predicting, on each. "
            f"<bookmark mark='v'/> {verdict} <bookmark mark='t'/> So on a laptop like this, try "
            "accelerator equals CPU, and measure, rather than assume."
        ) as vo:
            self.play(FadeIn(leg), FadeIn(tag))
            vo.wait_until("b")
            self.play(Create(pl.x_axis), FadeIn(yt))
            self.play(LaggedStart(*[FadeIn(b, shift=UP * 0.1) for b in bars], lag_ratio=0.15), run_time=2.0)
            vo.wait_until("t")
            self.play(FadeIn(tip))
        self.play(FadeOut(VGroup(pl.x_axis, yt, bars, leg, tip, tag)))

    # ------------------------------------------------------------------
    def context(self):
        sweep = sorted((r for r in self.runs if r["preset"] == "sm" and r["mode"] == "train" and r["device"] == "cpu"),
                       key=lambda r: r["length"])
        lengths = [r["length"] for r in sweep]
        ms = [r["ms_per_batch"] for r in sweep]
        mem = [r["peak_mib"] for r in sweep]
        pl = Plot((12, 340), (0, max(ms) * 1.25), width=5.6, height=3.6, log_x=True, x_ticks=lengths,
                  y_ticks=list(np.linspace(0, max(ms) * 1.2, 4).round(-1)), x_fmt=lambda v: f"{int(v)}",
                  y_fmt=lambda v: f"{v:.0f}", font_size=20).move_to(LEFT * 3.5 + DOWN * 0.4)
        pm = Plot((12, 340), (0, max(mem) * 1.25), width=5.6, height=3.6, log_x=True, x_ticks=lengths,
                  y_ticks=list(np.linspace(0, max(mem) * 1.2, 4).round(-2)), x_fmt=lambda v: f"{int(v)}",
                  y_fmt=lambda v: f"{v / 1024:.1f}", font_size=20).move_to(RIGHT * 3.4 + DOWN * 0.4)
        l1, l2 = pl.line(lengths, ms, C.EMBED, 4), pm.line(lengths, mem, C.RF_TARGET, 4)
        d1 = VGroup(*[pl.dot(x, y, color=C.EMBED, radius=0.06) for x, y in zip(lengths, ms)])
        d2 = VGroup(*[pm.dot(x, y, color=C.RF_TARGET, radius=0.06) for x, y in zip(lengths, mem)])
        t1 = label(r"ms per training batch", font_size=22, color=C.EMBED).next_to(pl, UP, buff=0.15)
        t2 = label(r"peak memory, GB", font_size=22, color=C.RF_TARGET).next_to(pm, UP, buff=0.15)
        x1 = pl.x_title(r"room for orders (\texttt{length})", font_size=20)
        x2 = pm.x_title(r"room for orders (\texttt{length})", font_size=20)
        head = label(r"the same customers, with more and more room: \texttt{orders=rf.Branch(length=...)}",
                     font_size=24).to_edge(UP, buff=0.45)
        growth_t, growth_m = ms[-1] / ms[0], mem[-1] / mem[0]
        factor = lengths[-1] // lengths[0]
        assert factor == 16 and growth_t < 4 and growth_m < 4
        tag = real_tag(r"small preset, CPU")
        with self.voiceover(
            "What about context length? <bookmark mark='h'/> Keep the same customers, but give the orders branch more "
            f"and more room: from {lengths[0]} slots up to {lengths[-1]}, each with sixteen item slots inside. "
            f"<bookmark mark='t'/> Sixteen times the room costs {growth_t:.1f} times the time, <bookmark mark='m'/> "
            f"and {growth_m:.1f} times the memory. Attention skips the empty slots, but the batch is still built as "
            "full rectangles before it's packed, so the cost still grows, just far more slowly than the room does. "
            "<bookmark mark='c'/> Padding is cheap, not free: give a branch the room your data needs, not more."
        ) as vo:
            self.play(FadeIn(head), FadeIn(tag))
            vo.wait_until("h")
            self.play(Create(pl), Create(pm), FadeIn(t1), FadeIn(t2), FadeIn(x1), FadeIn(x2))
            vo.wait_until("t")
            self.play(Create(l1), FadeIn(d1))
            vo.wait_until("m")
            self.play(Create(l2), FadeIn(d2))
        self.play(FadeOut(VGroup(head, pl, pm, l1, l2, d1, d2, t1, t2, x1, x2, tag)))
        self.factor, self.growth = factor, (growth_t, growth_m)

    # ------------------------------------------------------------------
    def advice(self):
        rows = VGroup(
            label(r"measure the whole job: reading, preprocessing, encoding, the model, writing", font_size=25),
            label(r"pick a batch size with Lightning's \texttt{BatchSizeFinder}", font_size=25, color=GREY_A),
            label(r'on GPUs that support it: \texttt{precision="bf16-mixed"}', font_size=25, color=GREY_A),
            label(r"\texttt{model.compile()} for encoder and pooling regions; measure it", font_size=25, color=GREY_A),
            label(r"export only the outputs you need", font_size=25, color=GREY_A),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.28)
        note_ = note(r"one laptop and synthetic data: your numbers will differ, so measure your own", font_size=20)
        note_.to_edge(DOWN, buff=0.4)
        with self.voiceover(
            "These are one laptop's numbers on synthetic data, so treat them as a starting point. <bookmark mark='a'/> "
            "For your own workload, time the whole job, from reading the data to writing predictions. "
            "<bookmark mark='b'/> Let Lightning find a batch size, <bookmark mark='c'/> train in bfloat16 on hardware that "
            "supports it, <bookmark mark='d'/> try compiling the encoders, and keep it only if it helps, "
            "<bookmark mark='e'/> and ask only for the outputs you need."
        ) as vo:
            for mark, r in zip("abcde", rows):
                vo.wait_until(mark)
                self.play(FadeIn(r, shift=RIGHT * 0.1), run_time=0.7)
            self.play(FadeIn(note_))
        self.wait(0.5)
        self.play(FadeOut(VGroup(rows, note_)))
