"""End cards for the standalone episodes: the series, what comes next, and where the docs are.

``End01`` ... ``End21`` are generated from the manifest; ``tools/build.py relflow --episodes`` appends each
episode's card after its scene.  They are silent: the music bed lifts over them."""

from __future__ import annotations

from explainer import *  # noqa: F403
from videos.relflow.common import label, logo, wordmark
from videos.relflow.manifest import DOCS, EPISODES


class EndCard(Scene):
    number = 1

    def construct(self):
        e = EPISODES[self.number - 1]
        nxt = EPISODES[self.number] if self.number < len(EPISODES) else None
        mark = VGroup(logo(1.1), wordmark(font_size=64)).arrange(RIGHT, buff=0.35)
        series = label(rf"visualized \;$\cdot$\; episode {e['number']} of {len(EPISODES)}", font_size=28,
                       color=C.DIM)
        rows = [mark, series]
        if nxt:
            rows.append(label(rf"Next: {nxt['title']}", font_size=40))
        else:
            rows.append(label(r"the docs, the code and the proofs:", font_size=36))
        rows.append(label(rf"\texttt{{{DOCS}}}", font_size=28, color=GREY_A))
        card = VGroup(*rows).arrange(DOWN, buff=0.4)
        card[2].shift(DOWN * 0.15)
        card[3].shift(DOWN * 0.15)
        self.play(FadeIn(card, shift=UP * 0.2), run_time=0.8)
        self.wait(3.6)
        self.play(FadeOut(card), run_time=0.8)


for _e in EPISODES:
    _name = f"End{_e['number']:02d}"
    globals()[_name] = type(_name, (EndCard,), {"number": _e["number"], "__module__": __name__})
