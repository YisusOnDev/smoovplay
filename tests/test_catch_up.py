"""Catch up at 60 fps: when the game's new frames keep arriving while the last in-between frame is still being made
(each pair of ticks only just fitting in the time of two), the pipeline leaves out an in-between frame now and then
instead of staying behind for good. A one-off delay is left to drain by itself. CPU only, stand-ins as in
test_120fps."""
import time

import torch

from framegen.io import TICK, Tick
from framegen.pipeline import CATCH_UP_MS, Pipeline
from framegen.settings import Settings
from test_120fps import AI, _Interp, _pipeline, _Sink


def _busy(ms):
    end = time.perf_counter() + ms / 1000
    while time.perf_counter() < end:
        pass


class _SlowInterp(_Interp):
    """An in-between frame takes `ms` milliseconds and every tick `tick_ms` (busy, like a graphics card being waited
    for, and the work every frame gets)."""

    def __init__(self, ms, tick_ms=0.0):
        super().__init__()
        self.ms, self.tick_ms = ms, tick_ms

    def load(self, frame):
        _busy(self.tick_ms)
        super().load(frame)

    def interpolate(self, slot, t=0.5):
        _busy(self.ms)
        return super().interpolate(slot, t)


def _paced_game(n):
    """A 30 fps game on a 60 Hz stream, each tick stamped when it was due (so a slow pipeline falls behind it)."""
    t0 = time.perf_counter()
    for i in range(n):
        due = t0 + i * TICK
        while time.perf_counter() < due:
            pass
        yield Tick(torch.full((8, 8, 3), (i // 2) % 150, dtype=torch.uint8), due, i, new_hint=(i % 2 == 0))


def _run(it, ticks):
    sink = _Sink()
    st = _pipeline(it).run(ticks, sink, label="RIFE", settings=Settings({"hz": 60, "even_out": False,
                                                                          "overlay": False, "audio_sync": False}))
    return st, sink


def test_a_pipeline_that_cant_keep_up_catches_up_instead_of_falling_further_behind():
    # 36 ms of work for each 33.3 ms pair (as DirectML's 21 + 7 + 7 ms on a slow moment): without help the wait
    # would grow by about 3 ms every pair, for good
    st, sink = _run(_SlowInterp(22.0, tick_ms=7.0), _paced_game(240))
    assert st.caught_up > 0
    late = sorted(st.queue_ms[-60:])
    assert late[-1] < CATCH_UP_MS + 2000 * TICK  # it doesn't run away
    assert sink.shown.count(AI + 2) > 3 * st.caught_up  # it still makes most of its in-between frames


def test_a_pipeline_that_only_just_keeps_up_is_left_alone():
    # 31 ms for each pair: the in-between frame's tick overruns, and the next one makes it up
    st, sink = _run(_SlowInterp(31.0), _paced_game(180))
    assert st.caught_up == 0


def test_a_fast_pipeline_never_leaves_frames_out():
    st, sink = _run(_SlowInterp(2.0), _paced_game(120))
    assert st.caught_up == 0
    assert sink.shown.count(AI + 2) >= 55  # an in-between frame for (nearly) every new one


def test_catch_up_line_in_the_results():
    st, _ = _run(_SlowInterp(22.0, tick_ms=7.0), _paced_game(180))
    assert "left out to catch up" in st.summary()
