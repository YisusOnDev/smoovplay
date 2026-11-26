"""Even 60 fps: with a model that takes most of a tick, the in-between frame goes out late and the real frame used to
follow on its repeat tick a few ms later, so the in-between frame only flashed (5.5 ms against 27.7 in a real
DirectML session). With the setting on, the real frame is held for the start of the next model call, so both stay
up for about a tick, without sleeping, without the queue growing, and with catch-up still working. Off changes
nothing. CPU only, stand-ins as in test_catch_up; gaps are between present calls, as a sink sees them."""
import time

import numpy as np
import torch

from framegen.io import TICK, Tick, Waiting
from framegen.pipeline import CATCH_UP_MS
from framegen.settings import BY_KEY, Settings
from test_120fps import AI, _pipeline, _Sink
from test_catch_up import _busy, _paced_game, _SlowInterp


class _TimedSink(_Sink):
    """A window whose present takes `ms` (upload, draw and a swap that doesn't wait, about 1.2 ms in the real
    session), stamping when each frame went out."""

    def __init__(self, ms=1.2):
        super().__init__()
        self.ms, self.at = ms, []

    def write(self, frame, text=None):
        _busy(self.ms)
        super().write(frame, text)
        self.at.append(time.perf_counter())

    def wait(self, w):
        self.shown.append("wait")
        self.at.append(time.perf_counter())


def _run(it, ticks, even, sink=None):
    sink = sink or _TimedSink()
    st = _pipeline(it).run(ticks, sink, label="RIFE", settings=Settings({
        "hz": 60, "even_out": False, "overlay": False, "audio_sync": False, "even_60": even}))
    return st, sink


def _gaps(sink, skip=20):
    """(how long each in-between frame was up, how long each real frame was up) in ms, from present to present,
    leaving out the first `skip` frames while it settles."""
    mid, real = [], []
    for k in range(skip, len(sink.shown) - 1):
        ms = 1000 * (sink.at[k + 1] - sink.at[k])
        (mid if sink.shown[k] == AI + 2 else real).append(ms)
    return mid, real


def test_off_by_default_and_says_it_costs_delay():
    spec = BY_KEY["even_60"]
    assert spec.default is False and Settings().even_60 is False
    assert "delay" in spec.help and "delay" in spec.impact["speed"]


def test_on_the_in_between_and_real_frames_stay_up_about_as_long():
    # As the real machine: 15 ms of model and 4 ms of everything else a tick, 1.2 ms to present
    st, sink = _run(_SlowInterp(15.0, tick_ms=4.0), _paced_game(240), even=True)
    mid, real = _gaps(sink)
    m, r = np.median(mid), np.median(real)
    assert abs(m - 1000 * TICK) < 3.0 and abs(r - 1000 * TICK) < 3.0, (m, r)
    assert 0.8 < m / r < 1.25, (m, r)
    assert st.even_held > 90 and st.caught_up == 0
    assert abs(np.median(st.even_gap_ms) - m) < 1.0  # the log's figure is the same gap
    assert "even 60 fps:" in st.summary()


def test_off_nothing_changes_the_in_between_frame_still_flashes():
    st, sink = _run(_SlowInterp(15.0, tick_ms=4.0), _paced_game(240), even=False)
    mid, real = _gaps(sink)
    assert np.median(mid) < 9.0 and np.median(real) > 24.0  # the short-long beat the setting is for
    assert st.even_held == 0 and not st.even_gap_ms and "even 60 fps" not in st.summary()


def test_the_same_frames_in_the_same_order_and_the_delay_it_costs():
    on, s_on = _run(_SlowInterp(15.0, tick_ms=4.0), _paced_game(180), even=True)
    off, s_off = _run(_SlowInterp(15.0, tick_ms=4.0), _paced_game(180), even=False)
    assert s_on.shown == s_off.shown
    assert len(on.real_delay_ms) == len(off.real_delay_ms)  # every real frame counted once, when it really went out
    cost = np.median(on.real_delay_ms) - np.median(off.real_delay_ms)
    assert 6.0 < cost < 16.0, cost  # about 12 ms: a tick, less the few ms the real frame used to follow its AI one


def test_on_no_sleeping_and_the_queue_doesnt_grow():
    # 22 ms ticks with the model and 7 ms ticks without: 29 of the pair's 33.3 ms, plus the two presents
    st, sink = _run(_SlowInterp(15.0, tick_ms=7.0), _paced_game(240), even=True)
    assert st.even_held > 60 and st.caught_up == 0
    # No slow climb: the end waits about as long as the start (medians and p90, so one hiccup of a busy PC doesn't
    # count), and never a tick
    early, late = st.queue_ms[20:80], st.queue_ms[-60:]
    assert np.median(late) < np.median(early) + 3.0 and np.percentile(late, 90) < np.percentile(early, 90) + 5.0
    assert np.percentile(late, 90) < 1000 * TICK


def test_on_a_pipeline_that_cant_keep_up_still_catches_up():
    # test_catch_up's 36 ms of work for each 33.3 ms pair (38 here, with the presents' 1.2 ms)
    st, sink = _run(_SlowInterp(22.0, tick_ms=7.0), _paced_game(240), even=True)
    off, s_off = _run(_SlowInterp(22.0, tick_ms=7.0), _paced_game(240), even=False)
    assert st.caught_up > 0
    assert max(st.queue_ms[-60:]) < CATCH_UP_MS + 2000 * TICK  # it doesn't run away
    assert st.caught_up <= off.caught_up + 5  # (no more left out than with the setting off)
    assert sink.shown.count(AI + 2) >= s_off.shown.count(AI + 2) - 5


def test_a_fast_model_is_left_alone():
    # A 3 ms model: the next model call comes too soon after the frames arrive, so holding would make it worse
    st, sink = _run(_SlowInterp(3.0, tick_ms=2.0), _paced_game(180), even=True)
    assert st.even_held == 0


def test_a_held_frame_goes_out_before_a_wait_and_at_the_end():
    def stalls():
        yield from _paced_game(42)  # ends on a new frame's repeat tick: held
        yield Waiting(0.5, "test", "")
        t = time.perf_counter()
        yield Tick(torch.full((8, 8, 3), 99, dtype=torch.uint8), t, 41, new_hint=True)
        yield Tick(torch.full((8, 8, 3), 99, dtype=torch.uint8), t + TICK, 42, new_hint=False)

    st, sink = _run(_SlowInterp(15.0, tick_ms=4.0), stalls(), even=True)
    k = sink.shown.index("wait")
    assert sink.shown[k - 1] == 20 and sink.shown[k - 2] == AI + 2  # the real frame, then the wait
    assert sink.shown[-1] == 99  # the source ended with a frame held: it still went out
    assert st.exit == "end"
