"""Even pacing (settings.even_pacing): every frame on screen at its ideal time plus a fixed delay D.

Without it, each frame is shown as soon as it's made. At 60 fps that's uneven: the in-between frame is ready late
(the model's time after its new frame arrived) and the real frame follows straight after, on the next tick, which
costs about a millisecond. So the in-between frame is up for a few ms and the real one for about 30: the frame
counter says 60, the eye sees a limp. Veyra shows its frames on a fixed schedule instead; this does the same.

The ideal time of a frame is its tick's time on an even clock fitted to the stream (framegen/vcam.py StreamClock:
network and capture jitter don't move it) plus its place within the tick (pipeline.py, present's `at`). D is how
long after its ideal time a frame is usually ready: the 95th percentile over the last few seconds, plus a margin,
so nearly every frame is held until ideal + D and they go out evenly. The cost is that delay: about 10-15 ms more
for the real frames, which used to go out the moment their (cheap) tick was processed.

The pipeline does one thing at a time, so holding a frame also holds back the next tick's work. A frame is never
held past the next tick's ideal time (`cap`): then the next tick starts as soon as it can, and a slow model (an
in-between frame taking over a tick, 16.7 ms) can't push the frames later and later. With such a model the
spacing isn't perfectly even, but much closer than without (e.g. 14/19 ms instead of 3/30).
"""
from __future__ import annotations

import time
from collections import deque

import numpy as np

from .io import TICK


def wait_until(t: float) -> None:
    """Sleep until perf_counter() reaches t: a timer sleep to about a millisecond before, then a short spin."""
    rest = t - time.perf_counter()
    if rest > 0.0015:
        time.sleep(rest - 0.001)
    while time.perf_counter() < t:
        pass


class EvenPacer:
    WINDOW = 240  # frames D is measured over (2-4 s)
    EVERY = 30  # frames between updates of D
    MARGIN = 0.0015  # added to the 95th percentile
    MIN_D, MAX_D = 0.004, 2 * TICK
    START_D = TICK  # until there's enough to measure
    SHRINK = 0.0005  # D goes down by at most this an update (up at once): a calm stretch doesn't undo it.
    # The first measurement (after a start, or a wait for frames) is taken as it is.
    LATE = 0.002  # a frame ready this far after ideal + D counts as late

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.restart()
        self.frames = self.late = self.capped = 0
        self.held_ms: list[float] = []
        self.d_ms: list[float] = []

    def restart(self) -> None:
        """D measured afresh (turned on again, or frames coming again after a wait); the counts stay."""
        self.d = self.START_D
        self._ready: deque[float] = deque(maxlen=self.WINDOW)
        self._n = 0
        self._measured = False

    def due(self, ideal: float, now: float, cap: float) -> float:
        """When to show a frame whose ideal time is `ideal` and that is ready `now`: ideal + D, not after `cap`
        (the next tick's ideal time), and never before now."""
        self._ready.append(now - ideal)
        self.frames += 1
        self._n += 1
        if self._n % self.EVERY == 0 and len(self._ready) >= 2 * self.EVERY:
            want = float(np.percentile(self._ready, 95)) + self.MARGIN
            want = min(max(want, self.MIN_D), self.MAX_D)
            self.d = want if want > self.d or not self._measured else max(want, self.d - self.SHRINK)
            self._measured = True
        self.d_ms.append(self.d * 1000)
        target = ideal + self.d
        if target > cap:
            target = cap
            self.capped += 1
        if now - (ideal + self.d) > self.LATE:
            self.late += 1
        if target <= now:
            self.held_ms.append(0.0)
            return now
        self.held_ms.append((target - now) * 1000)
        return target

    def summary(self) -> str:
        if not self.frames:
            return ""
        d = np.array(self.d_ms)
        line = (f"even pacing: frames shown {np.median(d):.1f} ms after their even time (D; p95 {np.percentile(d, 95):.1f}),"
                f" held p50 {np.median(self.held_ms):.1f} ms; {100 * self.late / self.frames:.1f}% ready later than D")
        if self.capped:
            line += (f"; {100 * self.capped / self.frames:.1f}% shown before D so the next frame isn't held up "
                     "(in-between frames taking over 16.7 ms)")
        return line
