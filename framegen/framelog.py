"""The per-frame CSV log (frames-<session>.csv), written by a thread of its own.

The pipeline's loop only hands each row over: a file write that waits on the disk (antivirus, a busy drive, the
OS flushing its cache) must never hold up the frame being shown. If the writer falls far behind, rows are left out
and counted (`lost`) rather than queued without end.
"""
from __future__ import annotations

import csv
import queue
import threading


class FrameLog:
    def __init__(self, path: str, header: list[str], maxsize: int = 4096) -> None:
        self.lost = 0
        self._q: queue.Queue = queue.Queue(maxsize)
        self._f = open(path, "w", newline="")
        self._w = csv.writer(self._f)
        self._w.writerow(header)
        self._thread = threading.Thread(target=self._run, name="frame log", daemon=True)
        self._thread.start()

    def row(self, values: list) -> None:
        try:
            self._q.put_nowait(values)
        except queue.Full:
            self.lost += 1

    def _run(self) -> None:
        while (values := self._q.get()) is not None:
            self._w.writerow(values)

    def close(self) -> None:
        self._q.put(None)
        self._thread.join(timeout=5)
        self._f.close()
