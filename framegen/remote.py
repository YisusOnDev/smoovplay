"""The launcher's live link to a running session (run.py --control), so its Settings page works while you play.

The launcher writes one command per line to our stdin:

    set <key> <json value>    change a setting and save it, as if it were picked in the video window's menu
    restart                   apply restart-only settings now (the menu's "Restart now")
    quit                      stop, as Esc in the video window does
    state                   print the current state again

and reads a `[settings-state] {json}` line on our stdout at the start of every session and after every change,
whoever made it (this link, the menu or a hotkey): the current values, which choices can be picked right now,
rows that are unavailable and why, and the restart-only changes still waiting. Commands are read on a thread but
applied on the window's thread (poll()), the thread the menu's keys run on, so they act exactly like menu changes.
"""
from __future__ import annotations

import json
import queue
import sys
import threading
from typing import Callable, TextIO

from .settings import BY_KEY, SPECS, Settings

PREFIX = "[settings-state] "


class Remote:
    """allowed(key, value) and disabled(key) are the window's (the same rules as its menu); restart_reasons() lists
    the labels of restart-only settings changed since the session started."""

    def __init__(self, settings: Settings, allowed: Callable[[str, object], bool],
                 disabled: Callable[[str], str | None], restart_reasons: Callable[[], list[str]],
                 commands: TextIO | None = None, out: TextIO | None = None) -> None:
        self.settings, self.allowed, self.disabled, self.restart_reasons = settings, allowed, disabled, restart_reasons
        self._out = out
        self._q: queue.Queue[str] = queue.Queue()
        threading.Thread(target=self._read, args=(commands or sys.stdin,), name="remote", daemon=True).start()
        settings.watch(lambda key: self.publish())

    def _read(self, commands: TextIO) -> None:
        for line in commands:  # ends when the launcher closes our stdin, or exits
            line = line.strip().lstrip("﻿")  # in case a writer starts with a byte-order mark
            if line:
                self._q.put(line)

    def poll(self) -> str | None:
        """Apply the commands that arrived since the last call (on the window's thread). Returns "quit" or "restart"
        if one asked for it (quit first)."""
        out = None
        while True:
            try:
                line = self._q.get_nowait()
            except queue.Empty:
                return out
            cmd, _, rest = line.partition(" ")
            if cmd == "quit":
                out = "quit"
            elif cmd == "restart":
                out = out or "restart"
            elif cmd == "state":
                self.publish()
            elif cmd == "set":
                key, _, text = rest.partition(" ")
                self._set(key, text)
            else:
                print(f"[remote] unknown command: {line[:80]}")

    def _set(self, key: str, text: str) -> None:
        spec = BY_KEY.get(key)
        try:
            value = json.loads(text)
        except ValueError:
            value = None
        why = ("unknown setting" if spec is None else
               f"{text[:40]!r} isn't one of its choices" if value not in spec.choices else
               self.disabled(key) or (None if self.allowed(key, value) else "not available right now"))
        if why:
            print(f"[remote] can't set {key}: {why}")
            self.publish()  # so the launcher shows the real value again
            return
        value = spec.choices[spec.choices.index(value)]  # the spec's own type (JSON 1 -> 1.0)
        if value == getattr(self.settings, key):
            self.publish()  # nothing changes, so no watcher would answer
        else:
            self.settings.set(key, value)  # saved; the watchers apply it (window, pipeline) and publish()

    def state(self) -> dict:
        return {
            "values": self.settings.values(),
            "choices": {s.key: [v for v in s.choices if self.allowed(s.key, v)] for s in SPECS},
            "disabled": {s.key: why for s in SPECS if (why := self.disabled(s.key))},
            "restart": list(self.restart_reasons()),
        }

    def publish(self) -> None:
        out = self._out or sys.stdout
        out.write(PREFIX + json.dumps(self.state()) + "\n")  # one write, so other threads' lines don't split it
        out.flush()
