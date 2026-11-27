"""Shortcuts for App window mode (Experimental). SmoovPlay's picture sits over the app as a click-through layer that
never takes the keyboard: the app keeps it, as some apps stop drawing without it. So the video window's keys come
through Windows' system-wide shortcuts (RegisterHotKey) instead, Ctrl+Alt with the usual key by default:

    Ctrl+Alt+M  menu        Ctrl+Alt+F  fullscreen      Ctrl+Alt+Esc  quit
    Ctrl+Alt+S  stats       Ctrl+Alt+I  smooth on/off

A shortcut another program already has goes to the next one in its list (Ctrl+Alt+Space for the menu), and the video
and the launcher say which keys SmoovPlay got. People can pick their own on the launcher's Play page, which passes
them as `run.py --window-keys "menu=Ctrl+Alt+K;quit=Ctrl+Alt+End"`.

While the menu is open, its own keys (arrows, Enter, Esc, Backspace) are taken too, without Ctrl+Alt, and given back
when it closes. Windows treats AltGr as Ctrl+Alt, so while SmoovPlay plays in this mode, AltGr with these keys types
nothing (AltGr+M is µ on a German keyboard), which is why only five keys are taken.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass

MODS = {"ctrl": 0x0002, "alt": 0x0001, "shift": 0x0004, "win": 0x0008}  # MOD_CONTROL, MOD_ALT, MOD_SHIFT, MOD_WIN
MOD_ORDER = ("Ctrl", "Alt", "Shift", "Win")
NOREPEAT = 0x4000  # MOD_NOREPEAT: holding the keys down fires once
WM_HOTKEY = 0x0312
WM_QUIT = 0x0012

# Key names (as written and shown) -> Windows' virtual-key codes
KEYS = {"Esc": 0x1B, "Space": 0x20, "Enter": 0x0D, "Backspace": 0x08, "Tab": 0x09, "End": 0x23, "Home": 0x24, "PageUp": 0x21,
        "PageDown": 0x22, "Insert": 0x2D, "Delete": 0x2E, "Pause": 0x13, "Left": 0x25, "Up": 0x26, "Right": 0x27,
        "Down": 0x28}
KEYS |= {chr(c): c for c in range(ord("A"), ord("Z") + 1)}
KEYS |= {str(d): 0x30 + d for d in range(10)}
KEYS |= {f"F{n}": 0x6F + n for n in range(1, 25)}
_ALIASES = {"escape": "Esc", "return": "Enter", "pgup": "PageUp", "pgdn": "PageDown", "del": "Delete", "ins": "Insert",
            "control": "Ctrl", "windows": "Win"}

# action: (what it does, in words; the video window's own key it stands for; its shortcuts, first free one wins)
ACTIONS = {
    "menu": ("Menu", "M", ("Ctrl+Alt+M", "Ctrl+Alt+Space")),
    "quit": ("Quit", "Q", ("Ctrl+Alt+Esc", "Ctrl+Alt+End")),
    "fullscreen": ("Fullscreen", "F", ("Ctrl+Alt+F",)),
    "stats": ("Stats overlay", "S", ("Ctrl+Alt+S",)),
    "smooth": ("Smooth on/off", "I", ("Ctrl+Alt+I",)),
}
NAV = ("Up", "Down", "Left", "Right", "Enter", "Esc", "Backspace")  # the menu's keys, plain, while it's open

# Windows' own, or keys people need to get out of things: never taken
RESERVED = {"Ctrl+Alt+Delete", "Ctrl+Shift+Esc", "Alt+Tab", "Alt+Esc", "Alt+F4", "Ctrl+Alt+Tab", "Win+L", "Win+D",
            "Win+Tab", "Ctrl+Esc", "Win"}


@dataclass(frozen=True)
class Combo:
    mods: int  # MODS flags
    vk: int
    name: str  # as shown: "Ctrl+Alt+M"


def parse(text: str) -> Combo:
    """A shortcut from its name ("Ctrl+Alt+M", "ctrl + alt + escape"), or ValueError saying what's wrong with it. It
    needs Ctrl, Alt or Win: a key on its own, or with Shift only, would be taken from the app."""
    parts = [p.strip() for p in text.replace(" ", "").split("+") if p.strip()]
    if not parts:
        raise ValueError("no keys")
    mods, names, key = 0, [], None
    for p in parts:
        word = _ALIASES.get(p.lower(), p)
        if word.lower() in MODS:
            flag = MODS[word.lower()]
            if mods & flag:
                raise ValueError(f"{word} twice")
            mods |= flag
            continue
        if key is not None:
            raise ValueError("more than one key besides Ctrl, Alt, Shift and Win")
        key = next((k for k in KEYS if k.lower() == word.lower()), None)
        if key is None:
            raise ValueError(f"unknown key {p!r}")
    if key is None:
        raise ValueError("Ctrl, Alt, Shift or Win needs a key with it")
    if not mods & (MODS["ctrl"] | MODS["alt"] | MODS["win"]):
        raise ValueError("it needs Ctrl, Alt or Win, or the app would lose that key")
    names = [m for m in MOD_ORDER if mods & MODS[m.lower()]]
    name = "+".join(names + [key])
    if name in RESERVED:
        raise ValueError(f"{name} belongs to Windows")
    return Combo(mods, KEYS[key], name)


def parse_arg(text: str | None) -> dict[str, str]:
    """--window-keys "menu=Ctrl+Alt+K;quit=Ctrl+Alt+End": the shortcuts people picked, checked. ValueError if one
    can't be used, or two actions share one."""
    out: dict[str, str] = {}
    for item in (text or "").split(";"):
        if not item.strip():
            continue
        action, _, combo = item.partition("=")
        action = action.strip().lower()
        if action not in ACTIONS:
            raise ValueError(f"no action {action!r} (there are {', '.join(ACTIONS)})")
        try:
            out[action] = parse(combo).name
        except ValueError as e:
            raise ValueError(f"{action}: {combo.strip()!r}: {e}") from None
    names = list(out.values())
    if len(set(names)) != len(names):
        raise ValueError("two actions have the same shortcut")
    return out


def choices(picked: dict[str, str] | None = None) -> dict[str, tuple[str, ...]]:
    """Each action's shortcuts to try, in order: the one picked for it, or its defaults (minus any picked for another
    action)."""
    picked = picked or {}
    taken = set(picked.values())
    return {a: (picked[a],) if a in picked else tuple(c for c in d if c not in taken)
            for a, (_, _, d) in ACTIONS.items()}


def describe(got: dict[str, str | None], sep: str = "\n") -> str:
    """The keys, menu first, one a line by default ("Ctrl+Alt+M: menu", "Ctrl+Alt+F: fullscreen"...), in plain
    ASCII: the video window's font has nothing else."""
    order = ("menu", "fullscreen", "quit", "stats", "smooth")
    return sep.join(f"{got[a]}: {ACTIONS[a][0].lower()}" for a in order if got.get(a))


class Hotkeys:
    """The shortcuts, registered for the calling thread (the one that draws the video and polls its messages)."""

    def __init__(self, picked: dict[str, str] | None = None, user32=None) -> None:
        self.u32 = user32 if user32 is not None else _user32()
        self.got: dict[str, str | None] = {}  # action -> the shortcut it got (None: all of its shortcuts were taken)
        self.taken: list[str] = []  # shortcuts another program has
        self._ids: dict[int, str] = {}  # registration id -> action, or "nav:<key>"
        self.nav_on = False
        for i, (action, combos) in enumerate(choices(picked).items(), start=1):
            self.got[action] = None
            for name in combos:
                c = parse(name)
                if self.u32.RegisterHotKey(None, i, c.mods | NOREPEAT, c.vk):
                    self.got[action], self._ids[i] = name, action
                    break
                self.taken.append(name)

    def set_nav(self, on: bool) -> None:
        """The menu's keys, taken while it's open (with key repeat, for holding an arrow down)."""
        if on == self.nav_on:
            return
        self.nav_on = on
        for j, key in enumerate(NAV, start=100):
            if on:
                if self.u32.RegisterHotKey(None, j, 0, KEYS[key]):
                    self._ids[j] = "nav:" + key
            elif self._ids.pop(j, None) is not None:
                self.u32.UnregisterHotKey(None, j)

    def poll(self) -> list[str]:
        """What was pressed since the last look: actions ("menu") and menu keys ("nav:Up"), in order. Run it before
        GLFW looks at the messages: these come to the thread, not a window, and GLFW would drop them.

        It takes every message waiting, not only WM_HOTKEY, and hands the rest to their windows as GLFW does
        (glfw.poll_events then finds the queue empty). Looking for WM_HOTKEY alone (PeekMessage filtered to it) makes
        Windows call the thread hung after 5 s, however often GLFW looks at the messages after it: Windows then puts
        a frozen copy of the video window on screen, which window capture reads back as a black picture, and offers
        to close SmoovPlay ("Python is not responding" after 10 to 20 s, or on a click elsewhere).
        """
        import ctypes
        from ctypes import wintypes as wt
        out, msg = [], wt.MSG()
        while self.u32.PeekMessageW(ctypes.byref(msg), None, 0, 0, 1):  # every message, PM_REMOVE
            if msg.message == WM_HOTKEY and not msg.hWnd:
                what = self._ids.get(int(msg.wParam))
                if what:
                    out.append(what)
            elif msg.message == WM_QUIT:  # GLFW's to act on: put back, and leave the rest to it
                self.u32.PostQuitMessage(int(msg.wParam))
                break
            else:
                self.u32.TranslateMessage(ctypes.byref(msg))
                self.u32.DispatchMessageW(ctypes.byref(msg))
        return out

    def close(self) -> None:
        for i in list(self._ids):
            self.u32.UnregisterHotKey(None, i)
        self._ids.clear()
        self.nav_on = False

    def report(self) -> str:
        """The line for the launcher: "[window-keys] {json}" with what each action got and what was taken."""
        import json
        return "[window-keys] " + json.dumps({"keys": self.got, "taken": self.taken})


def _user32():
    if sys.platform != "win32":
        raise OSError("system-wide shortcuts need Windows")
    import ctypes
    from ctypes import wintypes as wt
    u32 = ctypes.WinDLL("user32", use_last_error=True)
    u32.RegisterHotKey.argtypes = [wt.HWND, ctypes.c_int, wt.UINT, wt.UINT]
    u32.UnregisterHotKey.argtypes = [wt.HWND, ctypes.c_int]
    u32.PeekMessageW.argtypes = [ctypes.POINTER(wt.MSG), wt.HWND, wt.UINT, wt.UINT, wt.UINT]
    u32.TranslateMessage.argtypes = [ctypes.POINTER(wt.MSG)]
    u32.DispatchMessageW.argtypes = [ctypes.POINTER(wt.MSG)]
    u32.DispatchMessageW.restype = ctypes.c_ssize_t
    u32.PostQuitMessage.argtypes = [ctypes.c_int]
    return u32
