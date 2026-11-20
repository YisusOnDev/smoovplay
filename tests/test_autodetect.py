"""Auto-detect's choices (framegen.autodetect.recommend) for made-up PCs. No GPU needed."""
import re

from framegen.autodetect import Display, recommend, vfg_offer
from framegen.settings import BY_KEY, SPECS

DEFAULTS = {s.key: s.default for s in SPECS}


def timing(full: float, half: float, full_engine: str = "trt", estimated: bool = False) -> dict:
    return {"full": {"ms": full, "engine": full_engine, "estimated": estimated},
            "half": {"ms": half, "engine": "torch", "estimated": False}}


THIS_PC = timing(9.3, 6.8)  # RTX 4070 Ti SUPER, what the test measures there
UW120 = Display(3440, 1440, 120, 120)


def pick(t=THIS_PC, display=UW120, current=None, **kw) -> dict:
    r = recommend({**DEFAULTS, **(current or {})}, t, display, trt=True, **kw)
    for k, v in r["settings"].items():
        assert v in BY_KEY[k].choices
        assert r["reasons"][k]
    return r


def test_tested_card_gets_the_mode_it_was_tuned_for():
    r = pick()
    assert r["settings"] == {"hz": 90, "motion": "auto", "engine": "auto", "scaler": "fsr"}
    assert "120 fps would only fit with half motion detail" in r["reasons"]["hz"]


def test_display_caps_the_frame_rate():
    assert pick(display=Display(2560, 1440, 60, 60))["settings"]["hz"] == 60
    assert pick(display=Display(2560, 1440, 100, 100))["settings"]["hz"] == 90
    assert pick(t=timing(5.0, 3.5), display=Display(2560, 1440, 119, 119))["settings"]["hz"] == 120  # 119.88 Hz


def test_display_set_below_what_it_can_do_gets_a_note():
    r = pick(display=Display(2560, 1440, 60, 144))
    assert r["settings"]["hz"] == 60
    assert any("can run at 144 Hz" in n for n in r["notes"])


def test_fast_card_keeps_full_detail_at_120():
    r = pick(t=timing(5.0, 3.5), display=Display(2560, 1440, 144, 144))
    assert r["settings"]["hz"] == 120 and r["settings"]["motion"] == "full"
    assert any("G-Sync" in n for n in r["notes"])  # 120 into 144 Hz


def test_slower_cards_step_down():
    mid = pick(t=timing(12.0, 8.5), display=Display(2560, 1440, 144, 144))["settings"]
    assert (mid["hz"], mid["motion"]) == (90, "half")
    slow = pick(t=timing(20.0, 13.0), display=Display(2560, 1440, 144, 144))
    assert (slow["settings"]["hz"], slow["settings"]["motion"]) == (60, "half")
    too_slow = pick(t=timing(40.0, 30.0))
    assert too_slow["settings"]["hz"] == 60 and "Even 60 fps is tight" in too_slow["reasons"]["hz"]


def test_laptop_leaves_more_room():
    t = timing(8.9, 6.6)  # fits 90 fps on a desktop card, not with a laptop's margin
    assert pick(t=t)["settings"]["hz"] == 90
    r = pick(t=t, gpu="NVIDIA GeForce RTX 4080 Laptop GPU")
    assert (r["settings"]["hz"], r["settings"]["motion"]) == (90, "half")
    assert any("Laptop" in n for n in r["notes"])


def test_rtx_vsr_kept_only_where_it_fits():
    r = pick(current={"scaler": "vsr"})  # 90 fps leaves ~0.7 ms: not enough (in testing it cost a quarter of the frames)
    assert r["settings"]["scaler"] == "fsr" and "vsr_quality" not in r["settings"]
    r = pick(current={"scaler": "vsr"}, display=Display(2560, 1440, 60, 60))
    assert r["settings"]["scaler"] == "vsr" and r["settings"]["vsr_quality"] == 4
    r = pick(current={"scaler": "vsr"}, display=Display(2560, 1440, 60, 60), vsr="RTX Video SDK not set up")
    assert r["settings"]["scaler"] == "fsr" and "isn't set up" in r["reasons"]["scaler"]


def test_fsr_is_never_swapped_for_vsr():
    assert pick(t=timing(3.0, 2.0), display=Display(2560, 1440, 60, 60))["settings"]["scaler"] == "fsr"


def test_without_tensorrt_or_display():
    r = recommend(DEFAULTS, timing(10.4, 6.8, "torch"), None, trt=False)
    assert r["settings"]["hz"] == 60 and r["settings"]["engine"] == "auto"
    assert "TensorRT isn't installed" in r["reasons"]["engine"]
    assert any("assumes 60 Hz" in n for n in r["notes"])
    r = pick(t=timing(9.3, 6.8, estimated=True))
    assert any("estimated" in n for n in r["notes"])


def test_busy_card_note():
    assert any("busy" in n for n in pick(busy=85)["notes"])
    assert not any("busy" in n for n in pick(busy=3)["notes"])


def test_rtx_40_and_50_cards_without_vfg_are_offered_it():
    text = vfg_offer(True, "NVIDIA GeForce RTX 5070 Ti", "621.24")
    assert text.startswith("Your RTX 5070 Ti can also run NVIDIA VFG")
    assert "NVIDIA's licence" in text and "Auto-detect times it" in text
    assert "driver" not in text  # new enough
    assert not re.search(r"\d ?ms\b", text) and "faster" not in text  # NVIDIA's licence: no speed or quality claims


def test_vfg_offer_asks_for_a_newer_driver_first():
    assert "570.65 or newer, and this PC has 566.36" in vfg_offer(True, "NVIDIA GeForce RTX 4070", "566.36")
    assert "driver" not in vfg_offer(True, "NVIDIA GeForce RTX 4070", "570.65")
    assert "driver" not in vfg_offer(True, "NVIDIA GeForce RTX 4070", None)  # unknown: VFG itself will say
    assert "driver" not in vfg_offer(True, "NVIDIA GeForce RTX 4070", "r570")


def test_no_vfg_offer_when_set_up_or_the_card_cant_run_it():
    # not_set_up is False when it's set up, on RTX 20/30 cards, without an NVIDIA card, or when the check failed
    assert vfg_offer(False, "NVIDIA GeForce RTX 3080", "621.24") == ""
    assert vfg_offer(False, "NVIDIA GeForce RTX 5070 Ti", "621.24") == ""


def _detect_on(monkeypatch, vfg_why: str, driver: str = "621.24") -> dict:
    """detect() on a made-up PC (no GPU used): an RTX 4070 whose VFG state is vfg_why."""
    from types import SimpleNamespace

    import torch

    from framegen import autodetect, rtxvsr, trt, vfg
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i=0: "NVIDIA GeForce RTX 4070")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i=0: SimpleNamespace(total_memory=12 * 2**30))
    monkeypatch.setattr(autodetect, "primary_display", lambda: UW120)
    monkeypatch.setattr(autodetect, "nvidia_smi", lambda samples=3: {"driver": driver, "busy": 0})
    monkeypatch.setattr(trt, "trt_available", lambda: True)
    monkeypatch.setattr(rtxvsr, "unavailable_reason", lambda: None)
    monkeypatch.setattr(vfg, "vfg_unavailable", lambda: vfg_why)
    return autodetect.detect(DEFAULTS, log=lambda m: None, timing=THIS_PC)


def test_detect_offers_vfg_only_where_it_can_be_set_up(monkeypatch):
    from framegen.vfg import NOT_SET_UP
    r = _detect_on(monkeypatch, NOT_SET_UP)
    assert r["vfg_offer"].startswith("Your RTX 4070 can also run NVIDIA VFG") and r["vfg_offer_ready"] is True
    assert next(f for f in r["found"] if f["label"] == "NVIDIA VFG")["detail"] == "Optional: see below."
    r = _detect_on(monkeypatch, NOT_SET_UP, driver="566.36")  # too old: offered, but the button waits for the driver
    assert "566.36" in r["vfg_offer"] and r["vfg_offer_ready"] is False
    for why in ("NVIDIA VFG needs an RTX 40 or 50 series card.", "No NVIDIA graphics card was found.",
                "NVIDIA VFG couldn't be checked: boom"):
        r = _detect_on(monkeypatch, why)
        assert "vfg_offer" not in r and "vfg_offer_ready" not in r, why


# RX 9070 XT, PS5 Remote Play at 1080p: RIFE alone measured 9.6 ms (full) and 8.4 (half), but a game session logged
# 13 and 10.8 per tick; at 90 fps that stuttered (queue p50 18.8 ms against 6.2 at 60), and 60 keeps up.
RX9070XT_RIFE_ONLY = timing(9.6, 8.4, "dml")
RX9070XT = timing(13.0, 10.8, "dml")  # the whole tick, as time_dml() now times it
QHD144 = Display(2560, 1440, 144, 144)


def test_amd_card_gets_the_rate_a_session_keeps_up_with():
    old = recommend(DEFAULTS, RX9070XT_RIFE_ONLY, QHD144, gpu="AMD Radeon RX 9070 XT", trt=False)
    assert old["settings"]["hz"] == 90  # what RIFE alone suggested
    r = recommend(DEFAULTS, RX9070XT, QHD144, gpu="AMD Radeon RX 9070 XT", trt=False,
                  vsr="it needs an NVIDIA RTX card")
    assert (r["settings"]["hz"], r["settings"]["motion"]) == (60, "auto")  # 60 fps, full detail
    assert "DirectML" in r["reasons"]["engine"]


def test_nvidia_cards_unchanged_at_144_hz():
    assert pick(display=QHD144)["settings"] == {"hz": 90, "motion": "auto", "engine": "auto", "scaler": "fsr"}
    assert pick(t=timing(5.0, 3.5), display=QHD144)["settings"]["hz"] == 120


def test_directml_test_times_the_whole_tick(monkeypatch):
    """time_dml() times what a session does per tick with DirectML (frames on the CPU), not RIFE alone."""
    import time

    import numpy as np
    import torch

    from framegen import autodetect, cadence, dml

    calls = []

    class FakeDml:
        def __init__(self, w, h, model, scale):
            self.in_u8 = torch.zeros((h, w, 3), dtype=torch.uint8)
            self.out_u8 = torch.zeros((h, w, 3), dtype=torch.uint8)

        def load(self, frame):
            assert isinstance(frame, np.ndarray)
            np.copyto(self.in_u8.numpy(), frame)
            calls.append("load")

        def prepare(self, slot):
            time.sleep(0.002)
            calls.append("prepare")

        def interpolate(self, slot, t=0.5):
            time.sleep(0.002)
            calls.append("interpolate")
            return self.out_u8

    checked = []
    real_update = cadence.DuplicateDetector.update
    monkeypatch.setattr(cadence.DuplicateDetector, "update",
                        lambda self, f: checked.append(1) or real_update(self, f))
    monkeypatch.setattr(dml, "DmlInterpolator", FakeDml)
    lines = []
    out = autodetect.time_dml("4.22.lite", log=lines.append)
    assert set(out) == {"full", "half"} and all(v["engine"] == "dml" for v in out.values())
    assert all(v["ms"] >= 4.0 for v in out.values())  # at least prepare + interpolate
    ticks = 2 * 7 * 12  # each motion detail: 7 rounds of 12 ticks (RIFE alone is timed too, without loads)
    assert calls.count("load") == ticks + 2 and len(checked) == ticks + 2
    assert sum("with the frame copies and checks" in m for m in lines) == 2
