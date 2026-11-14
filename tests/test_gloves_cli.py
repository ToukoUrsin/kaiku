from __future__ import annotations

from kaiku.app import gloves


def test_checklist_has_the_operator_facing_lines(monkeypatch):
    monkeypatch.setattr(gloves, "_exe", lambda name: "/sdk/" + name)
    monkeypatch.setattr(gloves, "_open", lambda host, port, timeout=0.25: port in (12276, 12277))
    check = gloves.Checklist()
    check.stream = type("Stream", (), {
        "rate_hz": lambda self, side: 100.0,
        "latest": lambda self, side: type("Frame", (), {"coil_present": True})(),
    })()
    rows = check.rows()
    assert [name for name, _, _ in rows] == [
        "SDK found and licensed", "driver running", "hand solver answering",
        "left glove streaming", "right glove streaming", "position from Coil Pro on both hands",
    ]
    assert all(ok for _, ok, _ in rows)


def test_one_glove_is_a_valid_checklist(monkeypatch):
    monkeypatch.setattr(gloves, "_exe", lambda name: "/sdk/" + name)
    monkeypatch.setattr(gloves, "_open", lambda host, port, timeout=0.25: True)
    check = gloves.Checklist()
    check.stream = type("Stream", (), {
        "rate_hz": lambda self, side: 100.0 if side == "left" else 0.0,
        "latest": lambda self, side: type("Frame", (), {"coil_present": False})() if side == "left" else None,
        "connected": True,
        "emulated": False,
    })()
    assert check.ready
    assert any("One glove is connected" in message for message in check.messages())



def _stream(left, right):
    """A glove stream with these hands on: None, or whether that glove has a place on a Coil Pro."""
    frame = lambda coil: None if coil is None else type("Frame", (), {"coil_present": coil})()
    frames = {"left": frame(left), "right": frame(right)}
    return type("Stream", (), {"latest": lambda self, side: frames[side]})()


def test_what_streams_decides_how_kaiku_is_driven():
    pair, _ = gloves.drive_args(_stream(True, True), "127.0.0.1:12276")
    assert pair == ["--gloves", "127.0.0.1:12276"]
    one_hand = ["--mode", "visitor", "--one-glove", "127.0.0.1:12276"]
    for left, right in ((None, False), (True, None), (False, False), (True, False)):
        how, words = gloves.drive_args(_stream(left, right), "127.0.0.1:12276")
        assert how == one_hand                                # calibrating two hands would refuse here
        assert words.startswith("One hand drives")
