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

