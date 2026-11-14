"""Named hand postures: open, relaxed, thumb-to-finger pinches and a grip of a given aperture.

These are the vocabulary of the scripted operator in ``synth.py``. Every posture respects the joint limits
and the distal coupling of ``model.py``, so it is something the glove can observe and the solver can recover
exactly. The pinch postures are found by a small least-squares fit when first asked for (tens of
milliseconds, cached); nothing here runs per frame.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from kaiku.hand.model import HandModel
from kaiku.types import FINGERS

_DEG = np.pi / 180.0

# Columns: [flex 0, flex 1, flex 2 (filled in by the coupling), abduction], rows in FINGERS order.
_OPEN = np.array([
    [-5.0, 0.0, 0.0, 10.0],
    [4.0, 4.0, 0.0, 6.0],
    [4.0, 4.0, 0.0, 0.0],
    [4.0, 4.0, 0.0, -5.0],
    [4.0, 4.0, 0.0, -10.0],
]) * _DEG
_RELAXED = np.array([
    [10.0, 12.0, 0.0, 15.0],
    [22.0, 28.0, 0.0, 2.0],
    [26.0, 32.0, 0.0, 0.0],
    [30.0, 36.0, 0.0, -2.0],
    [34.0, 40.0, 0.0, -4.0],
]) * _DEG
# Hand shaped to receive an object: thumb lifted well away from the palm, fingers nearly straight.
_PRESHAPE = np.array([
    [5.0, 5.0, 0.0, 45.0],
    [12.0, 12.0, 0.0, 0.0],
    [12.0, 12.0, 0.0, 0.0],
    [14.0, 14.0, 0.0, 0.0],
    [16.0, 16.0, 0.0, 0.0],
]) * _DEG
# Where a pinch likes to be when the geometry leaves a choice: [thumb flex 0, thumb flex 1, thumb abduction,
# finger flex 0, finger flex 1, finger abduction].
_PINCH_PREFERENCE = np.array([20.0, 20.0, 30.0, 45.0, 45.0, 0.0]) * _DEG
_PINCH_COLUMNS = [0, 1, 3]   # the joints a pinch may move; the last joint follows by coupling
_GRIP_SAMPLES = 65


class HandPoses:
    """Postures of one hand model as (5, 4) joint arrays. The same values serve the left and the right hand."""

    def __init__(self, model: HandModel) -> None:
        self.model = model
        self.open = model.couple(_OPEN)
        self.relaxed = model.couple(_RELAXED)
        self._pinch: dict[str, np.ndarray] = {}
        self._grip_table: tuple[np.ndarray, np.ndarray] | None = None

    def pinch(self, finger: str) -> np.ndarray:
        """Posture in which the thumb tip touches the tip of ``finger`` ("index" ... "little").

        The other fingers stay relaxed. Raises ``ValueError`` if the model cannot bring the two tips together
        within its joint limits (it can, for the stock anatomy: tests/test_hand_model.py).
        """
        if finger not in FINGERS[1:]:
            raise ValueError(f"cannot pinch the thumb against {finger!r}")
        if finger not in self._pinch:
            self._pinch[finger] = self._solve_pinch(FINGERS.index(finger))
        return self._pinch[finger].copy()

    def grip(self, aperture: float) -> np.ndarray:
        """Grasp posture whose thumb-tip-to-index-tip distance is ``aperture`` metres (``HumanHand.pinch[0]``).

        The hand closes along a straight line in joint space from an open pre-shape to a pinch in which all
        four fingers curl like the index finger. Apertures beyond the pre-shape's are clamped to it.
        """
        if self._grip_table is None:
            self._grip_table = self._build_grip_table()
        apertures, closure = self._grip_table
        return self._grip_at(float(np.interp(aperture, apertures, closure)))

    def _solve_pinch(self, finger: int) -> np.ndarray:
        model = self.model
        rows = [0, finger]
        lower = model.lower[np.ix_(rows, _PINCH_COLUMNS)].ravel()
        upper = model.upper[np.ix_(rows, _PINCH_COLUMNS)].ravel()

        def posture(x: np.ndarray) -> np.ndarray:
            joints = self.relaxed.copy()
            joints[np.ix_(rows, _PINCH_COLUMNS)] = x.reshape(2, 3)
            return model.couple(joints)

        def residual(x: np.ndarray, preference: float) -> np.ndarray:
            tips = model.tips(posture(x))
            return np.concatenate([tips[0] - tips[finger], preference * (x - _PINCH_PREFERENCE)])

        # First settle near a natural-looking posture, then close the remaining gap exactly.
        start = np.clip(_PINCH_PREFERENCE, lower, upper)
        natural = least_squares(residual, start, bounds=(lower, upper), args=(1e-3,))
        exact = least_squares(residual, natural.x, bounds=(lower, upper), args=(0.0,), xtol=1e-14, ftol=1e-14)
        gap = float(np.linalg.norm(exact.fun[:3]))
        if gap > 1e-4:
            raise ValueError(f"thumb cannot reach the {FINGERS[finger]} fingertip: {gap * 1000:.1f} mm short")
        return posture(exact.x)

    def _grip_at(self, closure: float) -> np.ndarray:
        """Grip posture at ``closure`` in [0, 1]: 0 is the open pre-shape, 1 the closed pinch."""
        closed = self.pinch("index")
        closed[2:, :3] = closed[1, :3]
        closed[2:, 3] = 0.0
        return self.model.couple(_PRESHAPE + closure * (closed - _PRESHAPE))

    def _build_grip_table(self) -> tuple[np.ndarray, np.ndarray]:
        """Aperture as a function of closure, as ascending apertures with their closures, for ``np.interp``."""
        closure = np.linspace(0.0, 1.0, _GRIP_SAMPLES)
        tips = np.array([self.model.tips(self._grip_at(c)) for c in closure])
        apertures = np.linalg.norm(tips[:, 0] - tips[:, 1], axis=1)
        if np.any(np.diff(apertures) >= 0.0):
            raise ValueError("grip aperture does not shrink monotonically as the hand closes")
        return apertures[::-1].copy(), closure[::-1].copy()
