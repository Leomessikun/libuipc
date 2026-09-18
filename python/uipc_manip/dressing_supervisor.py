"""A supervisor that repairs the scripted dressing teacher where it was measured to fail.

The scripted expert passes nine of 25 cells under the corrected reward. Its failures
were measured, not guessed, and there are three of them:

* **it stalls at the elbow.** In eight of twelve failed episodes the opening moves less
  than a centimetre per twenty decisions for about 130 decisions while the translation
  command stays saturated: 1,340 of 7,500 decisions, of which 1,045 carry a saturated
  command (`2026-09-17-sleeve-path-audit.md`).
* **it loses the sleeve sideways.** The policy's own episodes end with the opening's
  centre about one ring radius from the arm's centreline, where the ring is balanced on
  the edge of the arm and then leaves it (`2026-09-19-intervention-and-agent-baseline.md`).
* **it strains the grasp.** Failed episodes reach 2.3–3.2 cm of tracking error against a
  2 cm limit.

The supervisor watches the privileged sleeve state — it is a teacher, so it may — and
overrides the expert's command when one of those three is happening.

**Which of the three is worth acting on was then measured**
(`2026-09-19-teacher-supervision.md`). Over the teacher's own 25 episodes, the
fraction of decisions each condition fires on, in the episodes that succeed against the
ones that fail:

* tracking error above 1.5 cm: 0.0 % against 40.2 % — it separates cleanly;
* the opening's centre past 0.6 of its own radius from the centreline: 5.7 % against
  15.0 % — it fires through successful dressings too;
* arc progress below 0.005 per twenty decisions: 35.7 % against 33.8 % — it does not
  separate at all, so a stall by that definition is ordinary.

A first version acted on all three and turned nine successes into none: the two
non-separating rules replaced a third of every episode. Only the grasp rule is enabled
by default, and its response is no longer to scale the whole command down — that kept
the grip in 21 of 25 cells but left only one dressed — but to drop the component that
pulls the cloth across the arm while keeping the one that advances it.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from .dressing_privileged import PRIVILEGED_LAYOUT

OFFSETS = {}
_position = 0
for _name, _width in PRIVILEGED_LAYOUT:
    OFFSETS[_name] = (_position, _position + _width)
    _position += _width


def read(privileged: np.ndarray, block: str) -> np.ndarray:
    start, stop = OFFSETS[block]
    return np.asarray(privileged, dtype=np.float64)[start:stop]


def arm_polyline(privileged: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fingertip, elbow and shoulder in the privileged state's own arm frame."""
    arm = read(privileged, "arm")
    length, shoulder_along, shoulder_across = float(arm[0]), float(arm[1]), float(arm[2])
    return np.zeros(3), np.array([length, 0.0, 0.0]), np.array([shoulder_along, shoulder_across, 0.0])


def sleeve_position(privileged: np.ndarray) -> dict:
    """Where the opening sits on the arm: arc fraction, lateral distance and ring radius."""
    finger, elbow, shoulder = arm_polyline(privileged)
    centre = read(privileged, "opening_center")
    total = float(np.linalg.norm(elbow - finger) + np.linalg.norm(shoulder - elbow))
    best = None
    for name, (a, b, base) in (("forearm", (finger, elbow, 0.0)),
                               ("upperarm", (elbow, shoulder, float(np.linalg.norm(elbow - finger))))):
        direction = b - a
        span = float(np.linalg.norm(direction))
        if span < 1e-9:
            continue
        unit = direction / span
        along = float(np.clip((centre - a) @ unit, 0.0, span))
        foot = a + along * unit
        gap = float(np.linalg.norm(centre - foot))
        if best is None or gap < best["lateral"]:
            best = dict(segment=name, arc=(base + along) / max(total, 1e-9), lateral=gap,
                        outward=centre - foot, axis=unit)
    radius = float(read(privileged, "opening_radius")[0])
    progress = read(privileged, "progress")
    best.update(radius=radius, containment=best["lateral"] / max(radius, 1e-9),
                tracking_cm=float(read(privileged, "tracking_error")[0]),
                forearm_ratio=float(progress[0]), upperarm_ratio=float(progress[1]),
                # Both overrides concern a sleeve that is already on the arm. Before that the
                # opening is legitimately off the arm's axis and legitimately not advancing along
                # it, and overriding there replaces the teacher's whole approach.
                on_arm=bool(progress[2] > 0.5 or progress[3] > 0.5))
    return best


class TeacherSupervisor:
    """Overrides a scripted teacher's command while one of the measured failures is happening.

    ``stall_window`` decisions of arc progress below ``stall_arc`` is a stall;
    a ring centre farther than ``containment`` of its own radius from the centreline is
    drifting off the arm; a tracking error above ``grasp_cm`` is straining the grasp.
    Each override lasts ``hold`` decisions and the reasons are reported per decision.
    """

    def __init__(self, *, stall_window: int = 20, stall_arc: float = 0.02, containment: float = 0.6,
                 grasp_cm: float = 1.2, hold: int = 8, step: float = 1.0,
                 rules: tuple[str, ...] = ("grasp",), grasp_response: str = "project"):
        if stall_window < 2 or hold < 1 or not 0 < containment:
            raise ValueError("Need a window of at least two decisions, a positive hold and containment")
        if not set(rules) <= {"grasp", "centre", "stall"} or grasp_response not in ("project", "halve"):
            raise ValueError("Unknown rule or grasp response")
        self.stall_window, self.stall_arc, self.containment = int(stall_window), float(stall_arc), float(containment)
        self.grasp_cm, self.hold, self.step = float(grasp_cm), int(hold), float(step)
        self.rules, self.grasp_response = tuple(rules), str(grasp_response)
        self._arc: deque[float] = deque(maxlen=stall_window + 1)
        self._plan: list[np.ndarray] = []
        self.reasons: list[str] = []

    def reset(self) -> None:
        self._arc.clear()
        self._plan.clear()
        self.reasons.clear()

    def command(self, base_action: np.ndarray, privileged: np.ndarray) -> tuple[np.ndarray, str]:
        """The command to execute this decision, and why it is not the teacher's own."""
        action = np.array(base_action, dtype=np.float64).copy()
        state = sleeve_position(privileged)
        self._arc.append(state["arc"])
        # A strained grasp is answered first and without a plan: it reshapes what is commanded.
        if "grasp" in self.rules and state["tracking_cm"] > self.grasp_cm:
            self.reasons.append("grasp")
            if self.grasp_response == "halve":
                return np.clip(action * 0.5, -1.0, 1.0), "grasp"
            # Keep what advances the sleeve along the arm and drop what pulls it across:
            # the strain comes from the transverse pull, the dressing from the axial one.
            axis = state["axis"]
            action[:3] = float(action[:3] @ axis) * axis
            return np.clip(action, -1.0, 1.0), "grasp"
        if self._plan:
            self.reasons.append("plan")
            return np.clip(self._plan.pop(0), -1.0, 1.0), "plan"
        drifting = ("centre" in self.rules and state["on_arm"]
                    and state["containment"] > self.containment)
        stalled = ("stall" in self.rules and state["on_arm"] and len(self._arc) > self.stall_window
                   and self._arc[-1] - self._arc[0] < self.stall_arc
                   and state["upperarm_ratio"] < 0.95)
        if drifting:
            direction = -state["outward"]
            reason = "centre"
        elif stalled:
            direction = state["axis"]
            reason = "stall"
        else:
            self.reasons.append("teacher")
            return np.clip(action, -1.0, 1.0), "teacher"
        norm = float(np.linalg.norm(direction))
        unit = direction / norm if norm > 1e-9 else state["axis"]
        override = np.zeros_like(action)
        override[:3] = self.step * unit
        self._plan = [override.copy() for _ in range(self.hold - 1)]
        self._arc.clear()
        self.reasons.append(reason)
        return np.clip(override, -1.0, 1.0), reason
