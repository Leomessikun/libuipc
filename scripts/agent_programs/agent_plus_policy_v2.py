"""Agent program: the trained policy drives, the agent watches the sleeve and intervenes.

The orchestration arm of the comparison. The agent supplies the slow decisions — when
the policy has stopped making the garment move, and which direction to push then — and
the policy supplies everything else. The recovery is the one the counterfactual branch
study measured to help most often across garments: a full-step push along the arm
toward the shoulder.

Measured context this is written against: the policy does not usually stall
(`2026-09-19-intervention-and-agent-baseline.md`), so this program should intervene
rarely; its interest is whether the few interventions change the episode.
"""
import numpy as np

STEP = 0.008
WATCH = 8
STALL = 0.15
RECOVERY = 8


def unit(vector):
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-9 else np.zeros(3)


def policy(session):
    observation = session.observe()
    interventions = 0
    while session.decisions_left >= WATCH:
        observation = session.run_policy(WATCH)
        response = observation["garment_response"]
        if response is not None and response < STALL and session.decisions_left >= RECOVERY:
            interventions += 1
            observation = session.move(*(STEP * unit(observation["goal_from_tool"])), repeat=RECOVERY)
    return dict(interventions=interventions)
