"""Agent program: let the trained policy drive and take over only when the garment stops moving.

This is the orchestration arm: the agent supplies the slow decision of when the policy
is stuck and which recovery to run, and the policy supplies everything else.
"""
import numpy as np

STEP = 0.008
WATCH = 8              # decisions handed to the policy between checks
STALL = 0.15
RECOVERY = 8


def unit(vector):
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-9 else np.zeros(3)


def policy(session):
    observation = session.observe()
    stalls = 0
    while session.decisions_left >= WATCH:
        observation = session.run_policy(WATCH)
        response = observation["garment_response"]
        if response is not None and response < STALL and session.decisions_left >= RECOVERY:
            stalls += 1
            observation = session.move(*(STEP * unit(observation["goal_from_tool"])), repeat=RECOVERY)
    return dict(stalls=stalls)
