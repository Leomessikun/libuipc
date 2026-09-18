"""Agent program: drive the sleeve along the arm, watch the garment's response, recover on a stall.

Written against the tool surface of ``uipc_manip.agent_harness``, using only what the
robot can see. The reasoning it encodes is the one a general agent would state: the
sleeve should travel along the arm toward the shoulder; if the garment stops following
the commands, the sleeve is caught, so free it before pushing again.
"""
import numpy as np

STEP = 0.008           # metres per decision, the controller's own limit
SEGMENT = 4            # decisions per committed command
STALL = 0.15           # garment metres moved per commanded metre, below which it is caught
RECOVERY = 8           # decisions spent freeing a caught sleeve


def unit(vector):
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-9 else np.zeros(3)


def policy(session):
    observation = session.observe()
    stalls = 0
    while session.decisions_left >= SEGMENT:
        toward_shoulder = unit(observation["goal_from_tool"])
        observation = session.move(*(STEP * toward_shoulder), repeat=SEGMENT)
        response = observation["garment_response"]
        if response is not None and response < STALL and session.decisions_left >= RECOVERY + SEGMENT:
            stalls += 1
            # Free the sleeve: lift it clear of the arm, then resume along the arm.
            observation = session.move(dz=STEP, repeat=RECOVERY // 2)
            observation = session.move(*(STEP * unit(observation["goal_from_tool"])), repeat=RECOVERY // 2)
    return dict(stalls=stalls)
