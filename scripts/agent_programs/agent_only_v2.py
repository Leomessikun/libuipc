"""Agent program, second attempt: thread the opening over the fingertip before travelling.

The first attempt drove the cuff straight toward the shoulder and dragged the garment
along the outside of the arm: at decision 150 the opening sat at the shoulder with the
arm outside it. Threading has to start at the hand, so this version finds the fingertip
in the arm point cloud, takes the opening past it along the arm's axis, and only then
travels toward the shoulder, keeping the tool a few centimetres clear of the arm.
"""
import numpy as np

STEP = 0.008
SEGMENT = 4
CLEAR = 0.05          # metres the tool is kept above the arm's axis while travelling
PAST_FINGER = 0.06    # metres beyond the fingertip the opening is taken before travelling
STALL = 0.15
RECOVERY = 6


def unit(vector):
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-9 else np.zeros(3)


def arm_frame(points):
    """Fingertip, shoulder and the axis between them, all relative to the tool."""
    arm, goal = points["arm"], points["goal"]
    if len(arm) < 8:
        return None
    # The fingertip is the arm point farthest from the shoulder.
    finger = arm[int(np.argmax(np.linalg.norm(arm - goal[None, :], axis=1)))]
    return dict(finger=finger, shoulder=goal, axis=unit(goal - finger))


def policy(session):
    session.observe()
    points = session.points()
    frame = arm_frame(points)
    if frame is None:
        return dict(reason="no arm points")
    # Stage one: carry the opening beyond the fingertip, approaching from outside the hand.
    target = frame["finger"] - frame["axis"] * PAST_FINGER + np.array([0.0, 0.0, CLEAR])
    stalls = 0
    while session.decisions_left >= SEGMENT:
        offset = target
        if np.linalg.norm(offset) < 0.02:
            break
        session.move(*(STEP * unit(offset)), repeat=SEGMENT)
        points = session.points()
        frame = arm_frame(points) or frame
        target = frame["finger"] - frame["axis"] * PAST_FINGER + np.array([0.0, 0.0, CLEAR])
    # Stage two: travel along the arm's axis toward the shoulder, staying clear of the arm.
    while session.decisions_left >= SEGMENT:
        points = session.points()
        frame = arm_frame(points) or frame
        observation = session.move(*(STEP * frame["axis"]), repeat=SEGMENT)
        response = observation["garment_response"]
        if response is not None and response < STALL and session.decisions_left >= RECOVERY + SEGMENT:
            stalls += 1
            session.move(dz=STEP, repeat=RECOVERY)
    return dict(stalls=stalls)
