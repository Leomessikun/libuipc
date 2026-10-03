"""Material sleeve goals and a small residual proposal controller.

Goals are privileged training/search targets. The decoder observes point-cloud
features and tool history; it receives no current ground-truth cloth descriptor.
This is a feasibility implementation, not a demonstrated novel algorithm.
"""
from __future__ import annotations

import numpy as np

RINGS = 4  # cuff, material sections at .25/.5, armhole
DESCRIPTOR_DIM = RINGS * 9
WAYPOINTS = 3


def arm_frame(landmarks):
    finger, elbow, shoulder = np.asarray(landmarks, float)
    forearm, upper = elbow - finger, shoulder - elbow
    length = np.linalg.norm(forearm) + np.linalg.norm(upper)
    if min(np.linalg.norm(forearm), np.linalg.norm(upper)) < 1e-6:
        raise ValueError("Degenerate arm segment")
    x = forearm / np.linalg.norm(forearm)
    y = upper - np.dot(upper, x) * x
    if np.linalg.norm(y) < .01 * np.linalg.norm(upper):
        raise ValueError("Near-straight arm has no intrinsic transverse frame")
    y /= np.linalg.norm(y)
    return finger, np.stack((x, y, np.cross(x, y))), float(length)


def polygon_normal(points):
    centered = points - points.mean(0)
    normal = np.cross(centered, np.roll(centered, -1, axis=0)).sum(0)
    if np.linalg.norm(normal) < 1e-10:
        raise ValueError("Degenerate sleeve section")
    return normal / np.linalg.norm(normal)


class InteractionCoordinates:
    def __init__(self, sections, landmarks):
        self.sections, self.landmarks = sections, np.asarray(landmarks, float)
        if not np.allclose(sections.fractions[:3], [0., .25, .5]):
            raise ValueError('No consistent cuff/.25/.5 material correspondence')
        self.origin, self.frame, self.length = arm_frame(landmarks)
        # The existing third interior section can move from .75 to .55 on
        # short sleeves. Do not silently equate those material locations.
        rings = sections.points(sections.vertices)[:3] + [sections.vertices[sections.armhole]]
        self.signs = [1. if polygon_normal(r) @ sections.axis >= 0 else -1. for r in rings]

    def describe(self, cloth):
        from physical_sleeve import ring_on_arm
        rings = self.sections.points(cloth)[:3] + [cloth[self.sections.armhole]]
        rows = []
        for ring, sign in zip(rings, self.signs):
            center = ring.mean(0)
            radii = np.linalg.norm(ring - center, axis=1) / self.length
            rows.append(np.r_[((center - self.origin) @ self.frame.T) / self.length,
                              sign * polygon_normal(ring) @ self.frame.T,
                              radii.mean(), radii.std(), float(ring_on_arm(ring, self.landmarks)['wrapped'])])
        result = np.asarray(rows, np.float32)
        if result.shape != (RINGS, 9) or not np.isfinite(result).all():
            raise ValueError("Unsupported interaction coordinates")
        return result


def normal_alignment(source, target):
    """Proper minimal rotation; antipodal normals have no unique transport."""
    a, b = np.asarray(source, float), np.asarray(target, float)
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    cosine = np.clip(a @ b, -1., 1.)
    if cosine < -.9999:
        raise ValueError("Antipodal section normals: ambiguous transport")
    v = np.cross(a, b)
    skew = np.array([[0., -v[2], v[1]], [v[2], 0., -v[0]], [-v[1], v[0], 0.]])
    return np.eye(3) + skew + skew @ skew / (1. + cosine)


def transport(path, target_start):
    """Transfer normalized displacements, radius ratios and event changes.

    Neither cloth vertices nor simulator state are changed. This operation
    proposes geometric targets; their reachability must be tested in IPC.
    """
    path, target = np.asarray(path, float), np.asarray(target_start, float)
    if path.ndim != 3 or path.shape[1:] != (RINGS, 9) or target.shape != (RINGS, 9):
        raise ValueError("Expected an ordered material-section path")
    start = path[0]
    result = path.copy()
    result[..., :3] = target[None, :, :3] + path[..., :3] - start[None, :, :3]
    for i in range(RINGS):
        rotation = normal_alignment(start[i, 3:6], target[i, 3:6])
        result[:, i, 3:6] = path[:, i, 3:6] @ rotation.T
    # Transfer both size statistics with the mean-radius scale; a nearly
    # circular source (zero radius standard deviation) must not divide by zero.
    ratio = target[:, 6] / np.maximum(start[:, 6], 1e-6)
    result[..., 6:8] = np.maximum(0., target[None, :, 6:8]
                                 + (path[..., 6:8] - start[None, :, 6:8]) * ratio[None, :, None])
    result[..., 8] = np.clip(target[None, :, 8] + path[..., 8] - start[None, :, 8], 0., 1.)
    result[0] = target
    return result.astype(np.float32)


def goal_input(path, landmarks, tool, model_rotation):
    """Desired geometry in measured tool-relative model coordinates.

    Arm landmarks instantiate the *requested goal* in the simulation teacher.
    They are not an additional observation to the correction decoder.
    """
    origin, frame, length = arm_frame(landmarks)
    result = np.asarray(path, float).copy()
    result[..., :3] = ((result[..., :3] * length) @ frame + origin - tool) @ model_rotation.T / .1
    result[..., 3:6] = result[..., 3:6] @ frame @ model_rotation.T
    result[..., 6:8] *= length / .1
    return result.reshape(-1).astype(np.float32)


def future_indices(t, horizon):
    if horizon < 1:
        raise ValueError("A goal needs at least one future decision")
    return t + np.maximum(1, np.rint(np.arange(1, WAYPOINTS + 1) * horizon / WAYPOINTS).astype(int))


def valid_prefix(grasp_valid, metadata):
    valid = np.asarray(grasp_valid, bool)
    invalid = np.flatnonzero(~valid)
    end = int(invalid[0]) if len(invalid) else len(valid)
    error = metadata.get('sim_error')
    if error and not str(error).startswith('grasp tracking exceeded validity limit'):
        return 0
    first_success = metadata.get('success_state')
    if first_success is not None:
        end = min(end, int(first_success))
    return end  # states 0..end; actions 0..end-1 only


def base_action(logits, rotation, max_rotation=.08726646):
    """Match the v5 collector's FMVP rotation postprocessing exactly."""
    raw = np.tanh(np.asarray(logits))
    world = np.concatenate((raw[..., :3] @ rotation, raw[..., 3:] @ rotation), axis=-1)
    vertical = (world[..., 3:] @ rotation.T)[..., 1]
    delta = np.abs(vertical)
    delta = np.where(delta > np.deg2rad(5.), delta * np.deg2rad(5.) / np.sqrt(3.), delta)
    world[..., 3:] = 0.
    world[..., 5] = np.sign(vertical) * delta / max_rotation
    return np.clip(world, -1., 1.).astype(np.float32)


def proposal_net(input_dim, chunk=8):
    import torch
    model = torch.nn.Sequential(torch.nn.Linear(input_dim, 256), torch.nn.ReLU(),
                                torch.nn.Linear(256, 128), torch.nn.ReLU(), torch.nn.Linear(128, 4 * chunk))
    torch.nn.init.zeros_(model[-1].weight)
    torch.nn.init.zeros_(model[-1].bias)
    return model


def residual_action(model, inputs, base):
    import torch
    # The existing action family translates xyz and rotates about world z.
    indices = [0, 1, 2, 5]
    correction = .5 * model(inputs).reshape(len(inputs), -1, 4).tanh()
    action = base[:, None].expand(-1, correction.shape[1], -1).clone()
    action[:, :, indices] = (action[:, :, indices] + correction).clamp(-1., 1.)
    return action, correction
