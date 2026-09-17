"""CPU checks of the multi-decision chain: its last decision agrees with the one-decision adjoint,
and its command bookkeeping (executed fractions, rotation through the held offsets) agrees with
differences of the linearised aim model."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

scipy = pytest.importorskip("scipy")
import scipy.sparse  # noqa: E402
import scipy.sparse.linalg  # noqa: E402

from uipc_manip import physics_gradient_adjoint as adjoint  # noqa: E402
from uipc_manip import physics_gradient_trajopt as trajopt  # noqa: E402
from uipc_manip.dressing_env import _rodrigues  # noqa: E402
from uipc_manip.tests.test_physics_gradient_adjoint import _upper_block_system  # noqa: E402

REPEAT = 6


def _env():
    return SimpleNamespace(cfg=SimpleNamespace(max_translation=0.01, max_rotation=0.0873, action_repeat=REPEAT, clip_rotation_to_yz=True))


def _layout(rng, n):
    return {"dof_offset": 6, "dof_count": 3 * n, "n": n, "mass": rng.uniform(0.5, 2.0, n), "strength": 40.0,
            "anchor_idx": np.array([0, 3, 5]), "opening_idx": np.array([6, 7]), "axis": np.array([0.0, 1.0, 0.0])}


def _frames(rng, n, actions, env, executed=None):
    """Frames of h decisions, every command executed (or the given fraction), systems random SPD."""
    h = actions.shape[0]
    anchor = np.array([0.1, 0.2, 0.3])
    offsets = rng.standard_normal((3, 3)) * 0.02
    frames = []
    for t in range(h):
        start_anchor, start_offsets = anchor.copy(), offsets.copy()
        cmd_t = actions[t, :3] * env.cfg.max_translation
        cmd_r = actions[t, 3:] * env.cfg.max_rotation
        cmd_r[0] = 0.0
        for k in range(1, REPEAT + 1):
            frac = 1.0 if executed is None else executed[t]
            anchor = start_anchor + frac * (k / REPEAT) * cmd_t
            offsets = _rodrigues(offsets, frac * cmd_r / REPEAT)
            dense = _upper_block_system(rng, n + 2)[0]
            frames.append({"H": scipy.sparse.csr_matrix(dense), "anchor": anchor.copy(), "offsets": offsets.copy(),
                           "decision": t, "substep": k, "start_anchor": start_anchor, "start_offsets": start_offsets})
    return frames


def test_last_decision_matches_the_one_decision_adjoint():
    rng = np.random.default_rng(3)
    n, env = 8, _env()
    layout = _layout(rng, n)
    actions = np.zeros((2, 6))
    actions[0] = [0.3, -0.2, 0.1, 0.0, 0.2, -0.1]
    frames = _frames(rng, n, actions, env)
    g = adjoint.axis_gradient(layout)
    chain = trajopt.chain_gradient(frames, layout, g, actions, env)
    lu = [scipy.sparse.linalg.splu(fr["H"].tocsc()) for fr in frames[REPEAT:]]
    single = adjoint.reverse_pass(lu, g, layout, chain=True)["dL_dDelta"]
    assert np.allclose(chain["per_metre"][1], single, rtol=1e-9, atol=1e-12)
    assert chain["per_action"][1, 3] == 0.0 and chain["per_action"][0, 3] == 0.0
    assert np.allclose(chain["per_action"][:, :3], chain["per_metre"] * env.cfg.max_translation)


def test_command_bookkeeping_matches_differences_of_the_aim_model():
    """With λ fixed, L is linear in the aims: L(a) = Σ_f Σ_I (K λ_{f,I}) · aim_{f,I}(a). Its differences
    over the first decision's translation and rotation must match the chain's mapping."""
    rng = np.random.default_rng(4)
    n, env = 8, _env()
    layout = _layout(rng, n)
    actions = np.zeros((2, 6))
    actions[0] = [0.2, 0.1, -0.3, 0.0, 0.15, -0.25]
    actions[1] = [-0.1, 0.2, 0.1, 0.0, -0.05, 0.1]
    frames = _frames(rng, n, actions, env)
    g = adjoint.axis_gradient(layout)
    chain = trajopt.chain_gradient(frames, layout, g, actions, env)

    def aims(a):
        anchor = frames[0]["start_anchor"].copy()
        offsets = frames[0]["start_offsets"].copy()
        out = []
        for t in range(a.shape[0]):
            start = anchor.copy()
            cmd_t = a[t, :3] * env.cfg.max_translation
            cmd_r = a[t, 3:] * env.cfg.max_rotation
            cmd_r[0] = 0.0
            for k in range(1, REPEAT + 1):
                anchor = start + (k / REPEAT) * cmd_t
                offsets = _rodrigues(offsets, cmd_r / REPEAT)
                out.append(anchor[None, :] + offsets)
        return out

    def L(a):
        return sum(float(np.sum(held * aim)) for held, aim in zip(chain["dL_daim"], aims(a), strict=True))

    fd = np.zeros(6)
    eps = 1e-6
    for k in range(6):
        plus, minus = actions.copy(), actions.copy()
        plus[0, k] += eps
        minus[0, k] -= eps
        fd[k] = (L(plus) - L(minus)) / (2 * eps)
    assert fd[3] == 0.0
    # Translation is exact; rotation is the small-angle model, good to a percent at these angles.
    assert np.allclose(chain["per_action"][0, :3], fd[:3], rtol=1e-6, atol=1e-12)
    assert np.allclose(chain["per_action"][0, 4:], fd[4:], rtol=2e-2, atol=1e-12)


def test_refused_substeps_contribute_nothing():
    rng = np.random.default_rng(5)
    n, env = 8, _env()
    layout = _layout(rng, n)
    actions = np.zeros((2, 6))
    actions[0] = [0.3, 0.0, 0.0, 0.0, 0.0, 0.2]
    actions[1] = [0.1, 0.0, 0.0, 0.0, 0.0, 0.0]
    frames = _frames(rng, n, actions, env, executed=[0.0, 1.0])
    chain = trajopt.chain_gradient(frames, layout, adjoint.axis_gradient(layout), actions, env)
    assert np.allclose(chain["per_action"][0], 0.0)
    assert np.linalg.norm(chain["per_action"][1, :3]) > 0.0
    assert np.allclose(chain["executed_translation_ratio"][:REPEAT], 0.0)


def test_coverage_objective_gradient_is_the_leading_triangle():
    cell = SimpleNamespace(shoulder=np.array([0.0, 0.0, 0.0]), elbow=np.array([1.0, 0.0, 0.0]),
                           polygon_triangles=lambda: np.array([[0, 1, 2], [3, 4, 5]]))
    env = SimpleNamespace(cells=[cell])
    # Two triangles crossing the ray, at x = 0.6 and x = 0.3; the reference's distance is measured
    # back from the elbow (x = 1) and takes the minimum over the triangles hit, here the one at 0.6.
    pos = np.array([[0.6, -1, -1], [0.6, 1, -1], [0.6, 0, 1], [0.3, -1, -1], [0.3, 1, -1], [0.3, 0, 1]], dtype=float)
    value, g, on = trajopt.coverage_objective(env, pos)
    assert on and value == pytest.approx(0.4)
    g = g.reshape(-1, 3)
    assert np.allclose(g[3:], 0.0)
    # Moving the hit triangle toward the shoulder (-x) raises the distance; the ray meets it at the
    # barycentric point (0.25, 0.25, 0.5), which is how the unit slope splits over its vertices.
    assert np.allclose(g[:3, 0], [-0.25, -0.25, -0.5], atol=1e-5) and np.allclose(g[:3, 1:], 0.0, atol=1e-5)
    value, g, on = trajopt.coverage_objective(env, pos + np.array([2.0, 0, 0]))
    assert not on and value == 0.0 and not g.any()


def test_controller_jacobian_zero_commands_respect_rejection_and_clipping():
    env = _env()
    actions = np.zeros((2, 6))
    actions[1, 0] = 1.2  # fully saturated: no local derivative
    anchor, offsets = np.array([0.1, 0.2, 0.3]), np.array([[0.02, 0.01, -0.01]])
    frames = []
    for t in range(2):
        start = anchor.copy()
        for k in range(REPEAT):
            # Entire first decision rejected, including its zero rotation.
            accepted = t == 1
            if accepted:
                anchor = anchor + np.array([env.cfg.max_translation / REPEAT, 0, 0])
            frames.append(dict(anchor=anchor.copy(), offsets=offsets.copy(), start_anchor=start,
                               start_offsets=offsets.copy(), decision=t, substep=k + 1,
                               translation_accepted=accepted, rotation_accepted=accepted))
    aims, tools = trajopt.controller_jacobians(frames, actions, env)
    assert np.allclose(aims[..., 0, :], 0) and np.allclose(tools[..., 0, :], 0)
    assert np.allclose(aims[..., 1, 0], 0) and np.allclose(tools[..., 1, 0], 0)
    assert np.allclose(aims[..., 3], 0)  # x rotation ignored
    assert np.allclose(tools[-1, :, 1, 1], [0, env.cfg.max_translation, 0])
    # At zero rotation, d offset / d angle_y = e_y cross offset.
    assert np.allclose(aims[-1, 0, :, 1, 4], np.cross([0, 1, 0], offsets[0]) * env.cfg.max_rotation)


def test_controller_jacobian_matches_finite_rotation_with_partial_acceptance():
    from scipy.spatial.transform import Rotation

    env = _env()
    env.cfg.max_rotation = 0.8
    action = np.array([[0.2, -0.1, 0.3, 0.9, 0.7, -0.5]])
    initial = np.array([[0.01, 0.02, -0.03], [-0.02, 0.01, 0.04]])
    anchor, offsets = np.zeros(3), initial.copy()
    frames = []
    for k in range(REPEAT):
        trans, rot = k % 2 == 0, k != 2
        if trans:
            anchor += action[0, :3] * env.cfg.max_translation / REPEAT
        if rot:
            offsets = Rotation.from_rotvec([0, action[0, 4] * 0.8 / REPEAT, action[0, 5] * 0.8 / REPEAT]).apply(offsets)
        frames.append(dict(anchor=anchor.copy(), offsets=offsets.copy(), start_anchor=np.zeros(3),
                           start_offsets=initial, decision=0, substep=k + 1,
                           translation_accepted=trans, rotation_accepted=rot))
    aims, tools = trajopt.controller_jacobians(frames, action, env)
    def endpoint(a):
        rot = Rotation.from_rotvec([0, a[4] * 0.8 * 5 / REPEAT, a[5] * 0.8 * 5 / REPEAT])
        return a[:3] * env.cfg.max_translation * 3 / REPEAT + rot.apply(initial)
    fd = np.zeros((2, 3, 6))
    for k in range(6):
        p, m = action[0].copy(), action[0].copy()
        p[k] += 1e-6
        m[k] -= 1e-6
        fd[:, :, k] = (endpoint(p) - endpoint(m)) / 2e-6
    assert np.allclose(aims[-1, :, :, 0, :], fd, atol=1e-9)
