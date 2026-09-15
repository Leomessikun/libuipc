"""CPU checks of the adjoint script's linear algebra: the two matrix readers agree, and the reverse
pass is the transpose of the tangent pass on a synthetic chain of frames."""
from __future__ import annotations

import numpy as np
import pytest

scipy = pytest.importorskip("scipy")
import scipy.io  # noqa: E402
import scipy.sparse  # noqa: E402
import scipy.sparse.linalg  # noqa: E402

from uipc_manip import physics_gradient_adjoint as adj  # noqa: E402


def _upper_block_system(rng, blocks: int):
    """A random block-sparse SPD-ish system stored as its upper block triangle (row <= col)."""
    dense = rng.standard_normal((3 * blocks, 3 * blocks))
    dense = dense @ dense.T + 3 * blocks * np.eye(3 * blocks)
    rows, cols, values = [], [], []
    for i in range(blocks):
        for j in range(i, blocks):
            if j > i and rng.random() < 0.5:
                dense[3 * i:3 * i + 3, 3 * j:3 * j + 3] = 0.0
                dense[3 * j:3 * j + 3, 3 * i:3 * i + 3] = 0.0
                continue
            rows.append(i)
            cols.append(j)
            values.append(dense[3 * i:3 * i + 3, 3 * j:3 * j + 3])
    return dense, np.asarray(rows), np.asarray(cols), np.stack(values)


def test_export_and_dump_readers_agree(tmp_path):
    rng = np.random.default_rng(0)
    dense, rows, cols, values = _upper_block_system(rng, 7)
    from_export = adj.system_to_matrix(rows, cols, values, 21)
    # The engine's dump holds the same upper block triangle under a 'general' header.
    i = (3 * rows[:, None, None] + np.arange(3)[None, :, None] + np.zeros((1, 1, 3), dtype=int)).reshape(-1)
    j = (3 * cols[:, None, None] + np.zeros((1, 3, 1), dtype=int) + np.arange(3)[None, None, :]).reshape(-1)
    path = tmp_path / "A.1.0.mtx"
    scipy.io.mmwrite(path, scipy.sparse.coo_matrix((values.reshape(-1), (i, j)), shape=(21, 21)), symmetry="general")
    from_dump, info = adj.read_symmetric(path)
    assert info["mirrored"] and info["lower_block_entries"] == 0
    assert np.allclose(from_export.toarray(), dense)
    assert np.allclose(from_dump.toarray(), dense)


def test_reverse_pass_is_the_transpose_of_the_tangent_pass():
    rng = np.random.default_rng(1)
    n, extra, frames = 9, 2, 4  # cloth vertices, extra rigid dofs ahead of them, frames per decision
    layout = {"dof_offset": 3 * extra, "dof_count": 3 * n, "n": n, "mass": rng.uniform(0.5, 2.0, n), "strength": 50.0,
              "anchor_idx": np.array([0, 4]), "opening_idx": np.array([6, 7, 8]), "axis": np.array([0.0, 0.0, 1.0])}
    lu = []
    for _ in range(frames):
        dense = _upper_block_system(rng, n + extra)[0]
        lu.append(scipy.sparse.linalg.splu(scipy.sparse.csc_matrix(dense)))
    tangent = adj.tangent_pass(lu, layout, chain=True)
    g = adj.axis_gradient(layout)
    reverse = adj.reverse_pass(lu, g, layout, chain=True)["dL_dDelta"]
    assert np.allclose(g @ tangent, reverse, rtol=1e-10, atol=1e-12)
    # Without the chain only the last frame's constraint term is differentiated.
    tangent_last = adj.tangent_pass(lu, layout, chain=False)
    reverse_last = adj.reverse_pass(lu, g, layout, chain=False)["dL_dDelta"]
    assert np.allclose(g @ tangent_last, reverse_last, rtol=1e-10, atol=1e-12)
    assert not np.allclose(reverse, reverse_last)
    frames_out = adj.tangent_pass(lu, layout, return_frames=True)
    np.testing.assert_allclose(frames_out[-1], tangent)
    assert frames_out.shape == (frames, 3*n, 3)


def test_compare_fields_reports_held_and_free_separately():
    layout = {"n": 4, "anchor_idx": np.array([1])}
    fd = np.arange(12, dtype=float).reshape(12, 1).repeat(3, axis=1) + 1.0
    out = adj.compare_fields(2.0 * fd, fd, layout)
    for group in ("all", "held", "free"):
        for ax in "xyz":
            assert out[group][ax]["cosine"] == pytest.approx(1.0)
            assert out[group][ax]["magnitude_ratio"] == pytest.approx(2.0)
    assert out["held_follow"]["tangent"] == pytest.approx([2 * v for v in out["held_follow"]["fd"]])


def test_tangent_pass_prev_coupling_matches_the_explicit_recurrence():
    """With a lagged coupling block B on one vertex, the chain solves H X_f = −(B_in + B) X_{f−1} − C X_{f−2} − F_u."""
    import scipy.sparse
    import scipy.sparse.linalg
    rng = np.random.default_rng(5)
    n, frames = 4, 3
    mass = rng.uniform(1, 2, n)
    layout = dict(dof_offset=0, dof_count=3 * n, n=n, mass=mass, strength=10.0, anchor_idx=np.array([0]))
    H = [scipy.sparse.csc_matrix(np.diag(np.repeat(mass, 3)) + 0.3 * np.eye(3 * n)) for _ in range(frames)]
    lu = [scipy.sparse.linalg.splu(h) for h in H]
    B = rng.normal(size=(3, 3))
    coupling = [(np.array([2]), np.array([0]), B[None]) for _ in range(frames)]  # pulled by the held vertex
    frames_out = adj.tangent_pass(lu, layout, return_frames=True, prev_coupling=coupling)
    m3 = np.repeat(mass, 3)
    for k in range(3):
        x_prev, x_prev2 = np.zeros(3 * n), np.zeros(3 * n)
        for f in range(frames):
            rhs = 2 * m3 * x_prev - m3 * x_prev2
            rhs[6:9] -= B @ x_prev[0:3]
            aim = np.zeros(3 * n)
            aim[3 * 0 + k] = (f + 1) / frames
            rhs += layout["strength"] * m3 * aim
            x = lu[f].solve(rhs)
            np.testing.assert_allclose(frames_out[f, :, k], x, rtol=1e-12, atol=1e-14)
            x_prev2, x_prev = x_prev, x
    plain = adj.tangent_pass(lu, layout, return_frames=True)
    assert not np.allclose(plain, frames_out)


def test_slot_factorizations_match_the_global_solve():
    """Two non-interacting slots: per-block LUs reproduce the whole-system tangent of each slot."""
    import scipy.sparse
    import scipy.sparse.linalg
    rng = np.random.default_rng(7)
    n, frames = 5, 2
    blocks, layouts, masses = [], [], []
    for j in range(2):
        mass = rng.uniform(1, 2, n)
        a = rng.normal(size=(3 * n, 3 * n)) * 0.1
        blocks.append(np.diag(np.repeat(mass, 3)) + a @ a.T)
        layouts.append(dict(dof_offset=3 * n * j, dof_count=3 * n, n=n, mass=mass, strength=10.0, anchor_idx=np.array([j])))
    mat = scipy.sparse.csr_matrix(scipy.linalg.block_diag(*blocks))
    global_lu = [scipy.sparse.linalg.splu(mat.tocsc()) for _ in range(frames)]
    per_slot = [adj.slot_factorizations(mat, layouts) for _ in range(frames)]
    for j in range(2):
        expected = adj.tangent_pass(global_lu, layouts[j], return_frames=True)
        got = adj.tangent_pass([f[j][0] for f in per_slot], per_slot[0][j][1], return_frames=True)
        np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-13)
        assert per_slot[0][j][1]["dof_offset"] == 0 and per_slot[0][j][0].shape == (3 * n, 3 * n)


def test_slot_factorizations_from_triplets_match_the_global_solve():
    """Per-slot LUs built straight from block triplets agree with the whole-system tangent, and a
    cross-slot block is refused."""
    import scipy.sparse
    import scipy.sparse.linalg
    rng = np.random.default_rng(8)
    n, frames = 4, 2
    layouts, dense_blocks = [], []
    for j in range(3):
        mass = rng.uniform(1, 2, n)
        a = rng.normal(size=(3 * n, 3 * n)) * 0.1
        dense_blocks.append(np.diag(np.repeat(mass, 3)) + a @ a.T)
        layouts.append(dict(dof_offset=3 * n * j, dof_count=3 * n, n=n, mass=mass, strength=10.0, anchor_idx=np.array([j])))
    full = scipy.linalg.block_diag(*dense_blocks)
    # exported form: upper block triangle of 3x3 blocks (row <= col)
    rows, cols, values = [], [], []
    nb = full.shape[0] // 3
    for r in range(nb):
        for c in range(r, nb):
            blk = full[3*r:3*r+3, 3*c:3*c+3]
            if np.any(blk):
                rows.append(r); cols.append(c); values.append(blk)
    rows, cols, values = np.array(rows), np.array(cols), np.array(values)
    global_lu = [scipy.sparse.linalg.splu(scipy.sparse.csc_matrix(full)) for _ in range(frames)]
    per_slot = [adj.slot_factorizations_from_triplets(rows, cols, values, layouts, workers=2) for _ in range(frames)]
    for j in range(3):
        expected = adj.tangent_pass(global_lu, layouts[j], return_frames=True)
        got = adj.tangent_pass([f[j][0] for f in per_slot], per_slot[0][j][1], return_frames=True)
        np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-13)
    with pytest.raises(ValueError, match="couples"):
        adj.slot_factorizations_from_triplets(np.append(rows, 0), np.append(cols, n), np.concatenate([values, np.eye(3)[None]]), layouts)
