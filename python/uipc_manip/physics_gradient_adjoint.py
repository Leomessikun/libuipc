"""Level 2a: the adjoint through the solver's own Hessians, checked against finite differences.

    PYTHONPATH=python python -m uipc_manip.physics_gradient_adjoint --garment tshirt_26 --body 14049 \\
        --state stall --out output/uipc_manip/physics_gradient_adjoint

No backend change: the engine's ``extras/debug/dump_linear_system`` switch writes the assembled
system of every Newton iteration to Matrix Market files. From a restored state one hold decision
(six frames) is run with the dump on; the last assembled Hessian of each frame is read back,
symmetrised (the file holds the upper block triangle), and the reverse pass

    H_f λ_f = ĝ_f,   ĝ_6 = ∂L/∂x_6,   ĝ_f = 2 M λ_{f+1} − M λ_{f+2}

follows the BDF1 inertia term x̃_f = 2 x_{f−1} − x_{f−2} + g dt², with M the lumped vertex masses.
The gripper enters through the soft position constraint E = ½ s m ‖x − aim‖² on the held vertices,
so ∂L/∂aim_f = s m λ_f there, and with aim_f = anchor_0 + (f/6) Δ the derivative per commanded
metre of translation is ∂L/∂Δ = Σ_f (f/6) Σ_I s m_I λ_{f,I}. Two objectives whose ∂L/∂x are known
exactly are used: the opening centroid's position along the upper arm (linear in x) and the
contact energy (its gradient is exported by the contact system). Each is compared with central
differences of the same quantity at 1 and 2 mm from the same restored state. What the comparison
cannot see: friction's lagged terms (the reverse pass ignores ∂G/∂x_prev outside inertia) and the
SPD projection of the assembled Hessian; a cosine below the gate is a measurement of exactly those.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import scipy.io
import scipy.sparse
import scipy.sparse.linalg

from uipc_manip import physics_gradient_probe as probe

ACCEPT_COSINE = 0.95


# ------------------------------------------------------------------------ dump plumbing
def enable_linear_system_dump():
    """Make every scene the coupler builds write its linear systems; read once at engine init."""
    from genesis.engine.couplers.ipc_coupler import coupler as coupler_module

    original = coupler_module.build_ipc_scene_config

    def build(options, sim_options):
        config = original(options, sim_options)
        config.setdefault("extras", {}).setdefault("debug", {})["dump_linear_system"] = 1
        return config

    coupler_module.build_ipc_scene_config = build


def workspace_of(env) -> Path:
    coupler = env.scene.sim.coupler
    uid = coupler.sim.scene.uid.full()
    return Path(tempfile.gettempdir()) / f"genesis_ipc_{uid}"


def dump_files(env) -> list[Path]:
    """Every Matrix Market file the engine has written under its workspace."""
    return list(workspace_of(env).rglob("*.mtx"))


def clear_dumps(env) -> None:
    for f in dump_files(env):
        f.unlink()


def dumped_frames(env) -> dict[int, dict[int, Path]]:
    """{frame: {newton_iter: path}} of the A matrices present."""
    out: dict[int, dict[int, Path]] = {}
    for path in dump_files(env):
        if not path.name.startswith("A."):
            continue
        _, frame, it = path.stem.split(".")
        out.setdefault(int(frame), {})[int(it)] = path
    return out


def dump_switch_value(env):
    """The scene's value of extras/debug/dump_linear_system, as the backend read it."""
    try:
        import uipc
        cfg = env.scene.sim.coupler._ipc_scene.config()
        attr = cfg.find("extras/debug/dump_linear_system")
        return None if attr is None else int(np.asarray(uipc.view(attr)).reshape(-1)[0])
    except Exception as exc:  # noqa: BLE001 - diagnostic only
        return f"unreadable: {exc!r}"


def read_symmetric(path: Path) -> tuple[scipy.sparse.csr_matrix, dict]:
    """The dump normally holds the upper block triangle (block row ≤ block col, diagonal blocks
    whole) under a 'general' header; if lower blocks are present the file is already full."""
    a = scipy.io.mmread(path).tocoo()
    r, c, v = a.row, a.col, a.data
    lower = int(np.count_nonzero((r // 3) > (c // 3)))
    upper = int(np.count_nonzero((r // 3) < (c // 3)))
    info = {"lower_block_entries": lower, "upper_block_entries": upper, "mirrored": lower == 0}
    if lower == 0:
        off = (r // 3) != (c // 3)
        a = scipy.sparse.coo_matrix(
            (np.concatenate([v, v[off]]), (np.concatenate([r, c[off]]), np.concatenate([c, r[off]]))), shape=a.shape
        )
    m = a.tocsr()
    info["asymmetry"] = float(abs(m - m.T).max() / abs(m).max())
    return m, info


# ------------------------------------------------------------------------ in-solver export (tree build)
def adjoint_feature(env):
    """The backend's LinearSystemAdjointFeature, or None on a build without it (the 0.0.28 wheel)."""
    try:
        from uipc.diff_sim import LinearSystemAdjointFeature
    except ImportError:
        return None
    return env._world.features().find(LinearSystemAdjointFeature)


def collect_via_feature(env, action, feature) -> tuple[dict, list]:
    """One decision whose every frame's assembled system is exported right after the frame."""
    systems = []
    original = env._sim_step

    def hooked():
        original()
        systems.append(feature.export_system())
    env._sim_step = hooked
    try:
        out = probe.decision(env, action)
    finally:
        env._sim_step = original
    return out, systems


def system_to_matrix(rows, cols, values, dofs: int) -> scipy.sparse.csr_matrix:
    """Scalar sparse matrix from the exported upper block triangle."""
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    values = np.asarray(values, dtype=np.float64).reshape(-1, 3, 3)
    t = rows.shape[0]
    i = (3 * rows[:, None, None] + np.arange(3)[None, :, None] + np.zeros((1, 1, 3), dtype=np.int64)).reshape(-1)
    j = (3 * cols[:, None, None] + np.zeros((1, 3, 1), dtype=np.int64) + np.arange(3)[None, None, :]).reshape(-1)
    v = values.reshape(-1)
    off = np.repeat(rows != cols, 9)
    m = scipy.sparse.coo_matrix((np.concatenate([v, v[off]]), (np.concatenate([i, j[off]]), np.concatenate([j, i[off]]))), shape=(dofs, dofs))
    return m.tocsr()


# ------------------------------------------------------------------------ cloth bookkeeping
def cloth_layout(env, constraint_factor: float = 1.0) -> dict:
    """Global DOF offset of the cloth, its global vertex offset, lumped vertex masses (rest shape)."""
    import uipc
    from uipc import builtin

    geo = env.slots[0].geometry()
    dof_offset = int(np.asarray(uipc.view(geo.meta().find(builtin.dof_offset))).reshape(-1)[0])
    dof_count = int(np.asarray(uipc.view(geo.meta().find(builtin.dof_count))).reshape(-1)[0])
    vertex_offset = int(np.asarray(uipc.view(geo.meta().find(builtin.global_vertex_offset))).reshape(-1)[0])
    cell = env.cells[0]
    x0, faces = np.asarray(cell.cloth, dtype=np.float64), np.asarray(cell.faces)
    n = x0.shape[0]
    area = 0.5 * np.linalg.norm(np.cross(x0[faces[:, 1]] - x0[faces[:, 0]], x0[faces[:, 2]] - x0[faces[:, 0]]), axis=1)
    mass = np.zeros(n)
    for k in range(3):
        np.add.at(mass, faces[:, k], area / 3.0)
    mass *= float(env.cfg.cloth_density) * float(env.cfg.cloth_thickness)
    return {"dof_offset": dof_offset, "dof_count": dof_count, "vertex_offset": vertex_offset, "n": n, "mass": mass,
            "strength": float(env.cfg.constraint_strength) * constraint_factor, "anchor_idx": np.asarray(env._pickers[0]["anchor_idx"]),
            "opening_idx": np.asarray(cell.opening_idx), "axis": (cell.shoulder - cell.elbow) / np.linalg.norm(cell.shoulder - cell.elbow)}


def contact_gradient(env, layout: dict) -> np.ndarray:
    """∂E_contact/∂x on the cloth vertices (3n), from the contact system's exported gradients."""
    from uipc import view
    from uipc.core import ContactSystemFeature
    from uipc.geometry import Geometry

    feature = env._world.features().find(ContactSystemFeature)
    if feature is None:
        raise RuntimeError("the backend exports no contact gradients")
    g = np.zeros((layout["n"], 3))
    lo, hi = layout["vertex_offset"], layout["vertex_offset"] + layout["n"]
    for prim in feature.contact_primitive_types():
        geom = Geometry()
        feature.contact_gradient(prim, geom)
        count = geom.instances().size()
        if count == 0:
            continue
        i = np.asarray(view(geom.instances().find("i")), dtype=np.int64).reshape(count)
        v = np.asarray(view(geom.instances().find("grad")), dtype=np.float64).reshape(count, 3)
        keep = (i >= lo) & (i < hi)
        np.add.at(g, i[keep] - lo, v[keep])
    return g.reshape(-1)


def axis_gradient(layout: dict) -> np.ndarray:
    """∂(upper-arm axis reading)/∂x: the opening centroid projected on the elbow→shoulder axis."""
    g = np.zeros((layout["n"], 3))
    g[layout["opening_idx"]] = layout["axis"][None, :] / len(layout["opening_idx"])
    return g.reshape(-1)


# ------------------------------------------------------------------------ the reverse pass
def reverse_pass(H: list, g_final: np.ndarray, layout: dict, chain: bool = True) -> dict:
    """λ_f for f = 1..6 (index 0..5) and ∂L/∂Δ per commanded metre of gripper translation; ``H`` holds
    one LU factorisation per frame."""
    off, cnt, n = layout["dof_offset"], layout["dof_count"], layout["n"]
    m3 = np.repeat(layout["mass"], 3)
    steps = len(H)
    lam = [None] * steps
    g_next = np.zeros(cnt)
    g_next2 = np.zeros(cnt)
    contributions = []
    for f in range(steps - 1, -1, -1):
        ghat = np.zeros(H[f].shape[0])
        if f == steps - 1:
            ghat[off:off + cnt] = g_final
        elif chain:
            ghat[off:off + cnt] = 2.0 * m3 * g_next - m3 * g_next2
        else:
            break
        sol = H[f].solve(ghat)
        lam[f] = sol[off:off + cnt]
        g_next2, g_next = g_next, lam[f]
        held = lam[f].reshape(n, 3)[layout["anchor_idx"]] * (layout["strength"] * layout["mass"][layout["anchor_idx"]])[:, None]
        contributions.append({"frame_index": f + 1, "dL_daim_sum": held.sum(axis=0).tolist(), "ratio": (f + 1) / steps})
    dL_dDelta = np.zeros(3)
    for c in contributions:
        dL_dDelta += c["ratio"] * np.asarray(c["dL_daim_sum"])
    return {"dL_dDelta": dL_dDelta, "contributions": contributions[::-1]}


def tangent_pass(lu: list, layout: dict, chain: bool = True) -> np.ndarray:
    """∂x_6/∂Δ for the three translation axes (3n × 3), propagated forward through the frames with
    the same inertia coupling the reverse pass uses; aim_f = anchor_0 + (f/6) Δ."""
    off, cnt, n = layout["dof_offset"], layout["dof_count"], layout["n"]
    m3 = np.repeat(layout["mass"], 3)
    held = (3 * layout["anchor_idx"][:, None] + np.arange(3)[None, :]).reshape(-1)
    s_m = np.repeat(layout["strength"] * layout["mass"][layout["anchor_idx"]], 3)
    steps = len(lu)
    out = np.zeros((cnt, 3))
    for k in range(3):
        dx_prev = np.zeros(cnt)
        dx_prev2 = np.zeros(cnt)
        for f in range(steps):
            rhs = np.zeros(lu[f].shape[0])
            local = np.zeros(cnt)
            if chain:
                local += 2.0 * m3 * dx_prev - m3 * dx_prev2
            daim = np.zeros(cnt)
            daim[held[k::3]] = (f + 1) / steps  # the k-th component of every held vertex's aim
            local += np.repeat(layout["strength"] * layout["mass"], 3) * daim
            rhs[off:off + cnt] = local
            dx = lu[f].solve(rhs)[off:off + cnt]
            dx_prev2, dx_prev = dx_prev, dx
        out[:, k] = dx_prev
    return out


def position_differences(env, snap: dict, eps_m: float) -> np.ndarray:
    """Central differences of every cloth vertex position (3n × 3) with respect to the commanded
    translation, from the restored state."""
    max_t = float(env.cfg.max_translation)
    cols = []
    for k in range(3):
        sides = []
        for sign in (+1.0, -1.0):
            action = np.zeros(6)
            action[k] = sign * eps_m / max_t
            probe.restore(env, snap)
            probe.decision(env, action)
            sides.append(env.positions()[0].reshape(-1).astype(np.float64))
        cols.append((sides[0] - sides[1]) / (2.0 * eps_m))
    return np.stack(cols, axis=1)


def compare_fields(tangent: np.ndarray, fd: np.ndarray, layout: dict) -> dict:
    """Cosine and magnitude of the predicted against the measured response, over all cloth DOFs,
    over the held vertices and over the rest."""
    held = np.zeros(layout["n"], dtype=bool)
    held[layout["anchor_idx"]] = True
    held3 = np.repeat(held, 3)
    out = {}
    for name, mask in (("all", np.ones_like(held3)), ("held", held3), ("free", ~held3)):
        row = {}
        for k, ax in enumerate("xyz"):
            a, b = tangent[mask, k], fd[mask, k]
            row[ax] = {"cosine": probe.cosine(a, b), "magnitude_ratio": float(np.linalg.norm(a) / np.linalg.norm(b)) if np.linalg.norm(b) > 0 else None,
                       "fd_rms_per_m": float(np.sqrt(np.mean(b * b))), "tangent_rms_per_m": float(np.sqrt(np.mean(a * a)))}
        out[name] = row
    # how far the held vertices follow the aim: the k-th component response to the k-th axis
    held_idx = (3 * layout["anchor_idx"][:, None] + np.arange(3)[None, :])
    out["held_follow"] = {"fd": [float(np.mean(fd[held_idx[:, k], k])) for k in range(3)],
                          "tangent": [float(np.mean(tangent[held_idx[:, k], k])) for k in range(3)]}
    return out


# ------------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--garment", default="tshirt_26")
    p.add_argument("--body", type=int, default=14049)
    p.add_argument("--region", type=int, default=13)
    p.add_argument("--state", default="stall", help="elbow, passed or stall")
    p.add_argument("--out", required=True)
    p.add_argument("--epsilons-mm", type=float, nargs="+", default=[1.0, 2.0])
    p.add_argument("--stall-window", type=int, default=15)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--check-only", action="store_true", help="Build, take one decision, report where the dump landed.")
    p.add_argument("--constraint-factor", type=float, default=2.0,
                   help="The soft position constraint's assembled stiffness as a multiple of strength × mass "
                        "(2 for an energy without the one-half; the held vertices' measured response settles it).")
    p.add_argument("--keep-last-matrix", action="store_true", help="Copy the last frame's A file next to the record.")
    p.add_argument("--source", choices=("dump", "feature"), default="dump",
                   help="Where the Hessians come from: the engine's debug dump (any build) or the "
                        "LinearSystemAdjointFeature export (this tree's build).")
    args = p.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.source == "dump":
        enable_linear_system_dump()
    t0 = time.time()
    env = probe.build_env(args.garment, args.body, args.region, out / "work")
    record = {"garment": args.garment, "body": args.body, "state": args.state, "build_s": time.time() - t0,
              "dt": float(env.cfg.dt), "action_repeat": int(env.cfg.action_repeat), "workspace": str(workspace_of(env))}
    try:
        if args.check_only:
            env.reset([0])
            probe.decision(env, np.zeros(6))
            files = dump_files(env)
            print(f"[adjoint-check] switch={dump_switch_value(env)} workspace={workspace_of(env)} "
                  f"files={len(files)} first={[str(f) for f in files[:3]]} frame={env._world.frame()}", flush=True)
            return
        # Drive with the dump on, discarding the files after every decision.
        original_decision = probe.decision

        def decision_discarding(env_, action):
            out_ = original_decision(env_, action)
            clear_dumps(env_)
            return out_
        probe.decision = decision_discarding
        snaps, trace = probe.drive_to_snapshots(env, args.stall_window, 0.01, args.max_steps)
        probe.decision = original_decision
        snap = next(s for s in snaps if s["name"] == args.state)
        record["episode_step"] = snap["episode_step"]
        record["state_measure"] = snap["measure"]
        layout = cloth_layout(env, args.constraint_factor)
        record["layout"] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in layout.items() if k not in ("mass",)}
        record["layout"]["mass_sum"] = float(layout["mass"].sum())

        # One hold decision whose six systems are kept: dumped files, or the feature's export.
        probe.restore(env, snap)
        frame_before = int(env._world.frame())
        H, lu = [], []
        record["source"] = args.source
        if args.source == "feature":
            feature = adjoint_feature(env)
            if feature is None:
                raise RuntimeError("this build has no LinearSystemAdjointFeature; use --source dump")
            t1 = time.time()
            hold, systems = collect_via_feature(env, np.zeros(6), feature)
            record["export_s"] = time.time() - t1
            dofs = int(feature.dof_count())
            if len(systems) != int(env.cfg.action_repeat):
                raise RuntimeError(f"expected {env.cfg.action_repeat} exported systems, got {len(systems)}")
            for rows, cols, values, gradient in systems:
                H.append(system_to_matrix(rows, cols, values, dofs))
            record["frames"] = [{"frame": frame_before + 1 + k, "triplets": int(len(s[0]))} for k, s in enumerate(systems)]
            # The backend's own solve against the last frame, checked against the host factorisation below.
            g_probe = np.zeros(dofs)
            rng = np.random.default_rng(0)
            g_probe[:] = rng.standard_normal(dofs)
            t1 = time.time()
            x_backend, reached = feature.solve(g_probe, 1e-6, 32)
            x_backend = np.asarray(x_backend)
            record["feature_solve_s"] = time.time() - t1
            record["feature_solve_check"] = {"g": g_probe, "x": x_backend, "reached": float(reached)}
        else:
            clear_dumps(env)
            hold = probe.decision(env, np.zeros(6))
            frames = dumped_frames(env)
            wanted = [f for f in sorted(frames) if frame_before < f <= frame_before + int(env.cfg.action_repeat)]
            if len(wanted) != int(env.cfg.action_repeat):
                raise RuntimeError(f"expected {env.cfg.action_repeat} dumped frames after {frame_before}, found {sorted(frames)}")
            record["frames"] = [{"frame": f, "newton_iterations": max(frames[f]) + 1} for f in wanted]
            t1 = time.time()
            for f in wanted:
                h, info = read_symmetric(frames[f][max(frames[f])])
                H.append(h)
                record.setdefault("matrix_files", []).append(info)
            record["read_s"] = time.time() - t1
            if args.keep_last_matrix:
                shutil.copy(frames[wanted[-1]][max(frames[wanted[-1]])], out / f"{args.garment}_{args.body}.{args.state}.A.mtx")
        record["dofs"] = int(H[-1].shape[0])
        record["nnz"] = int(H[-1].nnz)
        t1 = time.time()
        lu = [scipy.sparse.linalg.splu(h.tocsc()) for h in H]
        record["factorise_s"] = time.time() - t1
        if "feature_solve_check" in record:
            chk = record["feature_solve_check"]
            x_host = lu[-1].solve(chk["g"])
            residual = float(np.linalg.norm(H[-1] @ chk["x"] - chk["g"]) / np.linalg.norm(chk["g"]))
            record["feature_solve_check"] = {"cosine_vs_host_lu": probe.cosine(chk["x"], x_host), "backend_reported_residual": chk["reached"],
                                             "relative_error_vs_host_lu": float(np.linalg.norm(chk["x"] - x_host) / np.linalg.norm(x_host)),
                                             "relative_residual": residual, "solve_s": record.pop("feature_solve_s")}
        # A sanity read of the held rows: with s = 1e4 the constraint dominates the diagonal block.
        diag = H[-1].diagonal()
        held_dofs = (layout["dof_offset"] + 3 * layout["anchor_idx"][:, None] + np.arange(3)[None, :]).reshape(-1)
        m_from_H = diag[held_dofs].reshape(-1, 3).mean(axis=1) / (1.0 + layout["strength"])
        record["held_mass_check"] = {"lumped": layout["mass"][layout["anchor_idx"]].tolist(), "from_diagonal": m_from_H.tolist()}

        objectives = {"upperarm_axis_m": axis_gradient(layout), "contact_energy": contact_gradient(env, layout)}
        record["objectives"] = {}
        for name, g in objectives.items():
            t2 = time.time()
            full = reverse_pass(lu, g, layout, chain=True)
            single = reverse_pass(lu, g, layout, chain=False)
            record["objectives"][name] = {"adjoint_chain": full["dL_dDelta"].tolist(), "adjoint_last_frame": single["dL_dDelta"].tolist(),
                                          "contributions": full["contributions"], "solve_s": time.time() - t2,
                                          "g_norm": float(np.linalg.norm(g))}

        # The full response field, predicted forward through the frames and measured by differences.
        t3 = time.time()
        tangent = tangent_pass(lu, layout, chain=True)
        tangent_last = tangent_pass(lu, layout, chain=False)
        record["tangent_s"] = time.time() - t3
        fd_field = position_differences(env, snap, args.epsilons_mm[0] * 1e-3)
        record["field_comparison"] = {"chain": compare_fields(tangent, fd_field, layout), "last_frame": compare_fields(tangent_last, fd_field, layout),
                                      "epsilon_mm": args.epsilons_mm[0]}
        # The objectives through the tangent field must agree with the reverse pass (same linear algebra).
        record["tangent_objectives"] = {name: (g @ tangent).tolist() for name, g in objectives.items()}
        record["fd_field_objectives"] = {name: (g @ fd_field).tolist() for name, g in objectives.items()}

        # Finite differences of the same quantities from the same restored state.
        fd = probe.gradients(env, snap, np.zeros(6), [e * 1e-3 for e in args.epsilons_mm], 1)
        record["finite_differences"] = {str(e): {k: fd["per_epsilon"][str(e * 1e-3)]["mean"][k] for k in ("upperarm_axis_m", "contact_energy", "upperarm_ratio")}
                                        for e in args.epsilons_mm}
        record["fd_executed"] = {str(e): fd["per_epsilon"][str(e * 1e-3)]["mean_executed"] for e in args.epsilons_mm}
        record["comparison"] = {}
        for name in objectives:
            row = {}
            for e in args.epsilons_mm:
                fd_vec = np.asarray(record["finite_differences"][str(e)][name], dtype=np.float64)
                for kind in ("adjoint_chain", "adjoint_last_frame"):
                    adj = np.asarray(record["objectives"][name][kind])
                    row[f"{kind}_vs_fd_{e:g}mm"] = {"cosine": probe.cosine(adj, fd_vec),
                                                     "magnitude_ratio": float(np.linalg.norm(adj) / np.linalg.norm(fd_vec)) if np.linalg.norm(fd_vec) > 0 else None}
            record["comparison"][name] = row
        record["accepted"] = all(record["comparison"][n][f"adjoint_chain_vs_fd_{args.epsilons_mm[0]:g}mm"]["cosine"] >= ACCEPT_COSINE
                                 for n in objectives)
        if "feature_solve_check" in record:
            print(f"[adjoint-feature] export {record['export_s']:.2f}s for {len(H)} frames; backend solve vs host LU: "
                  f"cos={record['feature_solve_check']['cosine_vs_host_lu']:.6f} rel_err={record['feature_solve_check']['relative_error_vs_host_lu']:.2e} "
                  f"residual={record['feature_solve_check']['relative_residual']:.2e} backend_reported={record['feature_solve_check']['backend_reported_residual']:.2e} "
                  f"({record['feature_solve_check']['solve_s']*1e3:.0f} ms)", flush=True)
        print(f"[adjoint] {args.garment}/{args.body} {args.state}@{snap['episode_step']} dofs={record['dofs']} nnz={record['nnz']} "
              f"frames={[f.get('newton_iterations', f.get('triplets')) for f in record['frames']]} "
              + " ".join(f"{n}: chain={np.round(record['objectives'][n]['adjoint_chain'], 4).tolist()} "
                         f"fd1mm={np.round(record['finite_differences'][str(args.epsilons_mm[0])][n], 4).tolist()} "
                         f"cos={ {k: round(v['cosine'], 3) for k, v in record['comparison'][n].items()} }" for n in objectives)
              + f" accepted={record['accepted']}", flush=True)
        fc = record["field_comparison"]["chain"]
        print(f"[adjoint-field] cos all={[round(fc['all'][a]['cosine'], 3) for a in 'xyz']} held={[round(fc['held'][a]['cosine'], 3) for a in 'xyz']} "
              f"free={[round(fc['free'][a]['cosine'], 3) for a in 'xyz']} | magnitude ratio all={[round(fc['all'][a]['magnitude_ratio'], 3) for a in 'xyz']} "
              f"held={[round(fc['held'][a]['magnitude_ratio'], 3) for a in 'xyz']} free={[round(fc['free'][a]['magnitude_ratio'], 3) for a in 'xyz']} "
              f"| held follow fd={np.round(fc['held_follow']['fd'], 3).tolist()} tangent={np.round(fc['held_follow']['tangent'], 3).tolist()}", flush=True)
    finally:
        if args.source == "dump":
            clear_dumps(env)
        suffix = "" if args.source == "dump" else ".feature"
        (out / f"{args.garment}_{args.body}.{args.state}{suffix}.json").write_text(json.dumps(record, indent=1, default=float) + "\n")
        env.close()


if __name__ == "__main__":
    main()
