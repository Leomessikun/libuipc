"""Record a policy's complete garment geometry per decision, in the audit's saved format.

Writes ``positions_<slot>.npz`` and ``static_<slot>.npz`` so that
``scripts/audit_progress_metric.py`` can recompute the progress metric and compare it
with tests that do not depend on the opening's four fan triangles. Nothing is trained
and no solver, reward or controller setting is changed.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from uipc_manip import train_sac  # noqa: E402
from uipc_manip.dressing_env import GenesisIPCDressingEnv  # noqa: E402
from uipc_manip.dressing_live import LiveCellFactory  # noqa: E402
from uipc_manip.physics_gradient_actor import load_agent  # noqa: E402
from uipc_manip.sac import SACAgent  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cell", default="tshirt_392:14046")
    p.add_argument("--slots", type=int, default=2)
    p.add_argument("--seed", type=int, default=6601)
    p.add_argument("--decisions", type=int, default=300)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    payload = SACAgent.read_checkpoint(args.checkpoint)
    targs = train_sac.build_parser().parse_args(["--eval-only"])
    train_sac.restore_resume_args(targs, ["--eval-only"], payload)
    train_sac.resolve_defaults(targs)
    garment, body = args.cell.rsplit(":", 1)
    cfg = replace(train_sac.dressing_config(targs), cells=((garment, int(body)),) * args.slots,
                  contact_force_readout=False, decision_watchdog=False, workspace=str(args.out / "assets"))
    args.out.mkdir(parents=True)
    started = time.perf_counter()
    env = GenesisIPCDressingEnv(cfg, num_envs=args.slots, cell_factory=LiveCellFactory(cfg.live))
    try:
        agent = load_agent(args.checkpoint, env.spec.point_budget, env.action_dim, "cuda")
        seeds = [args.seed + i for i in range(args.slots)]
        obs = env.reset(seeds)
        frames: list[list[np.ndarray]] = [[p.copy()] for p in env.positions()]
        trace: list[list[dict]] = [[] for _ in range(args.slots)]
        for _ in range(args.decisions):
            obs, _, done, infos = env.step(agent.act(obs, deterministic=True), reset_on_done=False)
            if any(r.get("sim_error") for r in infos):
                raise RuntimeError("Simulator failure during the capture")
            for i, p in enumerate(env.positions()):
                frames[i].append(p.copy())
                trace[i].append({k: float(infos[i][k]) for k in ("upperarm_ratio", "forearm_ratio", "tracking_error")})
        for i, cell in enumerate(env.cells):
            np.savez_compressed(args.out / f"positions_{i}.npz", positions=np.stack(frames[i]).astype(np.float32))
            np.savez_compressed(args.out / f"static_{i}.npz", faces=cell.faces, opening_idx=cell.opening_idx,
                                opening_triangles=cell.polygon_triangles(), arm_vertices=cell.arm_points,
                                arm_faces=cell.arm_faces, finger=cell.finger, elbow=cell.elbow,
                                shoulder=cell.shoulder)
        (args.out / "result.json").write_text(json.dumps(dict(
            checkpoint=str(args.checkpoint), cell=args.cell, seeds=seeds, decisions=args.decisions,
            seconds=time.perf_counter() - started, trace=trace, completed=True), indent=1) + "\n")
        print(json.dumps(dict(slots=args.slots, decisions=args.decisions,
                              final=[t[-1]["upperarm_ratio"] for t in trace],
                              seconds=round(time.perf_counter() - started, 1))), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
