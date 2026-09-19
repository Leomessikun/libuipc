"""Once the sleeve is on the upper arm, does it stay there if the tool stops?

Four of the scripted teacher's twenty-five cells reach an upper-arm coverage of
0.997-1.000 and end the episode at 0.000, three of them with the grip held
comfortably inside the 2 cm rule the whole time (`2026-09-19-what-actually-fails.md`).
They fail by not stopping. Whether that is worth anything depends on a question nobody
has asked here: if the controller simply stops commanding once the coverage threshold
is met, does the coverage hold to the end of the episode, or does the sleeve come off
regardless?

An answer of "it holds" means a ceiling of fifteen cells out of twenty-five is really
available to any controller that knows when to stop, against the teacher's six and
against zero to three for every learner tried. An answer of "it comes off" means the
metric's end-of-episode reading is measuring something no controller can keep, and the
benchmark needs saying so.

The teacher drives; the only change is that a slot whose coverage crosses the threshold
stops commanding, and stays stopped.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from uipc_manip import train_sac  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True, help="Supplies the environment contract only")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cells", default="tshirt_4:14047,tshirt_26:14045,tshirt_26:14048,tshirt_68:14049",
                   help="The cells that reach the threshold and lose it")
    p.add_argument("--threshold", type=float, default=0.7)
    p.add_argument("--arm", choices=("stop", "drive"), default="stop",
                   help="stop: freeze the command once covered; drive: the teacher's own control, as a reference")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    cells = tuple((c.rsplit(":", 1)[0], int(c.rsplit(":", 1)[1])) for c in args.cells.split(",") if c.strip())
    from uipc_manip.dressing_env import GenesisIPCDressingEnv
    from uipc_manip.dressing_live import LiveCellFactory
    from uipc_manip.sac import SACAgent

    payload = SACAgent.read_checkpoint(args.checkpoint)
    targs = train_sac.build_parser().parse_args(["--eval-only"])
    train_sac.restore_resume_args(targs, ["--eval-only"], payload)
    train_sac.resolve_defaults(targs)
    base = train_sac.dressing_config(targs)
    cfg = replace(base, cells=cells, decision_watchdog=False, contact_force_readout=False,
                  workspace=str(args.out / "assets"))
    args.out.mkdir(parents=True)
    started = time.perf_counter()
    env = GenesisIPCDressingEnv(cfg, num_envs=len(cells), cell_factory=LiveCellFactory(cfg.live))
    result = dict(completed=False, arm=args.arm, threshold=args.threshold, env=cfg.to_dict(),
                  cells=[list(c) for c in cells], traces=[[] for _ in cells], records=[])

    def save():
        result["seconds"] = time.perf_counter() - started
        (args.out / "result.json").write_text(json.dumps(result, indent=1) + "\n")

    try:
        env.reset([args.seed + i for i in range(len(cells))])
        frozen = np.zeros(len(cells), dtype=bool)
        crossed_at = np.full(len(cells), -1, dtype=int)
        for step in range(cfg.horizon):
            actions = np.asarray(env.scripted_actions(), dtype=np.float64)
            if args.arm == "stop":
                actions[frozen] = 0.0
            _, _, done, info = env.step(np.clip(actions, -1, 1).astype(np.float32), reset_on_done=False)
            if any(r.get("sim_error") for r in info):
                raise RuntimeError(f"Simulator failure at decision {step}")
            for i, row in enumerate(info):
                result["traces"][i].append(dict(upperarm=float(row["upperarm_ratio"]),
                                                tracking=float(row["tracking_error"])))
                if not frozen[i] and row["upperarm_ratio"] >= args.threshold:
                    frozen[i] = True
                    crossed_at[i] = step
            if np.all(done):
                break
            if step % 50 == 0:
                save()
        for i, (garment, body) in enumerate(cells):
            upper = np.asarray([r["upperarm"] for r in result["traces"][i]])
            tracking = np.asarray([r["tracking"] for r in result["traces"][i]])
            result["records"].append(dict(
                cell=f"{garment}/{body}", crossed_at=int(crossed_at[i]),
                peak=float(upper.max()), final=float(upper[-1]),
                sustained_last12=float(upper[-12:].min()),
                worst_tracking_cm=float(tracking.max() * 100),
                grasp_held=bool(tracking.max() <= 0.02),
                success=bool(upper[-12:].min() >= args.threshold and tracking.max() <= 0.02)))
        result["completed"] = True
        save()
        print(json.dumps(dict(arm=args.arm, successes=sum(r["success"] for r in result["records"]),
                              of=len(cells), records=result["records"]), indent=1), flush=True)
    finally:
        env.close()
        gc.collect()


if __name__ == "__main__":
    main()
