"""Is the good action a basin in the prior's noise space, or a needle?

The third measurement of the prior-support audit. The first two ask where an action
sits in the prior's noise distribution and whether the flow can reproduce it; this one
asks what happens *around* that noise. A latent policy — behaviour cloning in noise
space, or reinforcement learning whose actions are noise — can only find a behaviour
whose neighbourhood also works.

At each state the world is snapshotted, the reference action is reversed to its noise,
and for a set of perturbation scales the perturbed noise is decoded back to an action
and executed for a short window before the base policy resumes, exactly as the
counterfactual branch study executed its macros. What is reported per scale is the
outcome, against the unperturbed noise at the same state and against the policy's own
continuation.
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

from uipc_manip import decision_branches as db  # noqa: E402
from uipc_manip import flow_reversal as fr  # noqa: E402
from uipc_manip import physics_gradient_probe as probe  # noqa: E402
from uipc_manip import train_sac  # noqa: E402

BEST_MACRO = {"tshirt_26": "lift", "tshirt_392": "forward", "tshirt_68": "forward",
              "hospital_gown": "forward", "tshirt_4": "forward"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prior", type=Path, required=True, help="An FQL checkpoint whose behavior flow decodes noise")
    p.add_argument("--checkpoint", type=Path, required=True, help="The SAC policy that drives and resumes")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cell", default="tshirt_26:14046")
    p.add_argument("--snapshot-steps", default="100,140,180")
    p.add_argument("--window", type=int, default=8)
    p.add_argument("--follow", type=int, default=32)
    p.add_argument("--slots", type=int, default=8)
    p.add_argument("--seed", type=int, default=3301)
    p.add_argument("--sigmas", default="0,0.25,0.5,1.0,2.0")
    p.add_argument("--samples", type=int, default=2, help="Perturbed draws per scale")
    p.add_argument("--step-m", type=float, default=0.008)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    steps = sorted(int(s) for s in args.snapshot_steps.split(","))
    sigmas = [float(s) for s in args.sigmas.split(",") if s.strip()]
    from uipc_manip.dressing_env import GenesisIPCDressingEnv
    from uipc_manip.dressing_live import LiveCellFactory
    from uipc_manip.fql import FQLAgent
    from uipc_manip.physics_gradient_actor import load_agent
    from uipc_manip.sac import SACAgent

    payload = SACAgent.read_checkpoint(args.checkpoint)
    targs = train_sac.build_parser().parse_args(["--eval-only"])
    train_sac.restore_resume_args(targs, ["--eval-only"], payload)
    train_sac.resolve_defaults(targs)
    garment, body = args.cell.rsplit(":", 1)
    cfg = replace(train_sac.dressing_config(targs), cells=((garment, int(body)),) * args.slots,
                  contact_force_readout=False, decision_watchdog=False, workspace=str(args.out / "assets"))
    if steps[-1] + args.window + args.follow >= cfg.horizon:
        raise ValueError("The last branch must end before the episode's time limit")
    args.out.mkdir(parents=True)
    started = time.perf_counter()
    env = GenesisIPCDressingEnv(cfg, num_envs=args.slots, cell_factory=LiveCellFactory(cfg.live))
    result = dict(completed=False, prior=str(args.prior), checkpoint=str(args.checkpoint), cell=args.cell,
                  env=cfg.to_dict(), sigmas=sigmas, samples=args.samples, window=args.window,
                  follow=args.follow, macro=BEST_MACRO.get(garment, "forward"), records=[], physical_decisions=0)

    def save():
        result["seconds"] = time.perf_counter() - started
        (args.out / "result.json").write_text(json.dumps(result, indent=1) + "\n")

    try:
        prior, _ = FQLAgent.load(args.prior, device="cuda")
        agent = load_agent(args.checkpoint, env.spec.point_budget, env.action_dim, "cuda")
        macro = next(m for m in db.MACROS if m["name"] == result["macro"])
        rng = np.random.default_rng(args.seed)
        obs = env.reset([args.seed + i for i in range(args.slots)])
        step = 0
        for target in steps:
            while step < target:
                obs, _, done, info = env.step(agent.act(obs, deterministic=True))
                result["physical_decisions"] += args.slots
                step += 1
                if np.any(done) or any(r.get("sim_error") for r in info):
                    raise RuntimeError(f"Approach terminated at {step}")
            snapshot = probe.take_snapshot(env, f"step{target}", target)
            directions = [db.slot_directions(env._anchor[i], c.finger, c.elbow, c.shoulder)
                          for i, c in enumerate(env.cells)]
            reference = np.stack([db.macro_action(macro, 0, 1, directions[i], np.zeros(6),
                                                  args.step_m, cfg.max_translation) for i in range(args.slots)])
            encoded = fr.encode(prior, obs)
            base_noise = fr.reverse_flow(prior, encoded, reference).cpu().numpy()
            reconstructed = fr.forward_flow(prior, encoded, base_noise).cpu().numpy()
            for sigma in sigmas:
                for sample in range(1 if sigma == 0 else args.samples):
                    noise = base_noise + (sigma * rng.standard_normal(base_noise.shape) if sigma else 0.0)
                    error = probe.restore(env, snapshot)
                    if error > 1e-5:
                        raise RuntimeError(f"Restore error {error} m")
                    branch = env.observation()
                    traces: list[list[dict]] = [[] for _ in range(args.slots)]
                    for t in range(args.window + args.follow):
                        if t < args.window:
                            action = fr.forward_flow(prior, fr.encode(prior, branch), noise).cpu().numpy()
                        else:
                            action = agent.act(branch, deterministic=True)
                        branch, _, done, rows = env.step(np.clip(action, -1, 1).astype(np.float32),
                                                        reset_on_done=False)
                        result["physical_decisions"] += args.slots
                        if np.any(done) or any(r.get("sim_error") for r in rows):
                            raise RuntimeError("Branch terminated early")
                        for i in range(args.slots):
                            traces[i].append({k: rows[i][k] for k in
                                              ("upperarm_ratio", "forearm_ratio", "tracking_error", "grasp_valid",
                                               "collision_rejected_substeps", "tether_rejected_substeps",
                                               "commanded_translation_m", "accepted_anchor_translation_m")})
                    for i in range(args.slots):
                        result["records"].append(dict(
                            step=target, slot=i, sigma=sigma, sample=sample,
                            latent_norm=float(np.linalg.norm(base_noise[i])),
                            reconstruction=float(np.linalg.norm(reconstructed[i] - reference[i])),
                            decoded_distance=float(np.linalg.norm(
                                fr.forward_flow(prior, fr.encode(prior, branch), noise).cpu().numpy()[i] - reference[i])),
                            **db.branch_summary(traces[i], min(12, args.window + args.follow))))
                    print(json.dumps(dict(step=target, sigma=sigma, sample=sample,
                                          sustained=[round(r["sustained_coverage"], 3)
                                                     for r in result["records"][-args.slots:]])), flush=True)
                    save()
            probe.restore(env, snapshot)
            obs = env.observation()
            step = target
        result["completed"] = True
        save()
    finally:
        env.close()
        gc.collect()


if __name__ == "__main__":
    main()
