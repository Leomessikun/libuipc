"""Matched GRAB motion pilot; this is not a full Dressing in Motion reproduction.

One fresh process runs one motion/onset and resets the same settled scene for
each controller. The oracle and shifted pause schedules are diagnostic only.
All failures, including failed builds, are retained in a new output directory.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import runpy
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
from uipc_manip.grab_motion import BodyMotion, sha256
from uipc_manip.motion_controls import bounded_correction, gicp, observed_arm_roi, pause_masks

WORKSPACE = Path("/home/ge47gax/kun")
METHODS = ("r1", "gicp", "oracle_pause", "causal_pause", "yoked_pause", "hold")


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--motion", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS[:-1]))
    p.add_argument("--checkpoint", type=Path, default=WORKSPACE / "libuipc/output/uipc_manip/fmvp_ipc_dagger_r1_20260924/model/fmvp_ipc_bc.pt")
    p.add_argument("--policy-package-root", type=Path, default=WORKSPACE / "libuipc/.claude/worktrees/residual-rl/python")
    p.add_argument("--hang", type=Path, default=WORKSPACE / "libuipc/output/uipc_manip/fmvp_better_rollouts_20260923/hang2.npz")
    p.add_argument("--hang-key", default="k300")
    p.add_argument("--onset", type=float, default=1., help="Seconds after reset; choose a later onset for contact-phase trials")
    p.add_argument("--motion-speed", type=float, default=1.)
    p.add_argument("--steps", type=int, default=350, help="Same total decision budget for every controller, including pauses")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--success", type=float, default=.7)
    p.add_argument("--hold", type=int, default=20, help="Consecutive successful states required; trials still use the full budget")
    p.add_argument("--future-horizon", type=float, default=.4)
    p.add_argument("--pause-displacement", type=float, default=.02)
    p.add_argument("--yoked-shift", type=int, default=20)
    p.add_argument("--correction-bound", type=float, default=.01)
    p.add_argument("--yaw", type=float, default=267.)
    p.add_argument("--obs-mode", choices=("wang_live_arm", "wang_static_arm"), default="wang_live_arm")
    p.add_argument("--tracking-tolerance", type=float, default=.002)
    p.add_argument("--drive-strength", type=float, default=1e6)
    p.add_argument("--preflight-only", action="store_true", help="CPU asset, placement, schedule and checkpoint checks; no physics")
    p.add_argument("--wait-for-gpu", type=float, default=0., help="Maximum seconds to wait for other compute jobs; never stops them")
    return p


def idle_gpu(timeout_s):
    """Do not start while other non-desktop CUDA clients are using the device."""
    deadline = time.monotonic() + timeout_s
    while True:
        result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
                                check=True, text=True, capture_output=True)
        busy = [line for line in result.stdout.splitlines()
                if not any(name in line for name in ("gnome-remote-desktop", "nvidia-cuda-mps-server", "/soffice.bin"))]
        if not busy:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError("GPU is in use; retry after the other jobs finish: " + "; ".join(busy))
        print("[gpu] waiting for existing jobs: " + "; ".join(busy), flush=True)
        time.sleep(min(20., max(0., deadline - time.monotonic())))


def prepare_cell(motion, live, hang, yaw):
    from uipc_manip.dressing_assets import DressingCell
    from uipc_manip.dressing_body import submesh, _reward_line_landmarks
    from uipc_manip.dressing_live import garment_arm_gap, load_offline_drape

    # Read the existing garment topology/semantic indices without starting a bake.
    drape = load_offline_drape("tshirt_26", live)
    if drape is None:
        raise FileNotFoundError("The tshirt_26 canonical drape and exported mesh are required")
    if hang.shape != drape["cloth"].shape or not np.isfinite(hang).all():
        raise ValueError("Hang does not match tshirt_26 topology or contains non-finite coordinates")
    body = motion.vertices[0].copy()
    landmarks = _reward_line_landmarks(body, motion.joints[0])
    arm, arm_faces = submesh(body, motion.faces, motion.arm_indices)
    c, s = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
    rotation = np.array([[1., 0, 0], [0, 0, 1.], [0, -1., 0]]) @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])
    finger = landmarks["right_finger"]
    target = finger + np.array([-.033, .111, -.003]) @ rotation
    best = None
    for degrees in range(0, 360, 5):
        c, s = np.cos(np.radians(degrees)), np.sin(np.radians(degrees))
        turn = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])
        cloth = hang @ turn.T + target
        gap = garment_arm_gap(cloth, drape["faces"], body, motion.faces)
        if gap < .003:
            continue
        opening = (cloth[drape["opening_idx"]].mean(0) - finger) @ rotation.T
        error = float(np.linalg.norm(opening - np.array([-.085, -.135, -.05])))
        if best is None or error < best[0]:
            best = error, degrees, gap, opening, cloth
    if best is None:
        raise RuntimeError("No legal full-body gravity-hung start for the selected body")
    error, degrees, gap, opening, cloth = best
    cell = DressingCell(garment="tshirt_26", human=int(motion.metadata["body_id"]),
                        cloth=cloth, faces=drape["faces"], grasp_idx=drape["grasp_idx"],
                        picker_idx=drape["picker_idx"], opening_idx=drape["opening_idx"],
                        alignment_idx=drape["alignment_idx"], picker_pos=target,
                        arm_points=arm, arm_faces=arm_faces, human_points=body,
                        landmarks=landmarks, pull_waypoints=np.stack([target, landmarks["right_shoulder"]]),
                        opening_radius_mean=float(drape["opening_radius_mean_m"]))
    return cell, dict(turn_degrees=degrees, gap_m=float(gap), opening_model=opening.tolist(),
                      opening_error_m=error, placement_offset_mm=[0., 5., 0.])


class PreparedFactory:
    def __init__(self, cfg, cell, placement):
        self.cfg, self.cell, self.placement = cfg, cell, placement

    def build(self, garment, human):
        if garment != self.cell.garment or human != self.cell.human:
            raise ValueError("Prepared cell does not match requested garment/body")
        return deepcopy(self.cell)

    def clearances(self, garment, human):
        return dict(self.placement)


def run(args):
    from uipc_manip import pretrain_wang, train_sac
    from uipc_manip.dressing_motion import MotionDressingEnv
    from uipc_manip.obs import FLAG_MARKER
    from physical_sleeve import DEFAULT_OBJ, SleeveSections, measure, read_obj

    motion = BodyMotion.load(args.motion)
    _, training, _ = pretrain_wang.prepare(["teacher", "--region", "13", "--seed", "1",
                                           "--obs-mode", "wang_static_arm", "--no-obs-augment"])
    cfg = train_sac.dressing_config(training)
    live = replace(cfg.live, body=replace(cfg.live.body, device="cpu"))
    cfg = replace(cfg, cells=(("tshirt_26", int(motion.metadata["body_id"])),), cell_source="live",
                  collision_geometry="full_body", live=live, horizon=args.steps + 1, seed=1,
                  decision_watchdog=False, clip_rotation_to_yz=False, anchor_count=48,
                  cloth_density=750., cloth_strain_rate=10., contact_force_readout=False,
                  workspace=str(args.out / "world"),
                  obs=replace(cfg.obs, mode=args.obs_mode, static_arm=False))
    with np.load(args.hang, allow_pickle=False) as data:
        hang = data[args.hang_key].copy()
    cell, placement = prepare_cell(motion, live, hang, args.yaw)
    rest, faces = read_obj(DEFAULT_OBJ)
    if not np.array_equal(faces, cell.faces):
        raise ValueError("Physical sleeve topology differs from the simulated garment")
    sleeve = SleeveSections(rest * 4., faces, cell.opening_idx)
    dt = cfg.dt * cfg.action_repeat
    masks = pause_masks(motion, np.arange(args.steps) * dt, onset_s=args.onset,
                        horizon_s=args.future_horizon, threshold_m=args.pause_displacement,
                        speed=args.motion_speed, yoked_shift=args.yoked_shift)
    code_paths = [ROOT / "python/uipc_manip" / name for name in
                  ("dressing_env.py", "dressing_obs.py", "dressing_motion.py", "grab_motion.py", "motion_controls.py")]
    metadata = dict(arguments=vars(args), checkpoint_sha256=sha256(args.checkpoint),
                    hang_sha256=sha256(args.hang), motion_sha256=sha256(args.motion), motion=motion.metadata,
                    code_sha256={str(p.relative_to(ROOT)): sha256(p) for p in [Path(__file__), *code_paths]},
                    bridge_sha256=sha256(args.policy_package_root / "uipc_manip/wang_bridge.py"),
                    client_sha256=sha256(args.policy_package_root / "uipc_manip/wang_client.py"),
                    revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    config=cfg.to_dict(), placement=placement, decision_dt_s=dt,
                    pause_counts={k: int(v.sum()) for k, v in masks.items()},
                    observations="obs[t] -> actions[t] -> obs[t+1]; state arrays have one extra frame",
                    registration="covariance-weighted nearest-neighbor GICP, observed spatial ROI, bounded point displacement; proxy only",
                    force_input="zero for all controllers", scope="single sleeve, one known garment; no garment-generalization claim",
                    success_rule="wrapped physical sleeve and upperarm ratio >= threshold for hold consecutive decisions, valid grasp throughout",
                    failure_policy="retain failed runs; no success filtering; simulation failure is invalid physics, not task failure")
    save_json(args.out / "run.json", metadata)
    np.savez_compressed(args.out / "schedules.npz", times=np.arange(args.steps) * dt, **masks)
    print("[preflight] " + json.dumps(dict(placement=placement, pause_counts=metadata["pause_counts"], decision_dt_s=dt)), flush=True)
    if args.preflight_only:
        return
    idle_gpu(args.wait_for_gpu)
    client = None
    env = None
    records = []
    try:
        if any(method != "hold" for method in args.methods):
            client_cls = runpy.run_path(str(args.policy_package_root / "uipc_manip/wang_client.py"))["WangPolicyClient"]
            client = client_cls(checkpoint=str(args.checkpoint), yaw_deg=args.yaw, device="cpu", package_root=args.policy_package_root)
        env = MotionDressingEnv(cfg, motion, onset_s=args.onset, speed=args.motion_speed,
                               drive_strength=args.drive_strength, tracking_tolerance_m=args.tracking_tolerance,
                               cell_factory=PreparedFactory(live, cell, placement))
        for method in args.methods:
            obs = env.reset([args.seed])
            states, transitions = [], []
            previous_roi = None
            hold_count, first_success = 0, None
            failure = None
            started = time.monotonic()

            def state():
                cloth = env.positions()[0]
                current = env.cells[0]
                geometry = measure(sleeve, cloth, np.stack([current.finger, current.elbow, current.shoulder]))
                progress = env.progress()[0]
                return dict(obs=obs[0].copy(), positions=cloth.astype(np.float32),
                            human_vertices=current.human_points.astype(np.float32),
                            tcp=env._anchor[0].astype(np.float32).copy(), time_s=env.motion_time_s,
                            finger=current.finger.copy(), elbow=current.elbow.copy(), shoulder=current.shoulder.copy(),
                            upperarm_ratio=float(progress.upperarm_ratio), forearm_ratio=float(progress.forearm_ratio),
                            sleeve_wrapped=bool(geometry["sleeve_wrapped"]), sleeve_cuff_s=float(geometry["cuff_s"]),
                            sleeve_proximal_upper_fraction=float(geometry["proximal_upper_fraction"]))

            states.append(state())
            for step in range(args.steps):
                pos, flags, valid, _ = env.spec.unpack_numpy(obs[0])
                policy = np.zeros(6) if client is None else client.act(pos[valid], flags[valid], np.zeros(3))
                if not np.isfinite(policy).all():
                    raise RuntimeError("Policy returned a non-finite action")
                policy = np.clip(policy, -1., 1.)
                tool = env._anchor[0].copy()
                arm = pos[valid & (flags[:, FLAG_MARKER] > .5)] + tool
                roi = observed_arm_roi(arm, tool)
                transform, diagnostics = (np.eye(4), dict(valid=False, reason="first_frame", matches=0))
                correction = np.zeros(3)
                if method not in ("r1", "hold") and previous_roi is not None:
                    transform, diagnostics = gicp(previous_roi, roi)
                    correction = bounded_correction(transform, tool, args.correction_bound)
                paused = bool(masks[method][step]) if method in masks else method == "hold"
                action = np.zeros(6) if paused else policy.copy()
                action[:3] += correction / cfg.max_translation
                action = np.clip(action, -1., 1.).astype(np.float32)
                previous_roi = roi
                obs, rewards, done, info = env.step(action[None], reset_on_done=False)
                if info[0].get("sim_error"):
                    # The base environment resets on a solver exception. Never append
                    # that reset state as the outcome of this attempted transition.
                    failure = dict(kind="invalid_physics", step=step, detail=info[0]["error"], attempted_action=action.tolist())
                    break
                states.append(state())
                transitions.append(dict(actions=action, policy_actions=policy, pause=paused,
                                        correction=correction, registration_valid=diagnostics["valid"],
                                        registration_matches=diagnostics["matches"], registration_transform=transform,
                                        rewards=float(rewards[0]), tracking_error=float(info[0]["tracking_error"]),
                                        grasp_valid=bool(info[0]["grasp_valid"]),
                                        executed_translation=env._anchor[0] - tool,
                                        body_tracking_max_m=env.body_tracking_max_m))
                if not info[0]["grasp_valid"]:
                    failure = dict(kind="invalid_grasp", step=step + 1, detail="grasp tracking limit exceeded")
                    break
                success = states[-1]["sleeve_wrapped"] and states[-1]["upperarm_ratio"] >= args.success
                hold_count = hold_count + 1 if success else 0
                if hold_count >= args.hold and first_success is None:
                    first_success = step + 1
                if (step + 1) % 25 == 0 or step + 1 == args.steps:
                    print(f"[{method}] {step + 1}/{args.steps} t={env.motion_time_s:.2f}s "
                          f"upper={states[-1]['upperarm_ratio']:.3f} wrapped={states[-1]['sleeve_wrapped']} "
                          f"body_error={env.body_tracking_max_m:.6g}m wall={time.monotonic()-started:.1f}s", flush=True)
            arrays = {key: np.asarray([row[key] for row in states]) for key in states[0]}
            if transitions:
                arrays.update({key: np.asarray([row[key] for row in transitions]) for key in transitions[0]})
            arrays.update(faces=cell.faces, human_faces=motion.faces, opening_idx=cell.opening_idx,
                          grasp_idx=cell.grasp_idx, picker_idx=cell.picker_idx)
            np.savez_compressed(args.out / f"{method}.npz", **arrays)
            record = dict(method=method, steps=len(transitions), success=hold_count >= args.hold and failure is None,
                          ever_held_success=first_success is not None and failure is None,
                          first_success_step=first_success, final_success=hold_count >= args.hold and failure is None,
                          max_upperarm_ratio=max(row["upperarm_ratio"] for row in states),
                          final_upperarm_ratio=states[-1]["upperarm_ratio"],
                          final_sleeve_wrapped=states[-1]["sleeve_wrapped"],
                          pause_count=sum(row["pause"] for row in transitions),
                          registration_failures=sum(not row["registration_valid"] for row in transitions[1:]) if method not in ("r1", "hold") else 0,
                          failure=failure, elapsed_s=time.monotonic() - started, **env.motion_diagnostics())
            records.append(record)
            save_json(args.out / "metrics.json", records)
            print("[result] " + json.dumps(record), flush=True)
    finally:
        if env is not None:
            env.close()
        if client is not None:
            client.close()


def main():
    args = parser().parse_args()
    if (args.steps < 2 or args.hold < 1 or not 0 < args.success <= 1
            or len(set(args.methods)) != len(args.methods)):
        raise ValueError("Need steps >= 2, hold >= 1, success in (0,1], and unique methods")
    values = [args.onset, args.motion_speed, args.future_horizon, args.pause_displacement,
              args.correction_bound, args.yaw, args.tracking_tolerance, args.drive_strength, args.wait_for_gpu]
    if not np.isfinite(values).all() or min(args.onset, args.wait_for_gpu) < 0 or min(values[1:5] + values[6:8]) <= 0:
        raise ValueError("Invalid time, speed, correction or tracking parameters")
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    try:
        run(args)
    except Exception as exc:
        save_json(args.out / "failure.json", dict(error=repr(exc), arguments=vars(args)))
        raise


if __name__ == "__main__":
    main()
