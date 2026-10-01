"""IPC lookahead on a moving GRAB arm: does motion information help the labeller? (M2 pilot)

Reuses the anticipatory-dressing worktree's motion pilot unmodified (GRAB-driven Empty-FEM body, placement,
policy client, success rule) and adds one planner run under three beliefs about the human during the
candidate rollouts. Every condition uses the same candidates, score, hold length and simulation budget; the
executed episode always follows the real GRAB motion.

  current  the arm holds its present pose during the candidate rollouts
  causal   body vertices and joints extrapolated at their velocity over the last decision
  true     the actual GRAB future (privileged diagnostic)
  none     no planning: the checkpoint alone (sanity check against probe_arm_motion.py's r1)

The snapshot covers the IPC world, the env's controller state and the motion state (clock, body target, target
joints, current cells, meshes), so every candidate starts from the same human and cloth. Candidates are scored
with the arm landmarks at the end of their own rollout, not the landmarks cached at reset.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import json
from pathlib import Path
import runpy
import subprocess
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
WT = Path("/home/ge47gax/kun/libuipc-anticipatory-dressing")
sys.path.insert(0, str(HERE))
PROBE = runpy.run_path(str(WT / "scripts/wang_transfer/probe_arm_motion.py"), run_name="motion_probe")
sys.path.insert(0, str(HERE))       # our physical_sleeve / ipc_action_filter before the worktree's copies

from ipc_action_filter import arm_progress  # noqa: E402
from physical_sleeve import DEFAULT_OBJ, SleeveSections, measure, read_obj  # noqa: E402

CONDITIONS = ("none", "current", "causal", "true")
BASE_STATE = ("_anchor", "_offsets", "_last_progress", "_privileged", "_episode_step", "_force_trackers", "_violated")
MOTION_STATE = ("motion_time_s", "motion_running", "_body_target", "_target_joints", "body_tracking_max_m",
                "cells", "arm_meshes", "collider_meshes", "arm_vertices", "collider_vertices")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--motion", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=["none", "current", "causal", "true"])
    p.add_argument("--checkpoint", type=Path, default=Path("/home/ge47gax/kun/libuipc/output/uipc_manip/fmvp_ipc_dagger_r1_20260924/model/fmvp_ipc_bc.pt"))
    p.add_argument("--policy-package-root", type=Path, default=Path("/home/ge47gax/kun/libuipc/.claude/worktrees/residual-rl/python"))
    p.add_argument("--hang", type=Path, default=Path("/home/ge47gax/kun/libuipc/output/uipc_manip/fmvp_better_rollouts_20260923/hang2.npz"))
    p.add_argument("--hang-key", default="k300")
    p.add_argument("--onset", type=float, default=1.)
    p.add_argument("--motion-speed", type=float, default=1.)
    p.add_argument("--steps", type=int, default=450)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--success", type=float, default=.7)
    p.add_argument("--hold", type=int, default=20)
    p.add_argument("--yaw", type=float, default=267.)
    p.add_argument("--obs-mode", default="wang_live_arm")
    p.add_argument("--tracking-tolerance", type=float, default=.002)
    p.add_argument("--drive-strength", type=float, default=1e6)
    p.add_argument("--newton-velocity-tolerance", type=float, default=.01)
    p.add_argument("--interval", type=int, default=1, help="Plan every N decisions inside the planning window")
    p.add_argument("--horizon", type=int, default=4, help="Decisions each candidate is held before scoring")
    p.add_argument("--window-before", type=float, default=.3, help="Seconds before motion onset when planning starts")
    p.add_argument("--window-after", type=float, default=1., help="Seconds after the motion ends when planning stops")
    p.add_argument("--keep-rollout-obs", action="store_true",
                   help="Build the point-cloud observation inside candidate rollouts too (slower; same outcome)")
    return p


def make_env_class():
    from uipc_manip.dressing_motion import MotionDressingEnv

    class PlanningMotionEnv(MotionDressingEnv):
        """``plan_mode`` None/'true' follow the real motion; 'current' and 'causal' replace the body target during
        candidate rollouts only. The real clock is restored with the snapshot afterwards."""
        plan_mode = None
        skip_plan_obs = True        # candidate rollouts never read the observation; the snapshot restores the rngs
        obs_wall_s = 0.

        def observation(self, positions=None):
            if self.plan_mode is not None and self.skip_plan_obs and hasattr(self, "_obs_placeholder"):
                return self._obs_placeholder
            started = time.perf_counter()
            obs = super().observation(positions)
            self.obs_wall_s += time.perf_counter() - started
            self._obs_placeholder = np.zeros_like(obs)
            return obs

        def _sim_step(self):
            if self.plan_mode in (None, "true"):
                return super()._sim_step()
            self._plan_clock += self.cfg.dt
            if self.plan_mode == "causal":
                target = self._plan_base[0] + self._plan_velocity[0] * self._plan_clock
                jump = float(np.max(np.linalg.norm(target - self._body_target, axis=1)))
                if jump > self.max_body_substep_m:
                    raise RuntimeError(f"Extrapolated body motion exceeds substep bound: {jump:g} m")
                self._body_target = target
                self._target_joints = self._plan_base[1] + self._plan_velocity[1] * self._plan_clock
            running, self.motion_running = self.motion_running, False
            try:
                super()._sim_step()
            finally:
                self.motion_running = running

    return PlanningMotionEnv


class MotionPlanner:
    def __init__(self, env, sleeve, masses, *, horizon):
        self.env, self.sleeve, self.horizon = env, sleeve, int(horizon)
        cfg = env.cfg
        self.indices = env._pickers[0]["anchor_idx"]
        self.stiffness = cfg.constraint_strength * masses[self.indices] / cfg.dt ** 2
        self.edges = sleeve.edges
        initial = env.positions()[0]
        self.rest = np.linalg.norm(initial[self.edges[:, 0]] - initial[self.edges[:, 1]], axis=1)
        self.noise_margin, self.replay_checked = .0002, False
        self.history = []                              # (body target, joints) at each executed decision
        self.fingers = []                              # observed fingertip at each executed decision

    def record(self):
        e = self.env
        self.history.append((e._body_target.copy(), e._target_joints.copy()))
        self.fingers.append(np.asarray(e.cells[0].finger, float).copy())

    def finger_shift(self, mode):
        """Fingertip displacement over the next decision under each belief; it enters the 'follow' candidates."""
        e = self.env
        if mode == "true":
            dt = e.cfg.dt * e.cfg.action_repeat
            return np.asarray(e.sample_future(dt)[2][0] - e.sample_future(0.)[2][0], float)
        if mode == "causal" and len(self.fingers) >= 2:
            return self.fingers[-1] - self.fingers[-2]
        return np.zeros(3)

    def snapshot(self):
        e = self.env
        if not e._world.dump():
            raise RuntimeError("IPC snapshot failed")
        state = {n: copy.deepcopy(getattr(e, n)) for n in BASE_STATE + MOTION_STATE if hasattr(e, n)}
        return dict(frame=int(e._world.frame()), state=state, positions=e.positions()[0].copy(),
                    sim_step=e.scene.sim._cur_substep_global,
                    rngs=[copy.deepcopy(r.bit_generator.state) for r in e.rngs])

    def restore(self, snap):
        e = self.env
        e.plan_mode = None
        if not e._world.recover(snap["frame"]):
            raise RuntimeError("IPC restore failed")
        e._world.retrieve()
        for name, value in snap["state"].items():
            setattr(e, name, copy.deepcopy(value))
        e.scene.sim._cur_substep_global = snap["sim_step"]
        for rng, state in zip(e.rngs, snap["rngs"]):
            rng.bit_generator.state = copy.deepcopy(state)
        e._update_targets()
        e._batched_obs.__dict__.get("_occluders", {}).clear()
        error = float(np.max(abs(e.positions()[0] - snap["positions"])))
        if error > 1e-8 or int(e._world.frame()) != snap["frame"]:
            raise RuntimeError(f"Snapshot did not restore exactly: {error} m")

    def _begin(self, mode):
        e = self.env
        e.plan_mode, e._plan_clock = mode, 0.
        if mode == "causal":
            now = (e._body_target.copy(), e._target_joints.copy())
            prev = self.history[-2] if len(self.history) >= 2 else now
            dt = e.cfg.dt * e.cfg.action_repeat
            e._plan_base = now
            e._plan_velocity = ((now[0] - prev[0]) / dt, (now[1] - prev[1]) / dt)

    def evaluate(self, action, nominal, mode):
        e = self.env
        self._begin(mode)
        grasp, tracking = True, 0.
        for _ in range(self.horizon):
            _, _, _, infos = e.step(action[None], reset_on_done=False)
            if infos[0].get("sim_error"):
                return dict(score=-10., feasible=False, sim_error=infos[0].get("error", "")[:200]), None
            grasp &= bool(infos[0]["grasp_valid"])
            tracking = max(tracking, float(infos[0]["tracking_error"]))
        positions = e.positions()[0]
        cell = e.cells[0]
        landmarks = np.stack([cell.finger, cell.elbow, cell.shoulder])     # at the end of this rollout
        rings = measure(self.sleeve, positions, landmarks)["rings"]
        progress = arm_progress(rings, float(np.linalg.norm(np.diff(landmarks, axis=0), axis=1).sum()))
        force = float(np.linalg.norm((self.stiffness[:, None] * (e._anchor[0] + e._offsets[0] - positions[self.indices])).sum(0)))
        stretch = np.linalg.norm(positions[self.edges[:, 0]] - positions[self.edges[:, 1]], axis=1) / self.rest
        score = progress - 2e-5 * max(force - 40., 0.) - .0002 * float(np.sum((action - nominal) ** 2))
        feasible = grasp and tracking <= .019 and np.quantile(stretch, .99) <= 2.25 and stretch.max() <= 4.
        if not feasible:
            score -= 1. + 100. * max(tracking - .019, 0.) + (0. if grasp else 1.)
        return dict(score=float(score), progress_m=progress, gripper_N=force, tracking_m=tracking,
                    grasp_valid=grasp, feasible=bool(feasible)), positions.copy()

    def candidates(self, nominal, shift):
        out = [nominal.copy(), nominal * .5, np.zeros(6)]
        no_rotation = nominal.copy()
        no_rotation[3:] = 0.
        out.append(no_rotation)
        for axis in range(3):
            for direction in (-1., 1.):
                c = nominal.copy()
                c[axis] += direction * .25
                out.append(np.clip(c, -1, 1))
        # Follow the arm: the policy's action and a stop, each shifted by the believed fingertip motion. Kept even
        # when the shift is zero ('current'), so every condition runs the same 12 rollouts.
        follow = np.zeros(6)
        follow[:3] = shift / self.env.cfg.max_translation
        out += [np.clip(nominal + follow, -1, 1), np.clip(follow, -1, 1)]
        return out

    def improve(self, nominal, mode):
        nominal = np.clip(np.asarray(nominal, float), -1, 1)
        shift = self.finger_shift(mode)
        cands = self.candidates(nominal, shift)
        snap = self.snapshot()
        rows = []
        try:
            for a in cands:
                self.restore(snap)
                row, positions = self.evaluate(a, nominal, mode)
                row["action"] = a.tolist()
                rows.append(row)
                if len(rows) == 1 and not self.replay_checked and positions is not None:
                    self.restore(snap)
                    replay, replay_positions = self.evaluate(a, nominal, mode)
                    self.noise_margin = max(.0002, 2 * abs(replay["score"] - row["score"]))
                    row["replay_position_error_m"] = float(np.max(abs(replay_positions - positions))) if replay_positions is not None else None
                    self.replay_checked = True
            best = int(np.argmax([r["score"] for r in rows]))
            if rows[best]["score"] < rows[0]["score"] + self.noise_margin:
                best = 0
        finally:
            self.restore(snap)
        return cands[best].astype(np.float32), dict(selected=best, candidates=rows, noise_margin=self.noise_margin,
                                                     finger_shift_m=shift.tolist())


def run(args):
    from uipc_manip import pretrain_wang, train_sac
    from uipc_manip.grab_motion import BodyMotion, sha256

    motion = BodyMotion.load(args.motion)
    _, training, _ = pretrain_wang.prepare(["teacher", "--region", "13", "--seed", "1",
                                           "--obs-mode", "wang_static_arm", "--no-obs-augment"])
    cfg = train_sac.dressing_config(training)
    live = replace(cfg.live, body=replace(cfg.live.body, device="cpu"))
    cfg = replace(cfg, cells=(("tshirt_26", int(motion.metadata["body_id"])),), cell_source="live",
                  collision_geometry="full_body", live=live, horizon=args.steps + args.horizon + 2, seed=1,
                  decision_watchdog=False, clip_rotation_to_yz=False, anchor_count=48,
                  newton_tolerance=args.newton_velocity_tolerance,
                  cloth_density=750., cloth_strain_rate=10., contact_force_readout=False,
                  workspace=str(args.out / "world"),
                  obs=replace(cfg.obs, mode=args.obs_mode, static_arm=False))
    with np.load(args.hang, allow_pickle=False) as data:
        hang = data[args.hang_key].copy()
    cell, placement = PROBE["prepare_cell"](motion, live, hang, args.yaw)
    rest, faces = read_obj(DEFAULT_OBJ)
    if not np.array_equal(faces, cell.faces):
        raise ValueError("Physical sleeve topology differs from the simulated garment")
    sleeve = SleeveSections(rest * 4., faces, cell.opening_idx)
    tri = cell.cloth[cell.faces]
    area = .5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    masses = np.zeros(len(cell.cloth))
    np.add.at(masses, cell.faces.ravel(), np.repeat(area / 3, 3))
    masses *= 2 * cfg.cloth_thickness * cfg.cloth_density
    dt = cfg.dt * cfg.action_repeat
    motion_end = args.onset + float(motion.times[-1]) / args.motion_speed
    window = (args.onset - args.window_before, motion_end + args.window_after)
    args_json = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    PROBE["save_json"](args.out / "run.json", dict(
        arguments=args_json, checkpoint_sha256=sha256(args.checkpoint), motion_sha256=sha256(args.motion),
        motion=motion.metadata, placement=placement, decision_dt_s=dt, planning_window_s=window,
        worktree_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=WT, text=True).strip(),
        revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip(),
        success_rule="three interior sections wrap the arm and armhole upper fraction >= threshold for hold+1 consecutive states, valid grasp throughout"))
    client = PROBE["make_client"](args)
    env = make_env_class()(cfg, motion, onset_s=args.onset, speed=args.motion_speed, drive_strength=args.drive_strength,
                           tracking_tolerance_m=args.tracking_tolerance,
                           cell_factory=PROBE["PreparedFactory"](live, cell, placement))
    env.skip_plan_obs = not args.keep_rollout_obs
    required = args.hold + 1
    records = []
    try:
        for condition in args.conditions:
            obs = env.reset([args.seed])
            planner = MotionPlanner(env, sleeve, masses, horizon=args.horizon)
            planner.record()
            log = (args.out / f"{condition}_lookahead.jsonl").open("w")
            hold_count, first_success, failure, grasp_failure = 0, None, None, None
            plans = changed = 0
            plan_wall = 0.
            env.obs_wall_s = 0.
            started = time.monotonic()
            max_upper = 0.

            def endpoint_state():
                c = env.cells[0]
                g = measure(sleeve, env.positions()[0], np.stack([c.finger, c.elbow, c.shoulder]))
                return dict(sleeve_sections_wrapped=all(r["wrapped"] for r in g["rings"][1:]),
                            sleeve_armhole_upper_fraction=float(g["armhole_upper_fraction"]))

            last = endpoint_state()
            for step in range(args.steps):
                pos, flags, valid, _ = env.spec.unpack_numpy(obs[0])
                policy = np.clip(client.act(pos[valid], flags[valid], np.zeros(3)), -1., 1.)
                completion_hold = PROBE["endpoint_reached"](last, "interior_armhole", args.success)
                action = np.zeros(6) if completion_hold else policy.copy()
                t = step * dt
                if (condition != "none" and not completion_hold and window[0] <= t <= window[1]
                        and step % args.interval == 0):
                    lookahead_started = time.monotonic()
                    action, diag = planner.improve(action, condition)
                    plans += 1
                    plan_wall += time.monotonic() - lookahead_started
                    changed += diag["selected"] != 0
                    log.write(json.dumps(dict(step=step, t=t, wall_s=time.monotonic() - lookahead_started, **diag),
                                         default=float) + "\n")
                    log.flush()
                obs, _, _, info = env.step(np.asarray(action, np.float32)[None], reset_on_done=False)
                physics_failure, grasp_failure = PROBE["classify_failures"](info[0], step, grasp_failure)
                if physics_failure is not None:
                    failure = physics_failure
                    break
                planner.record()
                last = endpoint_state()
                max_upper = max(max_upper, last["sleeve_armhole_upper_fraction"])
                if grasp_failure is not None:
                    failure = grasp_failure
                    break
                hold_count = hold_count + 1 if PROBE["endpoint_reached"](last, "interior_armhole", args.success) else 0
                if hold_count >= required and first_success is None:
                    first_success = step + 1
                if (step + 1) % 25 == 0:
                    print(f"[{condition}] {step + 1}/{args.steps} t={env.motion_time_s:.2f}s plans={plans} changed={changed} "
                          f"armhole={last['sleeve_armhole_upper_fraction']:.3f} wall={time.monotonic() - started:.0f}s", flush=True)
            log.close()
            # success_ever: held once and no later failure (the original rule); final_success: still held at the end,
            # the rule probe_arm_motion.py uses for r1/GICP. Compare across runners with final_success.
            record = dict(condition=condition, steps=step + 1, success=first_success is not None and failure is None,
                          final_success=hold_count >= required and failure is None,
                          final_sections_wrapped=bool(last["sleeve_sections_wrapped"]),
                          first_success_step=first_success, failure=failure, plans=plans, changed=changed,
                          max_armhole_upper_fraction=max_upper, final_armhole_upper_fraction=last["sleeve_armhole_upper_fraction"],
                          time_s=env.motion_time_s, body_tracking_max_m=env.body_tracking_max_m,
                          plan_wall_s=plan_wall, observation_wall_s=env.obs_wall_s,
                          elapsed_s=time.monotonic() - started)
            records.append(record)
            PROBE["save_json"](args.out / "metrics.json", records)
            print("[result] " + json.dumps(record, default=float), flush=True)
    finally:
        env.close()
        client.close()


def main():
    args = parser().parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    try:
        run(args)
    except Exception as exc:
        PROBE["save_json"](args.out / "failure.json", dict(error=repr(exc)))
        raise


if __name__ == "__main__":
    main()
