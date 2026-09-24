"""Record FMVP rollouts from the already verified gravity-hung start.

The default keeps checkpoint inputs and grasp unchanged. Explicit profiles
record experimental input, grasp or action interventions separately. Compare
controllers or test slowing before the shoulder. Hold for a complete
validation window. Save failed attempts too; only stable, valid-grasp episodes
enter accepted.json. No network training is performed here. Forces are recorded
for ranking, not certified against an absolute real-world safety threshold.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from rollout_controls import crop_observation, force_input, load_profiles, profile_dict, rigid_rotation, tracking_scale


ROOT = Path(__file__).resolve().parents[2]
VARIANTS = {"baseline": 1.0, "half": 0.5, "quarter": 0.25, "handoff": 1.0, "expert": 1.0}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def revision(path):
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def save_json(path, obj):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(obj, indent=2) + "\n")
    temporary.replace(path)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--package-root", type=Path, default=ROOT / ".claude/worktrees/residual-rl/python")
    p.add_argument("--checkpoint", type=Path, default=Path("/home/ge47gax/Desktop/fmvp_sim.pt"))
    p.add_argument("--policy-device", choices=("cpu", "cuda"), default="cpu",
                   help="Checkpoint inference device; independent of the IPC CUDA physics backend.")
    p.add_argument("--hang", type=Path, required=True)
    p.add_argument("--hang-key", default="k300")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--bodies", type=int, nargs="+", default=None)
    p.add_argument("--preflight-manifest", type=Path, default=None,
                   help="Use legal body IDs from a matching full-body preflight; --bodies can select a subset.")
    p.add_argument("--variants", nargs="+", choices=VARIANTS, default=["baseline", "handoff", "expert"])
    p.add_argument("--replicas", type=int, default=1,
                   help="Run identical controllers in separate IPC slots for repeated evaluation.")
    p.add_argument("--profiles-json", type=Path,
                   help="One explicit policy-input/control profile per variant, for controlled comparisons.")
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--steps", type=int, default=500,
                   help="Maximum decisions to first reach success; the hold window is allowed on top.")
    p.add_argument("--hold", type=int, default=30)
    p.add_argument("--autonomous-hold", action="store_true",
                   help="Continue querying the controller after first success to test whether it learned to stop.")
    p.add_argument("--freeze-from-state", type=int, default=None,
                   help="Diagnostic: record FMVP proposals but execute zero actions once this recorded state is reached.")
    p.add_argument("--success", type=float, default=0.7)
    p.add_argument("--success-geometry", choices=("legacy_ratio", "physical_sleeve"), default="legacy_ratio",
                   help="physical_sleeve additionally requires the real cuff and three sleeve sections to wrap the arm.")
    p.add_argument("--stop-proximal-upper", type=float, default=None,
                   help="Diagnostic endpoint: replace the legacy ratio with the proximal sleeve section's upper-arm fraction; requires physical_sleeve.")
    p.add_argument("--slow-along", type=float, default=0.9,
                   help="Slow when gripper projection reaches this fraction of the hand-shoulder chord.")
    p.add_argument("--handoff-forearm", type=float, default=0.5,
                   help="Hand off to the existing expert's forearm stage after this coverage.")
    p.add_argument("--yaw", type=float, default=267.0)
    p.add_argument("--rotation", choices=("off", "fmvp"), default="off",
                   help="Keep the historical zero rotation or apply FMVP's vertical-only PyBullet rotation rule.")
    p.add_argument("--rotation-gain", type=float, default=1.,
                   help="Scale the reference yaw increment; max_translation/0.025 preserves FMVP's rotation per metre.")
    p.add_argument("--bridge-voxel", type=float, default=None,
                   help="Override the policy bridge voxel size; 0 avoids downsampling an already voxelized observation twice.")
    p.add_argument("--collision-geometry", choices=("arm", "full_body"), default="arm",
                   help="IPC cloth collider: historical right arm or the complete SMPL-X body.")
    p.add_argument("--placement-offset-mm", type=float, nargs=3, default=[0., 0., 0.],
                   metavar=("DX", "DY", "DZ"),
                   help="Move the gravity-hung gripper and garment relative to the fingertip in the FMVP frame; record exact millimetres.")
    p.add_argument("--show-viewer", action="store_true", help="Show slot zero live in the native Genesis viewer.")
    p.add_argument("--abort-gripper-force", type=float, default=None,
                   help="Optional simulator-load cutoff to end clearly unusable attempts early; not a real-world safety threshold.")
    p.add_argument("--continue-invalid-grasp", action="store_true",
                   help="Diagnostic only: keep simulating after a grasp failure. By default save and end an already-invalid attempt.")
    p.add_argument("--cloth-density", type=float, default=None,
                   help="Optional kg/m^3 material-density override; save and extract separately from default physics.")
    p.add_argument("--cloth-strain-rate", type=float, default=None,
                   help="Optional Baraff-Witkin over-stretch coefficient; an experimental material parameter, not allowable strain.")
    return p


def main():
    args = parser().parse_args()
    if args.hold < 1 or args.steps < 1:
        raise ValueError("Need positive search and hold windows")
    if args.freeze_from_state is not None and not 0 <= args.freeze_from_state < args.steps:
        raise ValueError("--freeze-from-state must be within the rollout window")
    if args.replicas < 1 or (args.replicas > 1 and len(args.variants) != 1):
        raise ValueError("--replicas >1 requires exactly one variant")
    if args.profiles_json is None and args.replicas == 1 and len(set(args.variants)) != len(args.variants):
        raise ValueError("Duplicate variants require --replicas so output names stay unique")
    if args.abort_gripper_force is not None and args.abort_gripper_force <= 0:
        raise ValueError("--abort-gripper-force must be positive")
    if args.cloth_density is not None and args.cloth_density <= 0:
        raise ValueError("--cloth-density must be positive")
    if args.cloth_strain_rate is not None and args.cloth_strain_rate <= 0:
        raise ValueError("--cloth-strain-rate must be positive")
    if not np.isfinite(args.rotation_gain) or args.rotation_gain < 0:
        raise ValueError("--rotation-gain must be finite and nonnegative")
    if args.bridge_voxel is not None and (not np.isfinite(args.bridge_voxel) or args.bridge_voxel < 0):
        raise ValueError("--bridge-voxel must be finite and nonnegative")
    if args.stop_proximal_upper is not None and (args.success_geometry != "physical_sleeve"
                                               or not 0 < args.stop_proximal_upper <= 1):
        raise ValueError("--stop-proximal-upper requires physical_sleeve and a fraction in (0, 1]")
    profiles = load_profiles(args.profiles_json, len(args.variants)) * args.replicas
    if any(p.lookahead_interval for p in profiles) and (len(profiles) != 1 or args.success_geometry != "physical_sleeve"):
        raise ValueError("IPC lookahead requires one slot and physical_sleeve geometry")
    hang_hash = sha256(args.hang)
    if args.preflight_manifest is not None:
        preflight = json.loads(args.preflight_manifest.read_text())
        if (args.collision_geometry != "full_body" or preflight["collision_geometry"] != "full_body"
                or preflight["garment"] != "tshirt_26" or preflight["min_gap_m"] != 0.003
                or preflight["turn_increment_degrees"] != 5):
            raise ValueError("Preflight geometry does not match full-body tshirt_26 collector settings")
        if (preflight["hang_sha256"] != hang_hash or preflight["hang_key"] != args.hang_key
                or not np.allclose(preflight["placement_offset_mm"], args.placement_offset_mm, atol=1e-12)
                or not np.isclose(preflight["yaw"], args.yaw, atol=1e-12)):
            raise ValueError("Preflight hang, key, yaw, or placement offset differs from collector")
        legal_bodies = set(preflight["legal_bodies"])
        if args.bodies is None:
            args.bodies = preflight["legal_bodies"]
        else:
            if not set(args.bodies) <= legal_bodies:
                raise ValueError(f"Requested body IDs are not all legal in preflight: {sorted(set(args.bodies) - legal_bodies)}")
        if not args.bodies:
            raise ValueError("Preflight found no legal body starts")
    elif args.bodies is None:
        args.bodies = [14046, 14045, 14048]
    if len(set(args.bodies)) != len(args.bodies):
        raise ValueError("Duplicate body IDs would overwrite rollout outputs")
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "run.json").exists():
        raise FileExistsError(f"Choose a new output directory: {args.out}")
    sys.path.insert(0, str(args.package_root.resolve()))
    from uipc_manip import dressing_live, pretrain_wang, train_sac
    from uipc_manip.dressing_body import smplx_faces
    from uipc_manip.contact_force import vertex_forces_multi
    from uipc_manip.dressing_env import GenesisIPCDressingEnv
    from uipc_manip.obs import ObsSpec
    from uipc_manip.wang_bridge import up_axis_rotation
    from uipc_manip.wang_client import WangPolicyClient

    metadata = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    metadata.update(checkpoint_sha256=sha256(args.checkpoint), hang_sha256=hang_hash,
                    preflight_sha256=sha256(args.preflight_manifest) if args.preflight_manifest is not None else None,
                    collector_sha256=sha256(__file__),
                    controls_sha256=sha256(Path(__file__).with_name("rollout_controls.py")),
                    bridge_sha256=sha256(args.package_root / "uipc_manip/wang_bridge.py"),
                    environment_sha256=sha256(args.package_root / "uipc_manip/dressing_env.py"),
                    collector_revision=revision(ROOT), package_revision=revision(args.package_root),
                    policy_force_input="zero" if args.profiles_json is None else "per_profile; stored in policy_force",
                    profiles=[profile_dict(p) for p in profiles], force_sampling="end of decision, not substep peak",
                    observations="obs[t] -> actions[t] -> obs[t+1]; force arrays align with obs",
                    controller_id="0 FMVP, 1 scripted expert, 2 hold, 3 tracking-limited FMVP, 4 IPC-improved FMVP; policy_actions are proposals",
                    accepted_rule=("proximal sleeve fraction >= stop_proximal_upper, real sleeve wrapped throughout hold, valid grasp, no sim error"
                                   if args.stop_proximal_upper is not None else
                                   "upper >= success throughout hold, valid grasp throughout, no sim error"))
    save_json(args.out / "run.json", metadata)
    with np.load(args.hang) as source:
        hang = source[args.hang_key].copy()
    np.savez_compressed(args.out / "hang.npz", **{args.hang_key: hang})
    rotation = up_axis_rotation(args.yaw)
    full_body_faces = smplx_faces() if args.collision_geometry == "full_body" else None
    sleeve_template = None
    if args.success_geometry == "physical_sleeve":
        from physical_sleeve import DEFAULT_OBJ, SleeveSections, measure as measure_sleeve, read_obj
        rest_vertices, rest_faces = read_obj(DEFAULT_OBJ)
        sleeve_template = (rest_vertices * 4., rest_faces)
    variants = args.variants * args.replicas
    labels = [f"{v}_rep{i}" if args.replicas > 1 else v for i, v in enumerate(variants)]
    if args.profiles_json is not None:
        labels = [f"{v}_{profiles[i].name}_slot{i}" for i, v in enumerate(variants)]
    build_original = dressing_live.LiveCellFactory.build
    prepare_original = GenesisIPCDressingEnv._prepare_start
    placements = {}

    def prepare_with_profiles(environment):
        # Apply explicitly requested grasp changes before settling and snapshot
        # creation, so reset reproduces the same constraint definition.
        import uipc
        for i, profile in enumerate(profiles):
            if profile.anchor_count is not None:
                environment._pickers[i]["anchor_idx"] = environment.cells[i].anchor_indices(profile.anchor_count)
            if profile.grasp_strength_gain != 1:
                slot = environment.slots[i].geometry().vertices().find("strength_ratio")
                uipc.view(slot)[:] = environment.cfg.constraint_strength * profile.grasp_strength_gain
        prepare_original(environment)

    def hung_build(factory, garment, human):
        cell = build_original(factory, garment, human)
        finger = np.asarray(cell.finger, float)
        target = finger + (np.array([-0.033, 0.106, -0.003])
                           + np.asarray(args.placement_offset_mm, float) / 1000.) @ rotation
        desired_opening = np.array([-0.085, -0.135, -0.050])
        best = None
        for degrees in range(0, 360, 5):
            c, s = np.cos(np.radians(degrees)), np.sin(np.radians(degrees))
            turn = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])
            cloth = hang @ turn.T + target
            if full_body_faces is None:
                gap = dressing_live.garment_arm_gap(cloth, cell.faces, cell.arm_points, cell.arm_faces)
            else:
                gap = dressing_live.garment_arm_gap(cloth, cell.faces, cell.human_points, full_body_faces)
            if gap < 0.003:
                continue
            opening = (cloth[cell.opening_idx].mean(0) - finger) @ rotation.T
            error = float(np.linalg.norm(opening - desired_opening))
            if best is None or error < best[0]:
                best = (error, degrees, gap, opening, cloth)
        if best is None:
            raise RuntimeError(f"No legal gravity-hung placement for body {human}")
        error, degrees, gap, opening, cloth = best
        placements[str(human)] = dict(turn_degrees=degrees, gap_m=gap,
                                      collision_geometry=args.collision_geometry,
                                      placement_offset_mm=args.placement_offset_mm,
                                      opening_model=opening.tolist(), opening_error_m=error)
        return replace(cell, cloth=cloth, picker_pos=target,
                       pull_waypoints=np.stack([target, cell.pull_waypoints[-1]]))

    dressing_live.LiveCellFactory.build = hung_build
    GenesisIPCDressingEnv._prepare_start = prepare_with_profiles
    all_records = []
    skipped = []
    client = WangPolicyClient(checkpoint=str(args.checkpoint), yaw_deg=args.yaw,
                              device=args.policy_device, voxel=args.bridge_voxel, package_root=args.package_root)
    clients = {args.bridge_voxel: client}
    for profile in profiles:
        voxel = profile.bridge_voxel if profile.bridge_voxel is not None else args.bridge_voxel
        if voxel not in clients:
            clients[voxel] = WangPolicyClient(checkpoint=str(args.checkpoint), yaw_deg=args.yaw,
                                            device=args.policy_device, voxel=voxel, package_root=args.package_root)
    policy_clients = [clients[p.bridge_voxel if p.bridge_voxel is not None else args.bridge_voxel] for p in profiles]
    try:
        for body in args.bodies:
            _, training_args, _ = pretrain_wang.prepare(
                ["teacher", "--region", "13", "--seed", "1", "--obs-mode", "wang_static_arm", "--no-obs-augment"])
            n = len(variants)
            material = {} if args.cloth_density is None else {"cloth_density": args.cloth_density}
            if args.cloth_strain_rate is not None:
                material["cloth_strain_rate"] = args.cloth_strain_rate
            cfg = replace(train_sac.dressing_config(training_args),
                          cells=tuple(("tshirt_26", body) for _ in range(n)), cell_source="live",
                          collision_geometry=args.collision_geometry,
                          horizon=args.steps + args.hold + 10, seed=1, show_viewer=args.show_viewer, decision_watchdog=False,
                          clip_rotation_to_yz=False, anchor_count=48, contact_force_readout=True,
                          **material)
            print(f"[collect] building body={body} variants={labels}", flush=True)
            try:
                env = GenesisIPCDressingEnv(cfg, num_envs=n)
            except RuntimeError as exc:
                if "No legal gravity-hung placement" not in str(exc):
                    raise
                row = {"body": body, "reason": str(exc)}
                skipped.append(row)
                save_json(args.out / "skipped.json", skipped)
                print(f"[skip] {json.dumps(row)}", flush=True)
                continue
            try:
                obs = env.reset([args.seed] * n)
                env._arm_force_summaries()  # Initializes the lazy contact exporter before state zero.
                spec = ObsSpec(training_args.point_budget)
                cell = env.cells[0]
                sleeve = SleeveSections(*sleeve_template, cell.opening_idx) if sleeve_template is not None else None
                sleeve_landmarks = np.stack([cell.finger, cell.elbow, cell.shoulder])
                finger, shoulder = np.asarray(cell.finger), np.asarray(cell.shoulder)
                axis = shoulder - finger
                triangles = cell.cloth[cell.faces]
                areas = .5 * np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                                     triangles[:, 2] - triangles[:, 0]), axis=1)
                masses = np.zeros(len(cell.cloth))
                np.add.at(masses, cell.faces.ravel(), np.repeat(areas / 3, 3))
                masses *= 2 * cfg.cloth_thickness * cfg.cloth_density
                planner = None
                if profiles[0].lookahead_interval:
                    from ipc_action_filter import IPCActionFilter
                    planner = IPCActionFilter(env, sleeve, masses, strength_gain=profiles[0].grasp_strength_gain)
                body_dir = args.out / f"body_{body}_seed_{args.seed}"
                body_dir.mkdir()
                save_json(body_dir / "config.json", dict(config=cfg.to_dict(), placement=placements[str(body)],
                                                         mass_kg=float(masses.sum())))
                buffers = [{k: [] for k in ["obs", "positions", "tcp", "tcp_rotation", "gripper_force", "arm_force", "body_force",
                            "upperarm_ratio", "forearm_ratio", "actions", "policy_actions", "rewards",
                            "executed_translation", "tracking_error", "early_turn", "grasp_valid", "speed_scale", "controller_id",
                            "policy_force", "tracking_scale", "planner_choice"]}
                           for _ in range(n)]
                policy_forces = np.zeros((n, 3), np.float32)
                if sleeve is not None:
                    for buffer in buffers:
                        buffer.update(sleeve_wrapped=[], sleeve_cuff_s=[], sleeve_proximal_upper_fraction=[])
                success_at, finish_at = [None] * n, [None] * n
                handoff_at = [None] * n
                completed = np.zeros(n, bool)
                slowed = np.zeros(n, bool)
                failures = [None] * n

                def record_state():
                    positions = env.positions()
                    pairs = vertex_forces_multi(env._force_feature, cfg.dt, env._force_blocks)
                    for i in range(n):
                        if completed[i]:
                            continue
                        indices = env._pickers[i]["anchor_idx"]
                        target = env._anchor[i][None, :] + env._offsets[i]
                        stiffness = cfg.constraint_strength * profiles[i].grasp_strength_gain * masses[indices] / cfg.dt ** 2
                        grip = -(stiffness[:, None] * (target - positions[i][indices])).sum(0)
                        progress = env._last_progress[i]
                        body_force = pairs[i][0] + pairs[i][1]
                        policy_forces[i] = force_input(profiles[i], body_force.sum(0), grip, policy_forces[i])
                        values = dict(obs=crop_observation(obs[i], spec, profiles[i]).copy(),
                                      policy_force=policy_forces[i].copy(), positions=positions[i].astype(np.float32),
                                      tcp=np.asarray(env._anchor[i], np.float32).copy(), gripper_force=grip,
                                      tcp_rotation=rigid_rotation(env._initial_offsets[i], env._offsets[i]),
                                      arm_force=body_force[env._force_arm_indices[i]].sum(0),
                                      body_force=body_force.sum(0),
                                      upperarm_ratio=float(progress.upperarm_ratio),
                                      forearm_ratio=float(progress.forearm_ratio))
                        if sleeve is not None:
                            geometry = measure_sleeve(sleeve, positions[i], sleeve_landmarks)
                            values.update(sleeve_wrapped=geometry["sleeve_wrapped"],
                                          sleeve_cuff_s=geometry["cuff_s"],
                                          sleeve_proximal_upper_fraction=geometry["proximal_upper_fraction"])
                        for key, value in values.items():
                            buffers[i][key].append(value)

                record_state()
                started = time.monotonic()
                timing = dict(policy_s=0., environment_s=0., recording_s=0., lookahead_s=0.)
                for step in range(args.steps + args.hold):
                    actions = np.zeros((n, env.action_dim), np.float32)
                    proposed = np.zeros_like(actions)
                    scales = np.zeros(n)
                    tracking_scales = np.ones(n)
                    planner_choices = np.full(n, -1, dtype=np.int16)
                    controllers = np.full(n, 2, dtype=np.int8)  # 0 FMVP, 1 scripted expert, 2 hold
                    need_expert = any(v in ("expert", "handoff") for v in variants)
                    if need_expert:
                        if getattr(env, "_heuristic", None) is None:
                            from uipc_manip.dressing_heuristic import HeuristicDressingPolicy
                            env._heuristic = HeuristicDressingPolicy(env)
                        for i, variant in enumerate(variants):
                            if variant == "handoff" and handoff_at[i] is None and buffers[i]["forearm_ratio"][-1] >= args.handoff_forearm:
                                handoff_at[i] = step
                                # Start from the sleeve's current state on the forearm. The
                                # expert's shadow counters must not advance without its actions.
                                env._heuristic.stage[i] = 2
                                env._heuristic._steps[i] = 0
                                env._heuristic._align_steps[i] = 0
                                env._heuristic._best_upper[i] = buffers[i]["upperarm_ratio"][-1]
                                print(f"[handoff] body={body} state={step} forearm={buffers[i]['forearm_ratio'][-1]:.3f}", flush=True)
                        expert = env.scripted_actions()
                    else:
                        expert = None
                    for i, variant in enumerate(variants):
                        if completed[i] or (success_at[i] is not None and not args.autonomous_hold):
                            continue
                        if variant == "expert" or (variant == "handoff" and handoff_at[i] is not None):
                            proposed[i] = actions[i] = expert[i]
                            scales[i] = 1.
                            controllers[i] = 1
                        else:
                            controllers[i] = 0
                            pos, feat, valid, _ = (x[0] for x in spec.unpack_numpy(buffers[i]["obs"][-1][None]))
                            valid = valid.astype(bool)
                            policy_started = time.monotonic()
                            action = policy_clients[i].act(pos[valid], feat[valid], policy_forces[i])
                            timing["policy_s"] += time.monotonic() - policy_started
                            model_vertical_rotation = float((rotation @ action[3:])[1])
                            action[3:] = 0.
                            if args.rotation == "fmvp":
                                delta = abs(model_vertical_rotation)
                                if delta > np.deg2rad(5.):
                                    delta *= np.deg2rad(5.) / np.sqrt(3.)
                                gain = args.rotation_gain if profiles[i].rotation_gain is None else profiles[i].rotation_gain
                                action[5] = gain * np.sign(model_vertical_rotation) * delta / cfg.max_rotation
                            proposed[i] = action
                            along = float((env._anchor[i] - finger) @ axis / (axis @ axis))
                            slowed[i] |= along >= args.slow_along
                            scales[i] = VARIANTS[variant] if slowed[i] else 1.
                            actions[i] = action * scales[i]
                            if args.freeze_from_state is not None and step >= args.freeze_from_state:
                                actions[i] = 0.
                                scales[i] = 0.
                                controllers[i] = 2
                            if profiles[i].tracking_budget_m is not None and controllers[i] == 0:
                                tracking_scales[i] = tracking_scale(
                                    np.clip(actions[i], -1, 1), anchor=env._anchor[i], offsets=env._offsets[i],
                                    held=buffers[i]["positions"][-1][env._pickers[i]["anchor_idx"]],
                                    max_translation=cfg.max_translation, max_rotation=cfg.max_rotation,
                                    budget=profiles[i].tracking_budget_m)
                                actions[i] *= tracking_scales[i]
                                scales[i] *= tracking_scales[i]
                                if tracking_scales[i] < 1:
                                    controllers[i] = 3
                    actions = np.clip(actions, -1, 1)
                    if planner is not None and success_at[0] is None and not completed[0] and step >= 80:
                        load = float(np.linalg.norm(buffers[0]["gripper_force"][-1]))
                        if step % profiles[0].lookahead_interval == 0 and load > 15.:
                            lookahead_started = time.monotonic()
                            actions[0], diagnostic = planner.improve(actions[0])
                            timing["lookahead_s"] += time.monotonic() - lookahead_started
                            planner_choices[0] = diagnostic["selected"]
                            if diagnostic["selected"] != 0:
                                controllers[0] = 4
                            with (body_dir / "lookahead.jsonl").open("a") as log:
                                log.write(json.dumps(dict(state=step, **diagnostic)) + "\n")
                    anchors = np.stack(env._anchor).copy()
                    environment_started = time.monotonic()
                    obs, rewards, dones, infos = env.step(actions)
                    timing["environment_s"] += time.monotonic() - environment_started
                    if any(info.get("sim_error") for info in infos) or np.any(dones):
                        for i in range(n):
                            if not completed[i]:
                                failures[i] = infos[i].get("error", "unexpected environment reset")
                        break
                    recording_started = time.monotonic()
                    record_state()
                    timing["recording_s"] += time.monotonic() - recording_started
                    for i in range(n):
                        if completed[i]:
                            continue
                        values = dict(actions=actions[i].copy(), policy_actions=proposed[i].copy(), controller_id=controllers[i],
                                      tracking_scale=tracking_scales[i],
                                      planner_choice=planner_choices[i],
                                      rewards=float(rewards[i]), speed_scale=float(scales[i]),
                                      executed_translation=np.asarray(env._anchor[i]) - anchors[i],
                                      tracking_error=float(infos[i]["tracking_error"]),
                                      early_turn=bool(infos[i]["early_turn"]), grasp_valid=bool(infos[i]["grasp_valid"]))
                        for key, value in values.items():
                            buffers[i][key].append(value)
                        if args.abort_gripper_force is not None:
                            load = float(np.linalg.norm(buffers[i]["gripper_force"][-1]))
                            if load > args.abort_gripper_force:
                                failures[i] = f"simulated gripper load {load:.1f} N exceeded collection cutoff {args.abort_gripper_force:.1f} N"
                                completed[i] = True
                                continue
                        if not args.continue_invalid_grasp and not infos[i]["grasp_valid"]:
                            failures[i] = "grasp tracking exceeded validity limit; ended at first invalid transition"
                            completed[i] = True
                            continue
                        sleeve_ok = sleeve is None or buffers[i]["sleeve_wrapped"][-1]
                        endpoint_reached = (infos[i]["upperarm_ratio"] >= args.success
                                            if args.stop_proximal_upper is None else
                                            buffers[i]["sleeve_proximal_upper_fraction"][-1] >= args.stop_proximal_upper)
                        if success_at[i] is None and endpoint_reached and sleeve_ok:
                            success_at[i] = step + 1  # state index, after this transition
                            finish_at[i] = success_at[i] + args.hold
                        if finish_at[i] is not None and step + 1 >= finish_at[i]:
                            completed[i] = True
                        elif success_at[i] is None and step + 1 >= args.steps:
                            completed[i] = True
                    if step % 20 == 0 or np.all(completed):
                        status = " | ".join(f"{v}: upper={buffers[i]['upperarm_ratio'][-1]:.3f} "
                                            f"grip={np.linalg.norm(buffers[i]['gripper_force'][-1]):.1f}N "
                                            f"{'done' if completed[i] else 'hold' if success_at[i] is not None else 'run'}"
                                            for i, v in enumerate(labels))
                        print(f"[collect] body={body} step={step+1} {time.monotonic()-started:.0f}s | {status}", flush=True)
                    if np.all(completed):
                        break
                timing["rollout_wall_s"] = time.monotonic() - started
                timing["decisions"] = step + 1
                save_json(body_dir / "timing.json", timing)
                records = []
                for i, variant in enumerate(variants):
                    data = {k: np.asarray(v) for k, v in buffers[i].items()}
                    t = len(data["actions"])
                    if not t:
                        raise RuntimeError(f"No valid transitions for {body}/{variant}: {failures[i]}")
                    held = data["upperarm_ratio"][success_at[i]:] if success_at[i] is not None else np.array([])
                    hold_complete = bool(success_at[i] is not None and finish_at[i] is not None
                                         and t >= finish_at[i])
                    stable = bool(hold_complete and held.size >= args.hold + 1 and np.min(held) >= args.success)
                    if args.stop_proximal_upper is not None:
                        proximal_hold = (data["sleeve_proximal_upper_fraction"][success_at[i]:]
                                         if success_at[i] is not None else np.array([]))
                        stable = bool(hold_complete and proximal_hold.size >= args.hold + 1
                                      and proximal_hold.min() >= args.stop_proximal_upper)
                    if sleeve is not None and success_at[i] is not None:
                        stable = stable and bool(data["sleeve_wrapped"][success_at[i]:].all())
                    valid = bool(np.all(data["grasp_valid"]))
                    accepted = stable and valid and failures[i] is None
                    force = np.linalg.norm(data["gripper_force"][1:], axis=1)
                    arm_force = np.linalg.norm(data["arm_force"][1:], axis=1)
                    record = dict(body=body, variant=variant, replica=i if args.replicas > 1 else None,
                                  profile=profile_dict(profiles[i]),
                                  checkpoint_sha256=metadata["checkpoint_sha256"], simulator="Genesis+IPC",
                                  max_translation_m=cfg.max_translation, max_rotation_rad=cfg.max_rotation,
                                  decision_dt_s=cfg.dt * cfg.action_repeat,
                                  tcp_rotation_convention="world rotation relative to initial virtual tool orientation; identity at reset",
                                  collision_geometry=args.collision_geometry,
                                  success_geometry=args.success_geometry,
                                  stop_proximal_upper=args.stop_proximal_upper,
                                  seed=args.seed, transitions=t, success_state=success_at[i],
                                  handoff_state=handoff_at[i],
                                  hold_complete=hold_complete, timed_out=bool(success_at[i] is None and t >= args.steps),
                                  stable_success=stable, valid_grasp=valid,
                                  accepted=accepted, sim_error=failures[i], final_upper=float(data["upperarm_ratio"][-1]),
                                  peak_upper=float(data["upperarm_ratio"].max()),
                                  hold_min_upper=float(held.min()) if held.size else None,
                                  gripper_p90_N=float(np.percentile(force, 90)), gripper_peak_N=float(force.max()),
                                  arm_p90_N=float(np.percentile(arm_force, 90)), arm_peak_N=float(arm_force.max()),
                                  early_turn=bool(data["early_turn"].any()),
                                  duration_sim_s=t * cfg.dt * cfg.action_repeat,
                                  path=str((body_dir / f"{labels[i]}.npz").relative_to(args.out)))
                    full_body = (dict(human_vertices=env.collider_meshes[i][0],
                                      human_faces=env.collider_meshes[i][1])
                                 if args.collision_geometry == "full_body" else {})
                    np.savez_compressed(body_dir / f"{labels[i]}.npz", **data, **full_body,
                                        grasp_indices=env._pickers[i]["anchor_idx"], initial_grasp_offsets=env._initial_offsets[i],
                                        faces=cell.faces, arm_vertices=cell.arm_points, arm_faces=cell.arm_faces,
                                        opening_idx=cell.opening_idx, finger=finger, shoulder=shoulder,
                                        elbow=np.asarray(cell.elbow), metadata_json=json.dumps(record))
                    records.append(record)
                    print("[result] " + json.dumps(record), flush=True)
                save_json(body_dir / "metrics.json", records)
                all_records.extend(records)
                save_json(args.out / "metrics.json", all_records)
                save_json(args.out / "accepted.json", [r for r in all_records if r["accepted"]])
            finally:
                env.close()
    finally:
        for policy_client in clients.values():
            policy_client.close()
        dressing_live.LiveCellFactory.build = build_original
        GenesisIPCDressingEnv._prepare_start = prepare_original
    save_json(args.out / "skipped.json", skipped)
    if not (args.out / "metrics.json").exists():
        save_json(args.out / "metrics.json", [])
        save_json(args.out / "accepted.json", [])
    print(f"[complete] accepted {sum(r['accepted'] for r in all_records)}/{len(all_records)} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
