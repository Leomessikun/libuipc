"""Garment-general fork of collect_better_rollouts.py: --garment, --fit-sleeve-ratio, --body-fit-filter, --armhole-endpoint,
--batch-bodies (several bodies in one IPC world), --lookahead-from/--min-load.

Record FMVP rollouts from the already verified gravity-hung start.

The default keeps checkpoint inputs and grasp unchanged. Explicit profiles
record experimental input, grasp or action interventions separately. Compare
controllers or test slowing before the shoulder. Hold for a complete
validation window. Save failed attempts too; only stable, valid-grasp episodes
enter accepted.json. No network training is performed here. Forces are recorded
for ranking, not certified against an absolute real-world safety threshold.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from rollout_controls import crop_observation, force_input, load_profiles, profile_dict, rigid_rotation, tracking_scale


ROOT = Path("/home/ge47gax/kun/libuipc")
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
    p.add_argument("--policy-device", choices=("cpu", "cuda"), default=None,
                   help="Checkpoint inference device; independent of the IPC CUDA physics backend.")
    p.add_argument("--policy-socket", type=Path,
                   help="Reuse a resident policy server instead of loading a model in each worker.")
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
    p.add_argument("--no-placement-cache", action="store_true",
                   help="Diagnostic: repeat the identical full-body placement search for every controller slot.")
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
    p.add_argument("--garment", default="tshirt_26")
    p.add_argument("--align-armhole-axis", action="store_true",
                   help="Start with the forearm's extension through the armhole centre (first legal of the best-aligned placements).")
    p.add_argument("--sections-wrap", action="store_true",
                   help="Sleeve on the arm = its three interior sections wrap the arm; the cuff may hang past the hand.")
    p.add_argument("--arm-frame-placement", action="store_true",
                   help="Place the gripper and opening in a frame whose x follows this arm's forearm (FMVP's rule).")
    p.add_argument("--friction", type=float, default=None, help="Cloth-body friction override.")
    p.add_argument("--batch-bodies", type=int, default=1,
                   help="Put this many bodies (same garment) in one IPC world; each keeps its replicas. No lookahead.")
    p.add_argument("--body-fit-filter", type=float, default=None, help="Wang's body thickness filter in metres (0.18).")
    p.add_argument("--fit-sleeve-ratio", type=float, default=None,
                   help="Scale the garment up so the sleeve radius is at least this multiple of the upper-arm radius.")
    p.add_argument("--armhole-endpoint", action="store_true",
                   help="Stop and accept on the armhole seam's place on the upper arm (recorded in sleeve_proximal_upper_fraction).")
    p.add_argument("--lookahead-from", type=int, default=80)
    p.add_argument("--lookahead-min-load", type=float, default=15.)
    p.add_argument("--macro-recovery", action="store_true", help="Multi-decision IPC recovery macros when the sleeve stalls (slot 0).")
    p.add_argument("--macro-horizon", type=int, default=8)
    p.add_argument("--macro-stall-m", type=float, default=.004)
    p.add_argument("--cloth-strain-rate", type=float, default=None,
                   help="Optional Baraff-Witkin over-stretch coefficient; an experimental material parameter, not allowable strain.")
    return p


def main():
    args = parser().parse_args()
    # An atomic dataset-local setting can switch future workers without
    # interrupting the current batch or rewriting its data/configuration.
    runtime_path = args.out.parent / 'policy_runtime.json'
    args.policy_runtime = None
    if args.policy_socket is None and args.policy_device is None and runtime_path.exists():
        runtime = json.loads(runtime_path.read_text())
        args.policy_socket = Path(runtime['socket_path'])
        args.policy_device = runtime['device']
        args.policy_runtime = runtime_path
    args.policy_device = args.policy_device or ('cuda' if args.policy_socket is not None else 'cpu')
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
        rest_vertices, rest_faces = read_obj(DEFAULT_OBJ.parent / f"{args.garment}.obj")
        sleeve_template = (rest_vertices * 4., rest_faces)
    variants = args.variants * args.replicas
    labels = [f"{v}_rep{i}" if args.replicas > 1 else v for i, v in enumerate(variants)]
    if args.profiles_json is not None:
        labels = [f"{v}_{profiles[i].name}_slot{i}" for i, v in enumerate(variants)]
    build_original = dressing_live.LiveCellFactory.build
    prepare_original = GenesisIPCDressingEnv._prepare_start
    placements = {}
    prepared_cells = {}
    placement_timing = dict(searches=0, cache_hits=0, search_s=0.)

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
        # Controller slots share the same body, hang and requested placement.
        # Keep the fully checked geometry, but give every slot independent arrays.
        key = (garment, human)
        if not args.no_placement_cache and key in prepared_cells:
            placement_timing['cache_hits'] += 1
            return deepcopy(prepared_cells[key])
        placement_started = time.monotonic()
        cell = build_original(factory, garment, human)
        finger = np.asarray(cell.finger, float)
        frame = rotation
        if args.arm_frame_placement:
            # FMVP places the gripper along the forearm (wrist + 2 hand radii along it), not at a fixed world
            # offset. Keep the model frame's up and handedness, but point its x along this arm's forearm
            # (horizontal part), so the hung opening faces the hand whatever the elbow does.
            forward = np.asarray(cell.elbow, float) - finger
            forward[2] = 0.
            forward /= np.linalg.norm(forward)
            up = rotation[1] / np.linalg.norm(rotation[1])
            side = np.cross(forward, up)
            if np.sign(np.linalg.det(np.stack([forward, up, side]))) != np.sign(np.linalg.det(rotation)):
                side = -side
            frame = np.stack([forward, up, side])
        target = finger + (np.array([-0.033, 0.106, -0.003])
                           + np.asarray(args.placement_offset_mm, float) / 1000.) @ frame
        desired_opening = np.array([-0.085, -0.135, -0.050])
        fit_scale = 1.
        if args.fit_sleeve_ratio is not None:
            # Size the garment to the body, as one picks a size: scale the hang about the picker so the
            # mean sleeve cross-section radius is at least fit_sleeve_ratio times the mid-upper-arm radius.
            from physical_sleeve import SleeveSections as _Sections
            sections = _Sections(*sleeve_template, cell.opening_idx)
            sleeve_r = float(np.mean([np.linalg.norm(q - q.mean(0), axis=1).mean() for q in sections.points(hang)[1:]]))
            elbow, shoulder = np.asarray(cell.elbow, float), np.asarray(cell.shoulder, float)
            axis_u = (shoulder - elbow) / np.linalg.norm(shoulder - elbow)
            rel = np.asarray(cell.arm_points, float) - (elbow + shoulder) / 2
            along = rel @ axis_u
            radial = np.linalg.norm(rel[np.abs(along) < 0.01] - np.outer(along[np.abs(along) < 0.01], axis_u), axis=1)
            arm_r = float(np.median(radial[radial < 0.12]))
            fit_scale = max(1., args.fit_sleeve_ratio * arm_r / sleeve_r)
        hang_fit = hang * fit_scale
        aligned = False
        if args.align_armhole_axis:
            # Aim the forearm at the armhole: over turns and small gripper offsets, rank candidates by how far
            # the armhole centre sits from the forearm's extension beyond the fingertip (5-20 cm out), and take
            # the first legal one. The calibrated PyBullet opening coordinates do not ensure this for other
            # garments and scales, and the hand then meets the garment beside the opening.
            elbow = np.asarray(cell.elbow, float)
            fwd = (elbow - finger) / np.linalg.norm(elbow - finger)
            ranked = []
            for dy in (-40., -20., 0., 20., 40.):
                for dz in (-60., -40., -20., 0., 20., 40., 60.):
                    tgt = finger + (np.array([-0.033, 0.106, -0.003]) + (np.asarray(args.placement_offset_mm, float)
                                                                          + np.array([0., dy, dz])) / 1000.) @ frame
                    for degrees in range(0, 360, 5):
                        c, s_ = np.cos(np.radians(degrees)), np.sin(np.radians(degrees))
                        turn = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1.]])
                        ah = (hang_fit[cell.opening_idx].mean(0)) @ turn.T + tgt - finger
                        t = float(ah @ fwd)
                        off = float(np.linalg.norm(ah - t * fwd))
                        cost = off + 2. * max(0., -0.20 - t) + 2. * max(0., t + 0.05)   # armhole 5-20 cm past the tip
                        ranked.append((cost, degrees, tgt))
            ranked.sort(key=lambda r: r[0])
            for cost, degrees, tgt in ranked[:400]:
                c, s_ = np.cos(np.radians(degrees)), np.sin(np.radians(degrees))
                cloth = hang_fit @ np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1.]]).T + tgt
                if full_body_faces is None:
                    gap = dressing_live.garment_arm_gap(cloth, cell.faces, cell.arm_points, cell.arm_faces)
                else:
                    gap = dressing_live.garment_arm_gap(cloth, cell.faces, cell.human_points, full_body_faces)
                if gap >= 0.003:
                    target = tgt
                    hang_fit = hang_fit @ np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1.]]).T
                    aligned = True
                    break
        best = None
        for degrees in ((0,) if aligned else range(0, 360, 5)):
            c, s = np.cos(np.radians(degrees)), np.sin(np.radians(degrees))
            turn = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])
            cloth = hang_fit @ turn.T + target
            if full_body_faces is None:
                gap = dressing_live.garment_arm_gap(cloth, cell.faces, cell.arm_points, cell.arm_faces)
            else:
                gap = dressing_live.garment_arm_gap(cloth, cell.faces, cell.human_points, full_body_faces)
            if gap < 0.003:
                continue
            opening = (cloth[cell.opening_idx].mean(0) - finger) @ frame.T
            error = float(np.linalg.norm(opening - desired_opening))
            if best is None or error < best[0]:
                best = (error, degrees, gap, opening, cloth)
        if best is None:
            raise RuntimeError(f"No legal gravity-hung placement for body {human}")
        error, degrees, gap, opening, cloth = best
        placements[str(human)] = dict(turn_degrees=degrees, gap_m=gap,
                                      collision_geometry=args.collision_geometry,
                                      placement_offset_mm=args.placement_offset_mm,
                                      opening_model=opening.tolist(), opening_error_m=error, fit_scale=fit_scale)
        placed = replace(cell, cloth=cloth, picker_pos=target,
                         pull_waypoints=np.stack([target, cell.pull_waypoints[-1]]))
        placement_timing['searches'] += 1
        placement_timing['search_s'] += time.monotonic() - placement_started
        if args.no_placement_cache:
            return placed
        prepared_cells[key] = placed
        return deepcopy(placed)

    dressing_live.LiveCellFactory.build = hung_build
    GenesisIPCDressingEnv._prepare_start = prepare_with_profiles
    all_records = []
    skipped = []
    def make_client(voxel):
        kwargs = dict(checkpoint=str(args.checkpoint), yaw_deg=args.yaw,
                      device=args.policy_device, voxel=voxel, package_root=args.package_root)
        if args.policy_socket is not None:
            from policy_service import PersistentPolicyClient
            return PersistentPolicyClient(args.policy_socket, **kwargs)
        return WangPolicyClient(**kwargs)

    client = make_client(args.bridge_voxel)
    if args.policy_socket is not None:
        metadata['policy_server'] = client.info
        save_json(args.out / 'run.json', metadata)
    clients = {args.bridge_voxel: client}
    for profile in profiles:
        voxel = profile.bridge_voxel if profile.bridge_voxel is not None else args.bridge_voxel
        if voxel not in clients:
            clients[voxel] = make_client(voxel)
    policy_clients = [clients[p.bridge_voxel if p.bridge_voxel is not None else args.bridge_voxel] for p in profiles]
    try:
        batch = max(1, int(args.batch_bodies))
        if batch > 1 and any(p.lookahead_interval for p in profiles):
            raise ValueError("--batch-bodies does not combine with the IPC lookahead")
        base_variants, base_labels, base_profiles, base_clients = variants, labels, profiles, policy_clients
        groups = [args.bodies[k:k + batch] for k in range(0, len(args.bodies), batch)]
        for group in groups:
            prepared_cells.clear()
            placement_timing.update(searches=0, cache_hits=0, search_s=0.)
            slot_body = [b for b in group for _ in base_variants]
            variants = [v for _ in group for v in base_variants]
            labels = [lab for _ in group for lab in base_labels]
            profiles = [p for _ in group for p in base_profiles]
            policy_clients = [c for _ in group for c in base_clients]
            body = group[0] if len(group) == 1 else f"{group[0]}+{len(group) - 1}"
            _, training_args, _ = pretrain_wang.prepare(
                ["teacher", "--region", "13", "--seed", "1", "--obs-mode", "wang_static_arm", "--no-obs-augment"])
            n = len(variants)
            material = {} if args.cloth_density is None else {"cloth_density": args.cloth_density}
            if args.cloth_strain_rate is not None:
                material["cloth_strain_rate"] = args.cloth_strain_rate
            if args.friction is not None:
                material["friction"] = args.friction
            cfg = replace(train_sac.dressing_config(training_args),
                          cells=tuple((args.garment, b) for b in slot_body), cell_source="live",
                          collision_geometry=args.collision_geometry,
                          horizon=args.steps + args.hold + 10, seed=1, show_viewer=args.show_viewer, decision_watchdog=False,
                          clip_rotation_to_yz=False, anchor_count=48, contact_force_readout=True,
                          **material)
            if args.body_fit_filter is not None:
                cfg = replace(cfg, live=replace(cfg.live, body=replace(cfg.live.body, fit_filter_m=args.body_fit_filter)))
            if batch > 1:
                # Pre-check every body's placement alone with the world's own live config, so one illegal
                # start drops that body, not the world; the (garment, body) cache hands the result to the build.
                # Bodies on the CPU: torch touching CUDA before Genesis initialises breaks Genesis. The world
                # build reuses these cached cells, so every slot sees exactly the body checked here.
                probe = dressing_live.LiveCellFactory(replace(cfg.live, body=replace(cfg.live.body, device="cpu")))
                kept = []
                for body in group:
                    try:
                        hung_build(probe, args.garment, body)
                        kept.append(body)
                    except RuntimeError as exc:
                        row = {"body": body, "reason": str(exc)}
                        skipped.append(row)
                        save_json(args.out / "skipped.json", skipped)
                        print(f"[skip] {json.dumps(row)}", flush=True)
                if not kept:
                    continue
                body = group[0] if len(group) == 1 else f"{group[0]}+{len(group) - 1}"
                if kept != list(group):
                    group = kept
                    slot_body = [b for b in group for _ in base_variants]
                    variants = [v for _ in group for v in base_variants]
                    labels = [lab for _ in group for lab in base_labels]
                    profiles = [p for _ in group for p in base_profiles]
                    policy_clients = [c for _ in group for c in base_clients]
                    body = group[0] if len(group) == 1 else f"{group[0]}+{len(group) - 1}"
                    n = len(variants)
                    cfg = replace(cfg, cells=tuple((args.garment, b) for b in slot_body))
            print(f"[collect] building body={body} variants={labels}", flush=True)
            try:
                setup_started = time.monotonic()
                env = GenesisIPCDressingEnv(cfg, num_envs=n)
                setup_s = time.monotonic() - setup_started
            except RuntimeError as exc:
                if "No legal gravity-hung placement" not in str(exc):
                    raise
                row = {"body": body, "bodies": list(group), "reason": str(exc)}
                skipped.append(row)
                save_json(args.out / "skipped.json", skipped)
                print(f"[skip] {json.dumps(row)}", flush=True)
                continue
            try:
                obs = env.reset([args.seed] * n)
                env._arm_force_summaries()  # Initializes the lazy contact exporter before state zero.
                spec = ObsSpec(training_args.point_budget)
                def slot_geometry(c):
                    tri = c.cloth[c.faces]
                    area = .5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
                    m = np.zeros(len(c.cloth))
                    np.add.at(m, c.faces.ravel(), np.repeat(area / 3, 3))
                    m *= 2 * cfg.cloth_thickness * cfg.cloth_density
                    sec = SleeveSections(*sleeve_template, c.opening_idx) if sleeve_template is not None else None
                    fi, sh = np.asarray(c.finger), np.asarray(c.shoulder)
                    return dict(cell=c, sleeve=sec, landmarks=np.stack([c.finger, c.elbow, c.shoulder]),
                                finger=fi, shoulder=sh, axis=sh - fi, masses=m)
                slots = [slot_geometry(c) for c in env.cells]
                cell, sleeve, masses = slots[0]["cell"], slots[0]["sleeve"], slots[0]["masses"]
                finger, shoulder = slots[0]["finger"], slots[0]["shoulder"]
                planner = None
                if profiles[0].lookahead_interval:
                    from ipc_action_filter import IPCActionFilter
                    planner = IPCActionFilter(env, sleeve, masses, strength_gain=profiles[0].grasp_strength_gain)
                    if args.macro_recovery:
                        from ipc_macro_filter import IPCMacroFilter
                        planner = IPCMacroFilter(env, sleeve, masses, strength_gain=profiles[0].grasp_strength_gain,
                                                 horizon=args.macro_horizon)
                macro_queue, progress_history, macro_cooldown_until = [], [], 0
                body_dirs = {}
                for i, b in enumerate(slot_body):
                    if b not in body_dirs:
                        body_dirs[b] = args.out / f"body_{b}_seed_{args.seed}"
                        body_dirs[b].mkdir()
                        save_json(body_dirs[b] / "config.json", dict(config=cfg.to_dict(), placement=placements[str(b)],
                                                                     mass_kg=float(slots[i]["masses"].sum()),
                                                                     world_bodies=list(group)))
                body_dir = body_dirs[slot_body[0]]
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
                        stiffness = cfg.constraint_strength * profiles[i].grasp_strength_gain * slots[i]["masses"][indices] / cfg.dt ** 2
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
                        if slots[i]["sleeve"] is not None:
                            geometry = measure_sleeve(slots[i]["sleeve"], positions[i], slots[i]["landmarks"])
                            values.update(sleeve_wrapped=geometry["sections_wrapped" if args.sections_wrap else "sleeve_wrapped"],
                                          sleeve_cuff_s=geometry["cuff_s"],
                                          sleeve_proximal_upper_fraction=geometry["armhole_upper_fraction" if args.armhole_endpoint else "proximal_upper_fraction"])
                        for key, value in values.items():
                            buffers[i][key].append(value)

                record_state()
                started = time.monotonic()
                timing = dict(policy_s=0., environment_s=0., recording_s=0., lookahead_s=0.,
                              scene_setup_s=setup_s, placement=dict(placement_timing))
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
                            along = float((env._anchor[i] - slots[i]["finger"]) @ slots[i]["axis"] / (slots[i]["axis"] @ slots[i]["axis"]))
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
                    macro_step = False
                    if planner is not None and args.macro_recovery and success_at[0] is None and not completed[0]:
                        progress_now, _ = planner.progress(buffers[0]["positions"][-1])
                        progress_history.append(progress_now)
                        if macro_queue:
                            actions[0] = macro_queue.pop(0)
                            controllers[0], macro_step = 5, True
                        elif (step >= 50 and step >= macro_cooldown_until and len(progress_history) > 25
                              and progress_now - progress_history[-26] < args.macro_stall_m):
                            lookahead_started = time.monotonic()
                            plan, diagnostic = planner.plan(actions[0])
                            timing["lookahead_s"] += time.monotonic() - lookahead_started
                            with (body_dir / "macro.jsonl").open("a") as log:
                                log.write(json.dumps(dict(state=step, progress_m=progress_now, **diagnostic), default=float) + "\n")
                            print(f"[macro] body={body} state={step} progress={progress_now:.3f} -> {diagnostic['selected']}", flush=True)
                            if diagnostic["selected"] != "nominal":
                                macro_queue = list(plan)
                                actions[0] = macro_queue.pop(0)
                                controllers[0], macro_step = 5, True
                                macro_cooldown_until = step + len(plan) + 10
                            else:
                                macro_cooldown_until = step + 10
                    if (planner is not None and not macro_step and success_at[0] is None and not completed[0]
                            and step >= args.lookahead_from):
                        load = float(np.linalg.norm(buffers[0]["gripper_force"][-1]))
                        if step % profiles[0].lookahead_interval == 0 and load > args.lookahead_min_load:
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
                        sleeve_ok = slots[i]["sleeve"] is None or buffers[i]["sleeve_wrapped"][-1]
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
                    if slots[i]["sleeve"] is not None and success_at[i] is not None:
                        stable = stable and bool(data["sleeve_wrapped"][success_at[i]:].all())
                    valid = bool(np.all(data["grasp_valid"]))
                    accepted = stable and valid and failures[i] is None
                    force = np.linalg.norm(data["gripper_force"][1:], axis=1)
                    arm_force = np.linalg.norm(data["arm_force"][1:], axis=1)
                    j = i % len(base_variants)
                    record = dict(body=slot_body[i], variant=variant, replica=j if args.replicas > 1 else None,
                                  world_bodies=list(group),
                                  profile=profile_dict(profiles[i]),
                                  checkpoint_sha256=metadata["checkpoint_sha256"], simulator="Genesis+IPC",
                                  policy_device=args.policy_device,
                                  policy_server_pid=metadata.get('policy_server', {}).get('pid'),
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
                                  path=str((body_dirs[slot_body[i]] / f"{labels[i]}.npz").relative_to(args.out)))
                    full_body = (dict(human_vertices=env.collider_meshes[i][0],
                                      human_faces=env.collider_meshes[i][1])
                                 if args.collision_geometry == "full_body" else {})
                    c_i = slots[i]["cell"]
                    np.savez_compressed(body_dirs[slot_body[i]] / f"{labels[i]}.npz", **data, **full_body,
                                        grasp_indices=env._pickers[i]["anchor_idx"], initial_grasp_offsets=env._initial_offsets[i],
                                        faces=c_i.faces, arm_vertices=c_i.arm_points, arm_faces=c_i.arm_faces,
                                        opening_idx=c_i.opening_idx, finger=slots[i]["finger"], shoulder=slots[i]["shoulder"],
                                        elbow=np.asarray(c_i.elbow), metadata_json=json.dumps(record))
                    records.append(record)
                    print("[result] " + json.dumps(record), flush=True)
                for b, d in body_dirs.items():
                    save_json(d / "metrics.json", [r for r in records if r["body"] == b])
                    save_json(d / "timing.json", timing)
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
