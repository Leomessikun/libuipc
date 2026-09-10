"""Point-cloud SAC pretraining on the Genesis + libuipc manipulation tasks.

Usage (from the repository root, inside the Genesis environment)::

    PYTHONPATH=python python -m uipc_manip.train_sac --task cloth_drag --num-envs 4 \
        --total-transitions 20000 --work-dir output/uipc_manip --run-name cloth_drag_seed1

Dressing binds one (garment, body) cell to every slot for the life of the world:
``--body-seeds`` crosses ``--garments`` with a body axis, ``--heldout-bodies``
keeps whole bodies out of replay, and every evaluation round scores each cell
with ``heldout_``/``training_`` summaries (``cellplan``).

``--policy heuristic --eval-only`` runs the scripted reachability check that
every task must pass before SAC is worth running. ``--resume`` continues from a
checkpoint directory, and ``--eval-only`` with ``--resume`` plays a checkpoint
without modifying it.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import fields, is_dataclass, replace
from pathlib import Path

import numpy as np

from .cellplan import (
    checkpoint_score,
    complete_bodies,
    coverage_problems,
    plan_slots,
    select_heldout_bodies,
    summarize,
    training_rows,
)
from .curriculum import WANG_GARMENT_ORDER, curriculum_order, garment_curriculum_stage
from .dressing_env import DEFAULT_GARMENTS, DressingConfig, GenesisIPCDressingEnv
from .dressing_obs import RIG_MODES, DressingObsConfig
from .genesis_env import EnvConfig, GenesisIPCManipEnv, ViewerClosed
from .obs import ObsSpec, goal_rel, marker_centroid_rel
from .sac import (
    SACConfig,
    gradient_update_budget,
    wang_equivalent_alpha_lr,
    wang_equivalent_discount,
    wang_equivalent_reward_scale,
)
from .tasks import TASKS, heuristic_action


def _int_list(text: str) -> list[int]:
    """Parse a comma-separated list of integers such as ``0,1,2``."""
    return [int(token) for token in str(text).split(",") if token.strip()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", choices=[*sorted(TASKS), "dressing"], default="dressing")
    p.add_argument("--cell-source", choices=("cache", "live"), default="live", help="dressing: 'live' drapes each garment in libuipc and places it on a generated body, so every (garment, body) cell exists; 'cache' reads the Newton bake's 23 pre-worn cells over bodies 0-7.")
    body_axis = p.add_mutually_exclusive_group()
    body_axis.add_argument("--human", type=int, default=0, help="dressing: the single body whose cells fill every slot (a regional teacher); holds no body out.")
    body_axis.add_argument("--body-seeds", type=_int_list, default=None, help="dressing: comma-separated bodies, SMPL-X seeds for live cells or cached ids for cache cells; every (garment, body) cell gets a slot.")
    heldout_axis = p.add_mutually_exclusive_group()
    heldout_axis.add_argument("--heldout-bodies", type=int, default=None, help="dressing: whole bodies whose slots step and are evaluated but never feed replay, the greatest that carry every garment; default 2 for live and 1 for cache (always leaving a training body), 0 for a single body.")
    heldout_axis.add_argument("--heldout-body-seeds", type=_int_list, default=None, help="dressing: comma-separated held-out bodies, instead of --heldout-bodies.")
    p.add_argument("--allow-partial-cell-coverage", action="store_true", help="dressing: run although a cell has no slot, a requested garment or body has no cell, or a round has fewer evaluation episodes than cells; smoke tests only.")
    p.add_argument("--garments", type=str, nargs="+", default=list(DEFAULT_GARMENTS), help="dressing: garments of the cell grid, each crossed with every body.")
    p.add_argument("--anchor-count", type=int, default=48, help="dressing: cuff vertices held by the picker; 12 is Newton's patch, which lets the cuff slip off the hand on live cells.")
    p.add_argument("--cuff-strength", type=float, default=1.0e4, help="dressing: soft position constraint strength_rate of the held cuff; 100 lets the garment detach from the tool.")
    p.add_argument("--no-obs-augment", action="store_true", help="dressing: disable camera jitter and dropout.")
    p.add_argument("--obs-mode", choices=("visible_dual", "visible_single", "xray", *RIG_MODES), default="visible_dual", help="dressing: the cameras behind the point cloud; visible_dual is the FMVP preset, wang_static_arm Wang's single camera with the arm captured before dressing, stretch3_head and stretch3_head_wrist a Hello Robot Stretch 3's head and gripper cameras (dressing_obs.py).")
    p.add_argument("--cloth-shear-ratio", type=float, default=None, help="dressing: shear modulus as a fraction of the stretch modulus. libuipc's shear term carries no thickness factor, so the default shared modulus is about 3,300 times stiffer than stretch; 0.01 is what reproduces the reference drape.")
    p.add_argument("--cloth-youngs", type=float, default=None, help="dressing: stretch Young's modulus [Pa]; 6e3 matches the reference drape, 6e4 is the historical setting.")
    p.add_argument("--cloth-bending", type=float, default=None, help="dressing: discrete-shell bending stiffness; 0.1 matches the reference drape, 10 is the historical setting.")
    p.add_argument("--garment-curriculum-interval", type=int, default=0, help="dressing: vector steps between admitting one more garment's slots to replay, easiest first (Wang's curriculum_update_freq); 0 trains on every garment from the start.")
    p.add_argument("--garment-curriculum-order", type=str, default=",".join(WANG_GARMENT_ORDER), help="dressing: comma-separated garment names, easiest first; garments not named are appended.")
    p.add_argument("--num-envs", type=int, default=32, help="Deformable and robot copies solved together in one IPC world.")
    p.add_argument("--horizon", type=int, default=None, help="Decisions per episode; 900 for dressing, 150 otherwise.")
    p.add_argument("--action-repeat", type=int, default=None, help="Simulation steps per decision; 1 for dressing (the reference decides at 60 Hz), 5 otherwise. The tool speed cap is held over the whole decision.")
    p.add_argument("--dt", type=float, default=1.0 / 60.0, help="dressing: simulation step [s]; --action-repeat x --dt is the decision period, 0.1 s in the reference configuration. The held cuff's physical stiffness is --cuff-strength x mass / dt^2 and the settle lasts --settle-steps steps, so a matched run at dt 1/30 passes --action-repeat 3 --cuff-strength 4e4 --settle-steps 15; none of them is scaled automatically. Horizon and discount count decisions and do not change.")
    p.add_argument("--max-translation", type=float, default=0.006)
    p.add_argument("--point-budget", type=int, default=None, help="Points per observation; 768 for dressing, 256 otherwise.")
    p.add_argument("--friction", type=float, default=0.6)
    p.add_argument("--settle-steps", type=int, default=None, help="Simulation steps the world settles before its reset snapshot; 40 for the manipulation tasks, the dressing config's 30 for dressing.")
    p.add_argument("--vis", action="store_true", help="Open the Genesis viewer (forces a single environment).")
    p.add_argument("--policy", choices=("sac", "heuristic", "random"), default="sac")
    p.add_argument("--total-transitions", type=int, default=20_000, help="Additional transitions admitted to replay (including on resume); curriculum-excluded slots do not consume this budget.")
    p.add_argument("--init-steps", type=int, default=0, help="Vector steps of uniform random actions before the policy acts.")
    p.add_argument("--updates-per-step", type=int, default=0, help="Gradient updates per vector step; 0 = one per collected transition.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--replay-capacity", type=int, default=200_000)
    p.add_argument("--discount", type=float, default=None, help="Override the horizon-equivalent Wang discount.")
    p.add_argument("--alpha-lr", type=float, default=None)
    p.add_argument("--actor-lr", type=float, default=1.0e-4)
    p.add_argument("--critic-lr", type=float, default=1.0e-4)
    p.add_argument("--hidden-dim", type=int, default=1024)
    p.add_argument("--actor", choices=("wang-flow", "flat"), default="wang-flow", help="wang-flow is the reference tool-point actor.")
    p.add_argument("--algo", choices=("sac", "flashsac"), default="sac", help="Scalar reference critic or bounded categorical critic.")
    p.add_argument("--critic-input", choices=("points", "privileged"), default="points", help="dressing: the critic encodes the point cloud (reference) or reads the simulator's privileged state.")
    p.add_argument("--encoder-precision", choices=("fp32", "bf16"), default="fp32", help="Run the point encoders under bfloat16 autocast; heads, targets and losses stay fp32.")
    p.add_argument("--teacher-checkpoints", nargs="+", default=None, help="dressing: Wang's distillation from regional teachers, one checkpoint per arm-pose region; each replay row is pulled toward its slot's region teacher.")
    p.add_argument("--distill-weight", type=float, default=0.01, help="Weight of Wang's teacher loss (paper 0.01, reference launcher 0.002); used only with --teacher-checkpoints.")
    p.add_argument("--encoder", choices=("pointnet2", "transformer"), default="pointnet2")
    p.add_argument("--num-bins", type=int, default=51, help="flashsac: value atoms per critic head.")
    p.add_argument("--min-v", type=float, default=-50.0, help="flashsac: lowest value atom.")
    p.add_argument("--max-v", type=float, default=50.0, help="flashsac: highest value atom.")
    p.add_argument("--sa-neighbors", type=int, nargs="+", default=[8, 16], help="Ball-query neighbours per set-abstraction level.")
    p.add_argument("--point-jitter", type=float, default=0.0, help="Per-point jitter [m] applied to replay samples.")
    p.add_argument("--grad-clip-max-norm", type=float, default=0.0)
    p.add_argument("--min-alpha", type=float, default=0.0)
    p.add_argument("--init-temperature", type=float, default=0.1, help="Initial SAC temperature; the reference uses 0.1 at a 150-step horizon.")
    p.add_argument("--eval-freq", type=int, default=500, help="Vector steps between evaluations (0 disables).")
    p.add_argument("--num-eval-episodes", type=int, default=None, help="Episodes per evaluation round, rounded up to a whole number per slot; default one per slot, which is one deterministic episode per cell when every cell has its own slot.")
    p.add_argument("--checkpoint-interval", type=int, default=500)
    p.add_argument("--log-interval", type=int, default=20)
    p.add_argument("--work-dir", type=str, default="output/uipc_manip", help="Runs land here; the repository ignores output/.")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--resume", type=str, default=None, help="Checkpoint file to load (weights, temperature, optimizers).")
    p.add_argument("--resume-replay", type=str, default=None, help="Replay snapshot directory saved with the checkpoint.")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--save-trajectories", action="store_true", help="Store evaluation trajectories as .npz files.")
    return p


def resolve_defaults(args) -> None:
    """Fill the task-dependent launcher defaults in place."""
    if args.horizon is None:
        args.horizon = 900 if args.task == "dressing" else 150
    if args.action_repeat is None:
        args.action_repeat = 1 if args.task == "dressing" else 5
    if args.point_budget is None:
        args.point_budget = 768 if args.task == "dressing" else 256
    if args.settle_steps is None and args.task != "dressing":
        # Dressing keeps its config's own settle unless the flag is given.
        args.settle_steps = 40
    if args.num_eval_episodes is None:
        args.num_eval_episodes = args.num_envs


def restore_resume_args(args, argv: list[str], payload: dict) -> SACConfig:
    """Restore the saved experiment before constructing the simulator or agent.

    An explicit incompatible option is an error on training continuation; playback
    may override environment settings, but must still match the network protocol.
    Older checkpoints recover the launcher fields present in their env metadata.
    The cell-plan options are pinned the same way; the plan itself also depends on
    the cell library, so ``make_env`` compares it with the checkpoint's
    (``reconcile_resume_cell_plan``) before building the world.
    """
    explicit = {token.split("=", 1)[0][2:].replace("-", "_") for token in argv if token.startswith("--")}
    metadata = payload.get("metadata", {})
    saved = dict(metadata.get("training_args", {}))
    env = metadata.get("env", {})
    args._resume_env = env
    args._explicit_options = sorted(explicit)
    saved.update({key: env[key] for key in ("human", "garments", "horizon", "action_repeat", "dt", "settle_steps", "point_budget", "anchor_count") if key in env})
    if metadata.get("task") == "dressing":
        # Checkpoints from before the cell plan all trained on the bake cache.
        saved["cell_source"] = env.get("cell_source", "cache")
        args._resume_cell_plan = {key: metadata[key] for key in ("cells", "heldout_cells", "heldout_bodies", "live") if key in metadata}
        if args.eval_only and explicit & {"human", "body_seeds", "garments", "cell_source"}:
            # Playback on another cell axis plans that axis from its own defaults: an
            # explicit --human replaces the saved body seeds, and a saved held-out row
            # names bodies of the old axis. make_env then scores every cell the
            # checkpoint never trained on as held out.
            for key in {"heldout_bodies", "heldout_body_seeds", *(("body_seeds",) if "human" in explicit else ())} - explicit:
                saved.pop(key, None)
    if "constraint_strength" in env and metadata.get("task") == "dressing":
        saved["cuff_strength"] = env["constraint_strength"]
    if "augment_obs" in env:
        saved["no_obs_augment"] = not env["augment_obs"]
    if isinstance(env.get("obs"), dict) and "mode" in env["obs"]:
        saved["obs_mode"] = env["obs"]["mode"]
    if metadata.get("task") != "dressing":
        saved.update({key: env[key] for key in ("max_translation", "friction") if key in env})
    saved.update({key: metadata[key] for key in ("task", "seed", "num_envs") if key in metadata})
    cfg = SACConfig.from_dict(payload["sac_config"])
    cfg_names = {"actor": "actor_type", "point_jitter": "point_jitter_scale", "grad_clip_max_norm": "grad_clip_max_norm"}
    for key in ("discount", "alpha_lr", "init_temperature", "actor_lr", "critic_lr", "hidden_dim", "batch_size", "min_alpha", "algo", "num_bins", "min_v", "max_v", "critic_input", "encoder_precision", "distill_weight"):
        cfg_names[key] = key
    saved.update({key: getattr(cfg, name) for key, name in cfg_names.items()})
    saved.update(encoder=cfg.encoder.kind, sa_neighbors=cfg.encoder.sa_neighbors)
    network_keys = {"actor", "encoder", "hidden_dim", "point_budget", "sa_neighbors", "algo", "num_bins", "min_v", "max_v", "critic_input"}
    for key, value in saved.items():
        if not hasattr(args, key):
            continue
        current = getattr(args, key)
        if key in explicit:
            matches = list(current) == list(value) if isinstance(current, (list, tuple)) and isinstance(value, (list, tuple)) else current == value
            if (not args.eval_only or key in network_keys) and not matches:
                raise ValueError(f"Checkpoint requires saved --{key.replace('_', '-')}={value!r}; got {current!r}")
        else:
            setattr(args, key, value)
    return cfg


def restore_env_config(cfg, args):
    """Retain saved physics/reward/camera fields even when they have no CLI flag."""
    saved = getattr(args, "_resume_env", None)
    if not saved:
        return cfg
    aliases = {"constraint_strength": "cuff_strength", "augment_obs": "no_obs_augment"}
    explicit = set(getattr(args, "_explicit_options", ()))

    def restore(current, data, top_level=False):
        values = {}
        for field in fields(current):
            name = field.name
            if name not in data or (top_level and (name in {"show_viewer", "logging_level", "workspace", "seed"} or aliases.get(name, name) in explicit)):
                continue
            before, value = getattr(current, name), data[name]
            if is_dataclass(before):
                value = restore(before, value)
            elif isinstance(before, Path):
                value = Path(value)
            elif isinstance(before, tuple):
                value = tuple(value)
            values[name] = value
        return replace(current, **values)

    return restore(cfg, saved, top_level=True)


def validate_resume_replay(payload: dict, replay_metadata: dict) -> None:
    if int(replay_metadata.get("step", -1)) != int(payload["step"]):
        raise ValueError("Resume replay must come from the same checkpoint step")
    saved_scale = payload.get("metadata", {}).get("reward_scale")
    replay_scale = replay_metadata.get("reward_scale")
    if saved_scale is not None and replay_scale is not None and saved_scale != replay_scale:
        raise ValueError("Resume replay reward scale does not match the checkpoint")


def build_sac_config(args) -> SACConfig:
    """The reference SAC settings with the horizon-equivalent discount and temperature learning rate."""
    discount = wang_equivalent_discount(args.horizon) if args.discount is None else float(args.discount)
    alpha_lr = wang_equivalent_alpha_lr(args.horizon) if args.alpha_lr is None else float(args.alpha_lr)
    sac_cfg = SACConfig(
        discount=discount,
        alpha_lr=alpha_lr,
        init_temperature=args.init_temperature,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        grad_clip_max_norm=args.grad_clip_max_norm,
        min_alpha=args.min_alpha,
        point_jitter_scale=args.point_jitter,
        actor_type=args.actor,
        algo=args.algo,
        num_bins=args.num_bins,
        min_v=args.min_v,
        max_v=args.max_v,
        critic_input=args.critic_input,
        encoder_precision=args.encoder_precision,
        distill_weight=float(args.distill_weight) if args.teacher_checkpoints else 0.0,
    )
    neighbors = [int(n) for n in args.sa_neighbors]
    if len(neighbors) == 1:
        neighbors = neighbors * 2
    sac_cfg.encoder = replace(sac_cfg.encoder, kind=args.encoder, sa_neighbors=neighbors)
    return sac_cfg


def load_teachers(paths, spec: ObsSpec, action_dim: int, device) -> dict:
    """Regional teacher actors, keyed by the one arm-pose region each teacher's training cells share."""
    from .dressing_body import pose_region
    from .sac import SACAgent, SACConfig

    teachers = {}
    for path in paths:
        payload = SACAgent.read_checkpoint(path)
        meta = payload.get("metadata", {})
        heldout = {int(s) for s in meta.get("heldout_slots", [])}
        regions = {pose_region(b) for slot, (_, b) in enumerate(meta.get("cells", [])) if slot not in heldout}
        if len(regions) != 1 or None in regions:
            raise ValueError(f"Teacher {path} trained on arm-pose regions {sorted(regions, key=str)}; a regional teacher has exactly one")
        region = regions.pop()
        if region in teachers:
            raise ValueError(f"Two teachers for arm-pose region {region}")
        teacher = SACAgent(spec, action_dim, SACConfig.from_dict(payload["sac_config"]), device)
        teacher.load(path, load_optimizers=False)
        teachers[region] = teacher.actor
    return teachers


def cell_bodies(args) -> list[int]:
    """The body axis: ``--body-seeds``, else the single ``--human``."""
    seeds = getattr(args, "body_seeds", None)
    return sorted({int(b) for b in seeds}) if seeds else [int(args.human)]


def library_cells(args, cfg: DressingConfig) -> list[tuple[str, int]]:
    """Every cell the configured source can build for the requested garments and bodies."""
    garments, bodies = set(args.garments), set(cell_bodies(args))
    if args.cell_source == "cache":
        from .dressing_assets import DressingCache

        available = DressingCache(cfg.cache).cells
    else:
        from .dressing_live import available_garments

        available = [(g, b) for g in available_garments(cfg.live) for b in bodies]
    return sorted({(str(g), int(b)) for g, b in available if g in garments and int(b) in bodies})


def plan_cells(args, library) -> dict:
    """Bind one (garment, body) cell to every slot and hold whole bodies out of replay.

    Held-out cells take the leading slots and the rest are stratified by garment
    (``cellplan.plan_slots``). Unless ``--allow-partial-cell-coverage`` is given
    (``--vis`` implies it, since it forces one slot), a library cell without a slot,
    a requested garment or body without a cell, or a round of fewer evaluation
    episodes than cells is refused before any world is built. Duplicate slots are
    allowed, so a single-body regional teacher still cycles its garments over every
    slot.
    """
    library = sorted({(str(g), int(b)) for g, b in library})
    bodies = cell_bodies(args)
    problems = coverage_problems(library, args.garments, bodies, args.num_envs, args.num_eval_episodes)
    if problems and not (args.allow_partial_cell_coverage or args.vis):
        raise ValueError(
            "The slots do not cover the cell library: " + "; ".join(problems) + ". Pass --allow-partial-cell-coverage only for a smoke test."
        )
    if not library:
        raise ValueError(f"No {args.cell_source} cell exists for garments {list(args.garments)} and bodies {bodies}")
    garments = sorted({g for g, _ in library})
    if args.heldout_body_seeds is not None:
        heldout = sorted({int(b) for b in args.heldout_body_seeds})
        incomplete = sorted(set(heldout) - set(complete_bodies(library, garments)))
        if incomplete:
            raise ValueError(f"Held-out bodies {incomplete} are not complete rows of the cell library, which needs every garment of {garments}")
    else:
        count = args.heldout_bodies
        if count is None:
            count = 0 if len(bodies) == 1 else min(2 if args.cell_source == "live" else 1, len(bodies) - 1)
        heldout = select_heldout_bodies(library, garments, count)
    slots, heldout_slots = plan_slots(library, args.num_envs, heldout)
    return {"cells": [[g, b] for g, b in slots], "heldout_slots": heldout_slots, "heldout_bodies": heldout, "cell_source": args.cell_source}


def reconcile_resume_cell_plan(args, plan: dict) -> dict:
    """Hold a resumed run to its checkpoint's cell plan.

    A training resume onto other cells, another slot binding or another held-out
    row is refused. Playback may differ: the difference is printed, and every slot
    whose cell the checkpoint never trained on is scored as held out, so a
    zero-shot round on fresh bodies is not reported as ``training_``.
    """
    saved = getattr(args, "_resume_cell_plan", None) or {}
    if "cells" not in saved:
        return plan
    old = [(str(g), int(b)) for g, b in saved["cells"]]
    new = [(str(g), int(b)) for g, b in plan["cells"]]
    old_heldout = sorted(int(b) for b in saved.get("heldout_bodies", []))
    if old == new and old_heldout == list(plan["heldout_bodies"]):
        return plan
    added, removed = sorted(set(new) - set(old)), sorted(set(old) - set(new))
    changes = [f"new cells {added}"] if added else []
    changes += [f"dropped cells {removed}"] if removed else []
    changes += ["the same cells bound to other slots"] if not added and not removed and old != new else []
    changes += [f"held-out bodies {old_heldout} -> {list(plan['heldout_bodies'])}"] if old_heldout != list(plan["heldout_bodies"]) else []
    if not args.eval_only:
        raise ValueError(f"Checkpoint was trained on another cell plan ({'; '.join(changes)}); a training resume must keep its cells and held-out bodies")
    trained = set(old) - {(str(g), int(b)) for g, b in saved.get("heldout_cells", [])}
    heldout_slots = [i for i, cell in enumerate(new) if cell not in trained]
    unseen_bodies = sorted(b for b in {b for _, b in new} if all(cell not in trained for cell in new if cell[1] == b))
    print(
        f"[uipc-manip] eval-only cell plan differs from the checkpoint: {'; '.join(changes)}; "
        f"{len(heldout_slots)} of {len(new)} slots hold cells it never trained on and are scored as held out",
        flush=True,
    )
    return {**plan, "heldout_slots": heldout_slots, "heldout_bodies": unseen_bodies}


PLACEMENT_KEYS = ("pre_insertion", "hang_as_baked", "hang_as_baked_garments", "sleeve_outward_garments")
"""Live-cell fields that decide where a garment starts. A checkpoint written before a per-garment
list existed placed every garment without it, so a missing list reads as empty."""


def reconcile_resume_placement(args, live: dict | None) -> None:
    """Hold a resumed live-cell run to its checkpoint's garment placement.

    The cell plan names (garment, body) pairs, not where each garment starts, so a per-garment
    placement change (the gown and tshirt_68 in their baked hang, tshirt_4 and tshirt_392 sleeve
    outward) would otherwise resume training on another start state. Training refuses; playback
    prints the difference.
    """
    saved = (getattr(args, "_resume_cell_plan", None) or {}).get("live")
    if not saved or not live:
        return

    def read(block: dict, key: str):
        value = block.get(key, [] if key.endswith("_garments") else None)
        return sorted(str(g) for g in value) if isinstance(value, (list, tuple)) else value

    changes = [f"{key} {read(saved, key)} -> {read(live, key)}" for key in PLACEMENT_KEYS if read(saved, key) is not None and read(saved, key) != read(live, key)]
    if not changes:
        return
    if not args.eval_only:
        raise ValueError(f"Checkpoint placed its garments differently ({'; '.join(changes)}); a training resume must keep the placement")
    print(f"[uipc-manip] eval-only placement differs from the checkpoint: {'; '.join(changes)}", flush=True)


def describe_cell_plan(plan: dict) -> str:
    """The ``[uipc-manip distribution]`` line: what the world trains on and what it holds out."""
    cells = [(str(g), int(b)) for g, b in plan["cells"]]
    heldout = set(plan["heldout_slots"])
    training_cells = {cell for i, cell in enumerate(cells) if i not in heldout}
    heldout_cells = {cell for i, cell in enumerate(cells) if i in heldout}
    return (
        f"[uipc-manip distribution] source={plan['cell_source']} slots={len(cells)} unique_cells={len(set(cells))} "
        f"training_cells={len(training_cells)} heldout_cells={len(heldout_cells)} garments={sorted({g for g, _ in cells})} "
        f"bodies={sorted({b for _, b in cells})} heldout_bodies={list(plan['heldout_bodies'])}"
    )


def built_cell_plan(env, plan: dict | None) -> tuple[list[tuple[str, int]] | None, list[int]]:
    """Slot cells and held-out slots of the world ``make_env`` built from ``plan``.

    Without a plan (non-dressing tasks) every slot trains and evaluation reads each
    slot's cell from its episode infos.
    """
    if plan is None:
        return None, []
    cells = [(str(g), int(b)) for g, b in plan["cells"]]
    built = [(str(d["garment"]), int(d["human"])) for d in env.descriptions if "human" in d]
    if built and built != cells:
        raise RuntimeError(f"The world bound cells {built} to its slots but the plan is {cells}")
    return cells, [int(i) for i in plan["heldout_slots"]]


def dressing_config(args) -> DressingConfig:
    """The dressing settings the launcher flags ask for, with a resumed run's saved fields restored."""
    cfg = DressingConfig(
        human=args.human,
        garments=tuple(args.garments),
        cell_source=args.cell_source,
        horizon=args.horizon,
        action_repeat=args.action_repeat,
        dt=args.dt,
        point_budget=args.point_budget,
        anchor_count=args.anchor_count,
        constraint_strength=args.cuff_strength,
        **{
            name: value
            for name, value in (
                ("cloth_shear_ratio", args.cloth_shear_ratio),
                ("cloth_youngs", args.cloth_youngs),
                ("cloth_bending_stiffness", args.cloth_bending),
                ("settle_steps", args.settle_steps),
            )
            if value is not None
        },
        seed=args.seed,
        augment_obs=not args.no_obs_augment,
        obs=DressingObsConfig(mode=args.obs_mode),
        show_viewer=bool(args.vis),
    )
    cfg = restore_env_config(cfg, args)
    if "obs_mode" in set(getattr(args, "_explicit_options", ())):
        # Playback may look through another rig; the saved camera fields stay.
        cfg = replace(cfg, obs=replace(cfg.obs, mode=args.obs_mode))
    return cfg


def make_env(args):
    if args.task == "dressing":
        cfg = dressing_config(args)
        # The plan, not the checkpoint's env record, decides which cell each slot holds.
        plan = reconcile_resume_cell_plan(args, plan_cells(args, library_cells(args, cfg)))
        cfg = replace(cfg, cells=tuple((g, b) for g, b in plan["cells"]), cell_source=args.cell_source)
        plan["live"] = cfg.live.to_dict() if cfg.cell_source == "live" else None
        reconcile_resume_placement(args, plan["live"])
        args._cell_plan = plan
        print(describe_cell_plan(plan), flush=True)
        return GenesisIPCDressingEnv(cfg, num_envs=args.num_envs)
    return GenesisIPCManipEnv(restore_env_config(env_config(args), args), num_envs=args.num_envs)


def env_config(args) -> EnvConfig:
    return EnvConfig(
        task=args.task,
        horizon=args.horizon,
        action_repeat=args.action_repeat,
        max_translation=args.max_translation,
        point_budget=args.point_budget,
        seed=args.seed,
        friction=args.friction,
        settle_steps=args.settle_steps,
        show_viewer=bool(args.vis),
    )


def heuristic_actions(obs: np.ndarray, spec: ObsSpec, max_translation: float) -> np.ndarray:
    return np.stack([heuristic_action(marker_centroid_rel(o, spec), goal_rel(o, spec), max_translation) for o in obs])


def evaluate(
    env, policy, spec: ObsSpec, args, episodes: int, trajectory_dir: Path | None = None,
    *, slot_cells: list[tuple[str, int]] | None = None, heldout_slots=(),
) -> dict:
    """Play deterministic episodes and report success and distance statistics.

    Every slot is scored whatever the curriculum stage, so rounds stay comparable.
    ``slot_cells`` binds a (garment, body) cell to each slot and ``heldout_slots``
    names the slots held out of replay; the summary then adds per-cell rates and the
    ``heldout_``/``training_`` blocks of ``cellplan.summarize``. Without a plan each
    slot's cell is read from its episode infos.
    """
    if episodes < 1:
        raise ValueError("Evaluation requires at least one episode")
    requested_episodes = episodes
    # Fixed garment slots must each contribute equally, including when the
    # requested count is smaller than the vector width or failures finish early.
    episodes_per_slot = int(np.ceil(episodes / env.num_envs))
    episodes = episodes_per_slot * env.num_envs
    slot_finished = np.zeros(env.num_envs, dtype=np.int64)
    # A fresh seed block per round: the physics and the deterministic actor repeat
    # exactly, so reusing one block makes every round the same observation noise.
    seed_base = args.seed * 1000 + 97 * int(getattr(args, "_eval_round", 0))
    args._eval_round = int(getattr(args, "_eval_round", 0)) + 1
    obs = env.reset([seed_base + i for i in range(env.num_envs)])
    finished: list[dict] = []
    returns = np.zeros(env.num_envs)
    max_tracking = np.zeros(env.num_envs)
    early_turn_seen = np.zeros(env.num_envs, dtype=bool)
    metric_keys = tuple(getattr(env, "metric_keys", ()))
    running_max = {k: np.full(env.num_envs, -np.inf) for k in metric_keys}
    trajectories = [[] for _ in range(env.num_envs)] if trajectory_dir is not None else None
    episode_index = 0
    while len(finished) < episodes:
        actions = policy(obs, True)
        if trajectories is not None:
            for i, state in enumerate(env.states()):
                trajectories[i].append(state)
        obs, rewards, dones, infos = env.step(actions)
        returns += rewards
        for i, info in enumerate(infos):
            max_tracking[i] = max(max_tracking[i], float(info.get("tracking_error", 0.0)))
            early_turn_seen[i] |= bool(info.get("early_turn", False))
            for k in metric_keys:
                if k in info:
                    running_max[k][i] = max(running_max[k][i], float(info[k]))
            if dones[i]:
                record = {
                    "success": bool(info.get("success", False)),
                    "distance": float(info.get("distance", np.nan)),
                    "return": float(returns[i]),
                    "max_tracking_error": float(max_tracking[i]),
                    "sim_error": bool(info.get("sim_error", False)),
                    "early_turn": bool(early_turn_seen[i]),
                }
                # FMVP Appendix A.1 keeps a trajectory when it ends dressed and never cut the elbow.
                record["paper_filter"] = bool(record["success"] and not record["early_turn"])
                for k in metric_keys:
                    record[f"final_{k}"] = float(info.get(k, np.nan))
                    record[f"max_{k}"] = float(running_max[k][i])
                    running_max[k][i] = -np.inf
                record["slot"] = i
                for key in ("garment", "cell"):
                    if key in info:
                        record[key] = str(info[key])
                if "human" in info:
                    record["human"] = int(info["human"])
                if slot_finished[i] < episodes_per_slot:
                    finished.append(record)
                    slot_finished[i] += 1
                    if trajectories is not None:
                        static = None
                        if hasattr(env, "arm_meshes"):  # each slot has its own body
                            static = {"vertices": env.arm_meshes[i][0], "faces": env.arm_meshes[i][1]}
                        elif hasattr(env, "arm_vertices"):
                            static = {"vertices": env.arm_vertices, "faces": env.arm_faces}
                        _save_trajectory(trajectory_dir, episode_index, trajectories[i], env.descriptions[i], static)
                        episode_index += 1
                returns[i] = 0.0
                max_tracking[i] = 0.0
                early_turn_seen[i] = False
                if trajectories is not None:
                    trajectories[i] = []
    distances = np.array([r["distance"] for r in finished])
    summary = {
        "requested_episodes": requested_episodes,
        "episodes": len(finished),
        "success_rate": float(np.mean([r["success"] for r in finished])),
        "mean_final_distance": float(np.nanmean(distances)),
        "mean_return": float(np.mean([r["return"] for r in finished])),
        "max_tracking_error": float(max(r["max_tracking_error"] for r in finished)),
        "sim_errors": int(sum(r["sim_error"] for r in finished)),
        "early_turn_rate": float(np.mean([r["early_turn"] for r in finished])),
        "paper_filter_rate": float(np.mean([r["paper_filter"] for r in finished])),
    }
    for k in metric_keys:
        summary[f"mean_final_{k}"] = float(np.nanmean([r[f"final_{k}"] for r in finished]))
        summary[f"mean_max_{k}"] = float(np.nanmean([r[f"max_{k}"] for r in finished]))
    if slot_cells is None:
        slot_cells = _cells_from_records(finished, env.num_envs)
    if slot_cells is not None:
        ratio_keys = tuple(k for k in ("upperarm_ratio", "forearm_ratio") if k in metric_keys)
        summary.update(summarize(finished, slot_cells, heldout_slots, ratio_keys))
    summary["records"] = finished
    return summary


def _cells_from_records(records: list[dict], num_envs: int) -> list[tuple[str, int | None]] | None:
    """Each slot's (garment, body) as its episode infos report it; ``None`` without garments."""
    cells: list = [None] * num_envs
    for record in records:
        if "garment" in record:
            cells[record["slot"]] = (record["garment"], record.get("human"))
    return None if any(cell is None for cell in cells) else cells


def _save_trajectory(directory: Path, index: int, states: list[dict], description: dict, static: dict | None = None) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    static = static or {}
    np.savez_compressed(
        directory / f"episode_{index:03d}.npz",
        **{f"static_{k}": np.asarray(v) for k, v in static.items()},
        positions=np.stack([s["positions"] for s in states]),
        tcp=np.stack([s["tcp"] for s in states]),
        goal=np.stack([s["goal"] for s in states]),
        marker_centroid=np.stack([s["marker_centroid"] for s in states]),
        qpos=np.stack([s["qpos"] for s in states]),
        faces=np.asarray(description["faces"], dtype=np.int32).reshape(-1, 3),
        edges=np.asarray(description["edges"], dtype=np.int32).reshape(-1, 2),
        radius=float(description["radius"]),
        deformable=str(description["deformable"]),
        task=str(description["task"]),
    )


def _set_viewer_caption(env, text: str) -> None:
    try:
        env.scene.viewer._pyrender_viewer.set_caption(text)
    except Exception:  # viewer backends without a caption API
        pass


def _hold_viewer(env, text: str) -> None:
    """Keep the Genesis viewer open after evaluation so the scene can be inspected."""
    _set_viewer_caption(env, text)
    viewer = getattr(env.scene, "viewer", None)
    if viewer is None:
        return
    while viewer.is_alive():
        viewer.update(force=True)
        time.sleep(0.03)


class CsvLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._fields: list[str] | None = None
        if self.path.exists() and self.path.stat().st_size:
            with self.path.open(newline="") as handle:
                self._fields = next(csv.reader(handle))

    def log(self, row: dict) -> None:
        new_fields = [key for key in row if self._fields is not None and key not in self._fields]
        if new_fields:
            with self.path.open(newline="") as handle:
                previous = list(csv.DictReader(handle))
            self._fields.extend(new_fields)
            temporary = self.path.with_suffix(self.path.suffix + ".tmp")
            with temporary.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=self._fields)
                writer.writeheader()
                writer.writerows(previous)
            temporary.replace(self.path)
        if self._fields is None:
            self._fields = list(row)
            with self.path.open("w", newline="") as handle:
                csv.DictWriter(handle, fieldnames=self._fields).writeheader()
        with self.path.open("a", newline="") as handle:
            csv.DictWriter(handle, fieldnames=self._fields, extrasaction="ignore").writerow(
                {k: row.get(k, "") for k in self._fields}
            )


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    args = build_parser().parse_args(argv)
    payload = None
    saved_sac_cfg = None
    if args.resume_replay and not args.resume:
        raise ValueError("--resume-replay requires --resume")
    if args.resume:
        from .sac import SACAgent

        payload = SACAgent.read_checkpoint(args.resume)
        saved_sac_cfg = restore_resume_args(args, argv, payload)
        if not args.eval_only and not args.resume_replay:
            raise ValueError("Training continuation requires --resume-replay from the same checkpoint step")
    if args.vis:
        args.num_envs = 1
    np.random.seed(args.seed)
    run_name = args.run_name or f"{args.task}_{args.policy}_seed{args.seed}"
    run_dir = Path(args.work_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    seeds = [args.seed * 100 + i for i in range(args.num_envs)]
    resolve_defaults(args)
    env = make_env(args)
    slot_cells, heldout_slots = built_cell_plan(env, getattr(args, "_cell_plan", None))
    spec = ObsSpec(args.point_budget)
    description = env.descriptions[0]
    print(
        f"[uipc-manip] task={args.task} envs={env.num_envs} obs_dim={env.obs_dim} build={description['build_seconds']:.1f}s "
        f"settle_displacement={description['settle_displacement_m']:.4f} m",
        flush=True,
    )
    # default=str: the dressing config's live-cell settings hold Path objects.
    (run_dir / "env.json").write_text(json.dumps(description, indent=2, default=str) + "\n")

    if args.policy == "heuristic":
        if args.task == "dressing":
            policy = lambda obs, deterministic: env.scripted_actions()  # noqa: E731
        else:
            policy = lambda obs, deterministic: heuristic_actions(obs, spec, args.max_translation)  # noqa: E731
        agent = None
    elif args.policy == "random":
        policy = lambda obs, deterministic: np.random.uniform(-1, 1, size=(obs.shape[0], env.action_dim))  # noqa: E731
        agent = None
    else:
        import torch

        from .sac import SACAgent

        torch.manual_seed(args.seed)
        sac_cfg = saved_sac_cfg if saved_sac_cfg is not None else build_sac_config(args)
        if sac_cfg.critic_input == "privileged":
            env_dim = int(getattr(env, "privileged_dim", 0))
            if env_dim <= 0:
                raise ValueError("--critic-input privileged needs an environment that reports a privileged state; only dressing does")
            if saved_sac_cfg is None:
                sac_cfg.privileged_dim = env_dim
            elif sac_cfg.privileged_dim != env_dim:
                raise ValueError(f"The checkpoint's critic reads a {sac_cfg.privileged_dim}-float state; this environment reports {env_dim}")
        agent = SACAgent(spec, env.action_dim, sac_cfg, args.device)
        if args.resume:
            payload = agent.load(args.resume, load_optimizers=not args.eval_only)
            saved_task = payload["metadata"].get("task")
            if saved_task is not None and saved_task != args.task:
                raise ValueError(f"Checkpoint was trained on task {saved_task!r}, not {args.task!r}")
            print(f"[uipc-manip] resumed {args.resume} at step {payload['step']}", flush=True)
        policy = lambda obs, deterministic: agent.act(obs, deterministic)  # noqa: E731

    trajectory_dir = run_dir / "trajectories" if args.save_trajectories else None
    if args.eval_only or agent is None:
        if args.vis:
            _set_viewer_caption(env, f"uipc_manip {args.task} | {args.policy} policy | blue: deformable, green: goal, red: marker centroid")
        try:
            metrics = evaluate(env, policy, spec, args, args.num_eval_episodes, trajectory_dir, slot_cells=slot_cells, heldout_slots=heldout_slots)
        except ViewerClosed:
            print("[uipc-manip] viewer closed; exiting", flush=True)
            return
        (run_dir / "eval.json").write_text(json.dumps(metrics, indent=2) + "\n")
        print(json.dumps({k: v for k, v in metrics.items() if k != "records"}, indent=2), flush=True)
        if args.vis:
            _hold_viewer(env, f"uipc_manip {args.task} | done: success {metrics['success_rate']:.0%} | drag to orbit, close window to exit")
        env.close()
        return

    from .replay import FlatReplayBuffer

    privileged = agent.cfg.critic_input == "privileged"
    replay = FlatReplayBuffer(
        env.obs_dim, env.action_dim, args.replay_capacity, args.batch_size, args.device, priv_dim=agent.cfg.privileged_dim if privileged else 0,
        labelled=bool(args.teacher_checkpoints),
    )
    reward_scale = float(payload.get("metadata", {}).get("reward_scale", wang_equivalent_reward_scale(agent.cfg.discount))) if payload is not None else wang_equivalent_reward_scale(agent.cfg.discount)
    start_step = 0
    if args.resume:
        start_step = int(payload["step"])
        if args.resume_replay:
            validate_resume_replay(payload, replay.load(args.resume_replay))
    # Wang's garment curriculum: slots of garments not yet admitted keep stepping but do not feed replay.
    slot_garments = [str(d.get("garment", "")) for d in env.descriptions]
    order = curriculum_order(args.garment_curriculum_order.split(","), [g for g in slot_garments if g]) if any(slot_garments) else []
    slot_rank = np.array([order.index(g) if g else 0 for g in slot_garments], dtype=np.int64)
    curriculum_interval = int(args.garment_curriculum_interval) if order else 0
    # Held-out slots step with the policy and are evaluated, but never feed replay.
    training_slot_mask = np.ones(env.num_envs, dtype=bool)
    training_slot_mask[heldout_slots] = False
    # Wang's distillation: a replay row carries its slot's arm-pose region, whose teacher pulls on the actor.
    slot_region = [d.get("pose_region") for d in env.descriptions]
    teacher_regions: list[int] = []
    if args.teacher_checkpoints:
        teachers = load_teachers(args.teacher_checkpoints, spec, env.action_dim, args.device)
        uncovered = sorted({slot_region[i] for i in range(env.num_envs) if training_slot_mask[i]} - set(teachers), key=str)
        if uncovered:
            raise ValueError(f"Training slots in arm-pose regions {uncovered} have no teacher among {sorted(teachers)}")
        agent.set_teachers(teachers)
        teacher_regions = sorted(teachers)
        print(f"[uipc-manip] distilling from teachers for arm-pose regions {teacher_regions}, weight {agent.cfg.distill_weight}", flush=True)
    metadata = {
        "task": args.task,
        "env": description["config"],
        "sac_config": agent.cfg.to_dict(),
        "reward_scale": reward_scale,
        "seed": args.seed,
        "num_envs": env.num_envs,
        "training_args": {key: getattr(args, key) for key in (
            "garment_curriculum_interval", "garment_curriculum_order", "updates_per_step", "replay_capacity", "init_steps",
            "cell_source", "body_seeds", "heldout_bodies", "heldout_body_seeds", "allow_partial_cell_coverage", "teacher_checkpoints",
        )},
        "curriculum_order": order,
        "teacher_regions": teacher_regions,
    }
    plan = getattr(args, "_cell_plan", None)
    if plan is not None:
        # restore_resume_args and reconcile_resume_cell_plan hold a resume to these.
        metadata.update(
            cells=plan["cells"], heldout_slots=plan["heldout_slots"], heldout_cells=[plan["cells"][i] for i in plan["heldout_slots"]],
            heldout_bodies=plan["heldout_bodies"], cell_source=plan["cell_source"], live=plan["live"],
        )
    (run_dir / "config.json").write_text(json.dumps({"args": vars(args), **metadata}, indent=2, default=str) + "\n")
    logger = CsvLogger(run_dir / "train_log.csv")
    eval_logger = CsvLogger(run_dir / "eval_log.csv")
    target_transitions = replay.total_added + args.total_transitions
    print(
        f"[uipc-manip] discount={agent.cfg.discount:.6f} alpha_lr={agent.cfg.alpha_lr:.2e} reward_scale={reward_scale:.3f} "
        f"additional_replay_transitions={args.total_transitions}",
        flush=True,
    )
    active_garments = -1
    if curriculum_interval > 0:
        print(f"[uipc-manip] garment curriculum every {curriculum_interval} vector steps, order={order}", flush=True)

    obs = env.reset(seeds)
    priv = env.privileged() if privileged else None
    episode_return = np.zeros(env.num_envs)
    updates_started = agent.updates > 0
    best_score = None
    recent_returns: list[float] = []
    recent_success: list[float] = []
    recent_heldout_success: list[float] = []
    t_start = time.time()
    # Wall-clock seconds spent in each phase of the loop, cumulative over the run; evaluation is excluded.
    phase_s = {"act_s": 0.0, "env_s": 0.0, "update_s": 0.0}
    stats: dict = {}
    vector_step = start_step
    while replay.total_added < target_transitions:
        vector_step += 1
        t_phase = time.time()
        if vector_step <= args.init_steps:
            actions = np.random.uniform(-1.0, 1.0, size=(env.num_envs, env.action_dim)).astype(np.float32)
        else:
            actions = agent.act(obs, deterministic=False).astype(np.float32)
        phase_s["act_s"] += time.time() - t_phase
        t_phase = time.time()
        next_obs, rewards, dones, infos = env.step(actions)
        phase_s["env_s"] += time.time() - t_phase
        next_priv = env.privileged() if privileged else None
        if any(info.get("sim_error") for info in infos):
            env.close()
            raise RuntimeError("Simulator failed during training; invalid transitions were excluded. "
                               + str(next(info.get("error", "") for info in infos if info.get("sim_error"))))
        stage = garment_curriculum_stage(vector_step, interval=curriculum_interval, garment_count=max(1, len(order)))
        if stage != active_garments:
            active_garments = stage
            if curriculum_interval > 0:
                print(f"[uipc-manip] curriculum step={vector_step} active={stage}/{len(order)} garments={order[:stage]}", flush=True)
        admitted = training_rows(training_slot_mask, slot_rank, stage, [bool(info.get("sim_error")) for info in infos])
        added = 0
        for i, info in enumerate(infos):
            if info.get("sim_error"):
                episode_return[i] = 0.0
                continue
            if not training_slot_mask[i] and dones[i]:
                recent_heldout_success.append(float(info.get("success", False)))
            if not admitted[i]:
                continue
            # Time limits are not terminal states: bootstrap from the true final observation.
            terminal_obs = info.get("terminal_obs", None)
            state_pair = {}
            if privileged:
                # The terminal state pairs with the terminal observation, not with the reset one.
                terminal_priv = info.get("terminal_privileged", None)
                state_pair = {"priv": priv[i], "next_priv": next_priv[i] if terminal_priv is None else terminal_priv}
            if replay.labelled:
                state_pair["label"] = slot_region[i]
            replay.add(obs[i], actions[i], float(rewards[i]) * reward_scale, next_obs[i] if terminal_obs is None else terminal_obs, False, **state_pair)
            added += 1
            episode_return[i] += float(rewards[i])
            if dones[i]:
                recent_returns.append(float(episode_return[i]))
                recent_success.append(float(info.get("success", False)))
                episode_return[i] = 0.0
        obs = next_obs
        priv = next_priv
        budget, updates_started = gradient_update_budget(
            transitions_added=added,
            replay_size=replay.size,
            batch_size=args.batch_size,
            updates_started=updates_started,
            updates_per_step=args.updates_per_step,
        )
        t_phase = time.time()
        if vector_step > args.init_steps:
            for _ in range(budget):
                # The actor and temperature report only every ``actor_update_freq``
                # updates, and the budget is a multiple of it, so keeping only the
                # last update's dict drops those columns for the whole run.
                stats = {**stats, **agent.update(replay)}
        phase_s["update_s"] += time.time() - t_phase
        if vector_step % args.log_interval == 0:
            elapsed = time.time() - t_start
            row = {
                "step": vector_step,
                "transitions": replay.total_added,
                "simulated_transitions": vector_step * env.num_envs,
                "simulated_steps": vector_step * env.num_envs * args.action_repeat,
                "updates": agent.updates,
                "elapsed_s": round(elapsed, 1),
                **{k: round(v, 1) for k, v in phase_s.items()},
                "episode_return": float(np.mean(recent_returns[-20:])) if recent_returns else float("nan"),
                "episode_success": float(np.mean(recent_success[-20:])) if recent_success else float("nan"),
                **({"heldout_episode_success": float(np.mean(recent_heldout_success[-20:])) if recent_heldout_success else float("nan")} if heldout_slots else {}),
                "active_garments": int(active_garments),
                "alpha": round(float(agent.alpha), 6) if hasattr(agent, "alpha") else float("nan"),
                **{k: round(v, 5) for k, v in stats.items()},
            }
            logger.log(row)
            print("[uipc-manip] " + " ".join(f"{k}={v}" for k, v in row.items()), flush=True)
        do_eval = args.eval_freq > 0 and vector_step % args.eval_freq == 0
        do_ckpt = args.checkpoint_interval > 0 and vector_step % args.checkpoint_interval == 0
        finished_budget = replay.total_added >= target_transitions
        if do_eval or do_ckpt or finished_budget:
            ckpt_dir = run_dir / "checkpoints"
            path = agent.save(ckpt_dir / f"checkpoint_{vector_step:07d}.pt", vector_step, metadata)
            replay.save(ckpt_dir / f"replay_{vector_step:07d}", metadata={"step": vector_step, "reward_scale": reward_scale})
            if do_eval or finished_budget:
                agent.train(False)
                metrics = evaluate(env, policy, spec, args, args.num_eval_episodes, trajectory_dir, slot_cells=slot_cells, heldout_slots=heldout_slots)
                agent.train(True)
                summary = {k: v for k, v in metrics.items() if k != "records"}
                eval_logger.log({"step": vector_step, **summary})
                print(f"[uipc-manip] eval step={vector_step} " + json.dumps(summary), flush=True)
                score = checkpoint_score(metrics)
                if best_score is None or score > best_score:
                    best_score = score
                    agent.save(ckpt_dir / "best.pt", vector_step, {**metadata, "eval": summary})
                obs = env.reset(seeds)
                priv = env.privileged() if privileged else None
                episode_return[:] = 0.0
            print(f"[uipc-manip] saved {path}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
