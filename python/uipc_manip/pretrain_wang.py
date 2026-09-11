"""Wang RSS 2023's pretraining protocol on live dressing cells: regional teachers, then one student.

Usage (from the repository root, inside the Genesis environment)::

    PYTHONPATH=python python -m uipc_manip.pretrain_wang teacher --region 13
    PYTHONPATH=python python -m uipc_manip.pretrain_wang student --regions 4 13 22 \
        --teacher-checkpoints <r4>/checkpoints/best.pt <r13>/checkpoints/best.pt <r22>/checkpoints/best.pt
    PYTHONPATH=python python -m uipc_manip.pretrain_wang resume output/uipc_manip/<run> [--transitions N]

Any other ``train_sac`` flag (``--critic-input``, ``--encoder-precision``, ``--hidden-dim``,
``--updates-per-step`` ...) is passed through to the trainer settings; the flags this
protocol decides are refused there.

What follows the runnable reference, ``curl/train.py`` as ``launch_train_curl.py`` drives it:

* Distribution. 27 arm-pose regions (``dressing_body.region_intervals``) of 50 poses each;
  poses 0 to 44 train and 45 to 49 are held out (``launch_train_curl.py:429-436``), crossed
  with the five garments (``:270``). Body ``1000 (r + 1) + k`` is pose ``k`` of region ``r``.
* Per-episode resampling. Every episode draws its region, garment and pose uniformly
  (``train.py:544-563``; a student first visits each region once, ``:544-545``). A world
  here holds ``--num-envs`` episodes at once and a slot's cell
  is fixed for the life of the world, so every ``--rotate-every`` episodes the world is torn
  down and rebuilt on a fresh draw of training configurations (:class:`ConfigPool`).
* Replay. 400,000 transitions (``launch_train_curl.py:492``). A teacher keeps one buffer
  (``replay_buffer_num = 1``, ``:302``); the student keeps one per region of
  ``capacity // regions`` (``train.py:435``) and draws each update's batch from one buffer
  chosen uniformly (``train.py:595-600``). ``--replay-split`` selects none, garment or region.
* Temperatures. ``SAC_AWAC.py:633`` sizes ``log_alpha`` by the buffer count and indexes it with
  the batch's buffer (``:941``, ``:1006``, ``:1055``), but ``train.py:600`` hands the student's
  update a one-buffer list, so that index is always 0 and the student trains one shared
  temperature; only the PCGrad baseline path (``train.py:605``) uses one per buffer.
  ``--temperatures shared`` is therefore the default and ``per-buffer`` the alternative.
* Evaluation. Every 10,000 transitions (``launch_train_curl.py:440``), at the first episode
  boundary after the threshold, and once before training (``train.py:490-493``): one
  deterministic episode per held-out configuration, region x garment x pose 45-49
  (``train.py:195-206``), in worlds kept for the whole run. ``best.pt`` ranks the mean final
  upper-arm ratio over those unseen configurations first (``train.py:499-500``), which is
  ``cellplan.checkpoint_score``'s leading held-out key.
* No expert pre-filter: every placeable configuration of the region trains.

This port's deviations, all stated in the protocol record:

* 300 decisions of six 1/60 s steps against the reference's 150 FleX steps, with the
  horizon-equivalent discount, temperature learning rate and reward scale (``sac.py``).
* A rotation stratifies its slots over the admitted (region, garment) groups and takes
  poses without replacement inside a group; the marginal distribution is Wang's i.i.d.
  draw, and every replay buffer receives transitions from the first world on.
* A (garment, body) that cannot be placed clear of the arm (``NoClearPlacement``) is
  dropped from its pool and recorded. That is a spawn-legality limit of this port's
  placement, not a filter on whether the scripted expert can dress the cell.
* One gradient update per admitted transition after each vector step, ``train_sac``'s
  schedule, where the reference updates once before every environment step.
* Wang's older ``train_multi_garments.py`` (one buffer per garment, a garment curriculum)
  cannot run against the checked-in agent: it hands ``SAC_AWAC.update`` a bare buffer,
  whose ``len`` is undefined. ``--replay-split garment`` and ``--garment-curriculum-interval``
  reproduce it when wanted; the curriculum then limits which garments a rotation draws.
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from .cellplan import checkpoint_score, summarize
from .curriculum import WANG_GARMENT_ORDER, curriculum_order, garment_curriculum_stage
from .dressing_body import REGION_BODY_BASE, pose_region
from .dressing_env import GenesisIPCDressingEnv
from .dressing_live import LiveCellFactory, NoClearPlacement, available_garments
from .genesis_env import _ensure_genesis
from .obs import ObsSpec

REGION_COUNT = 27
POSES_PER_REGION = 50
TRAIN_POSES = tuple(range(45))
EVAL_POSES = tuple(range(45, 50))
WANG_REPLAY_CAPACITY = 400_000
WANG_BATCH_SIZE = 64
WANG_EVAL_TRANSITIONS = 10_000
HORIZON = 300
DECISION_SECONDS = 0.1
REFERENCE_DT = 1.0 / 60.0
REFERENCE_CUFF_STRENGTH = 1.0e4
REFERENCE_SETTLE_STEPS = 30
"""The decision period, held-cuff strength and settle of the 1/60 s configuration every
measured ceiling uses; :func:`matched_physics` carries them to another time step."""

# train_sac flags this protocol sets itself; passing one through would silently change the protocol.
RESERVED_TRAINER_FLAGS = frozenset({
    "task", "cell_source", "human", "body_seeds", "heldout_bodies", "heldout_body_seeds", "allow_partial_cell_coverage",
    "garments", "garment_curriculum_interval", "garment_curriculum_order", "num_envs", "horizon", "dt", "total_transitions",
    "replay_capacity", "batch_size", "eval_freq", "num_eval_episodes", "checkpoint_interval", "log_interval", "policy",
    "eval_only", "resume", "resume_replay", "vis", "seed", "work_dir", "run_name", "device", "teacher_checkpoints",
    "distill_weight", "save_trajectories",
})


# ------------------------------------------------------------------ distribution
def region_body(region: int, pose: int) -> int:
    """The body id whose arm takes pose ``pose`` of Wang's region ``region``."""
    if not 0 <= int(region) < REGION_COUNT:
        raise ValueError(f"Region {region} is outside 0-{REGION_COUNT - 1}")
    if not 0 <= int(pose) < POSES_PER_REGION:
        raise ValueError(f"Pose {pose} is outside 0-{POSES_PER_REGION - 1}")
    return REGION_BODY_BASE * (int(region) + 1) + int(pose)


def region_configs(regions, garments, poses) -> list[tuple[str, int]]:
    """Every (garment, body) configuration of these regions and poses, region-major."""
    return [(str(g), region_body(r, p)) for r in regions for g in garments for p in poses]


class ConfigPool:
    """The training configurations a rotation draws from, grouped by (region, garment).

    Wang draws region, garment and pose independently and uniformly for every episode. A
    world holds ``num_envs`` episodes at once, so :meth:`draw` deals the admitted groups
    over the slots in a random order and takes poses without replacement inside a group:
    the long-run frequency of every configuration is Wang's, and every group, hence every
    replay buffer, receives slots in every world.
    """

    def __init__(self, regions, garments, poses=TRAIN_POSES) -> None:
        self.groups = {(int(r), str(g)): [region_body(r, p) for p in poses] for r in regions for g in garments}
        self.dropped: list[dict] = []

    def drop(self, garment: str, body: int, reason: str) -> None:
        """Remove a configuration that cannot be built, and record why."""
        key = (pose_region(body), str(garment))
        if int(body) in self.groups.get(key, []):
            self.groups[key].remove(int(body))
            self.dropped.append({"garment": str(garment), "body": int(body), "reason": str(reason)})
        if not self.groups.get(key):
            raise RuntimeError(f"No pose of region {key[0]} can be placed for {garment}; dropped {self.dropped}")

    def configs(self) -> list[tuple[str, int]]:
        return [(g, b) for (_, g), bodies in sorted(self.groups.items()) for b in bodies]

    def draw(self, num_envs: int, rng: np.random.Generator, garments=None) -> list[tuple[str, int]]:
        """One world's cells: admitted groups dealt over the slots, poses drawn without replacement per group."""
        groups = [key for key in sorted(self.groups) if garments is None or key[1] in set(garments)]
        if not groups:
            raise ValueError(f"No configuration group admits garments {garments}")
        order = [groups[i] for i in rng.permutation(len(groups))]
        bags: dict[tuple[int, str], list[int]] = {}
        cells = []
        for slot in range(int(num_envs)):
            key = order[slot % len(order)]
            if not bags.get(key):
                bags[key] = [int(b) for b in rng.permutation(self.groups[key])]
            cells.append((key[1], bags[key].pop()))
        return cells


def buffer_key(split: str, garment: str, body: int):
    """The replay buffer a slot's transitions go to under ``--replay-split``."""
    if split == "garment":
        return str(garment)
    if split == "region":
        return pose_region(body)
    return 0


def matched_physics(dt: float) -> dict:
    """Action repeat, cuff strength and settle that keep the reference decision, hold and settle at ``dt``.

    A decision stays 0.1 s, the held cuff's physical stiffness ``strength * mass / dt**2``
    stays that of strength 1e4 at 1/60 s, and the settle keeps its duration. At 1/60 s these
    are the trainer's own values (6, 1e4, 30); at 1/30 s they are 3, 4e4 and 15.
    """
    dt = float(dt)
    repeat = DECISION_SECONDS / dt
    if dt <= 0.0 or abs(repeat - round(repeat)) > 1.0e-6:
        raise ValueError(f"dt {dt} does not divide the {DECISION_SECONDS} s decision")
    ratio = dt / REFERENCE_DT
    return {
        "action_repeat": int(round(repeat)),
        "cuff_strength": REFERENCE_CUFF_STRENGTH * ratio**2,
        "settle_steps": max(1, int(round(REFERENCE_SETTLE_STEPS / ratio))),
    }


# ---------------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m uipc_manip.pretrain_wang", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    stages = p.add_subparsers(dest="stage", required=True)
    for stage in ("teacher", "student"):
        s = stages.add_parser(stage, allow_abbrev=False, help=f"train a Wang {stage}; unknown flags go to train_sac")
        if stage == "teacher":
            s.add_argument("--region", type=int, required=True, help="The arm-pose region (0-26) this teacher trains on.")
        else:
            s.add_argument("--regions", type=int, nargs="+", required=True, help="Arm-pose regions the student trains on, each with a teacher.")
            s.add_argument("--teacher-checkpoints", nargs="+", required=True, help="One regional teacher checkpoint per region.")
            s.add_argument("--distill-weight", type=float, default=0.01, help="Weight of Wang's teacher loss; the paper states 0.01, the reference launcher 0.002.")
        s.add_argument("--garments", nargs="+", default=list(WANG_GARMENT_ORDER), help="Garments of the distribution; Wang's five by default.")
        s.add_argument("--train-poses", type=int, nargs="+", default=list(TRAIN_POSES), help="Poses per region that train; Wang's 0-44.")
        s.add_argument("--eval-poses", type=int, nargs="+", default=list(EVAL_POSES), help="Held-out poses per region; Wang's 45-49.")
        s.add_argument("--num-envs", type=int, default=24, help="Episodes a training world holds at once.")
        s.add_argument("--transitions", type=int, default=600_000, help="Total replay transitions of the run, counted across a resume (the reference launcher runs 5,000,000).")
        s.add_argument("--replay-capacity", type=int, default=WANG_REPLAY_CAPACITY, help="Total replay capacity, split evenly over the buffers.")
        s.add_argument("--batch-size", type=int, default=WANG_BATCH_SIZE)
        s.add_argument("--replay-split", choices=("none", "garment", "region"), default="none" if stage == "teacher" else "region",
                       help="One buffer, one per garment, or one per region; each update draws its batch from one buffer.")
        s.add_argument("--temperatures", choices=("shared", "per-buffer"), default="shared",
                       help="One entropy temperature, as the reference's runnable paths train, or one per replay buffer.")
        s.add_argument("--rotate-every", type=int, default=1, help="Episodes each world plays before it is rebuilt on a fresh draw; Wang redraws every episode.")
        s.add_argument("--eval-every", type=int, default=WANG_EVAL_TRANSITIONS, help="Transitions between evaluations; 0 evaluates only before and after training.")
        s.add_argument("--eval-slots", type=int, default=32, help="Largest evaluation world; more held-out configurations are split over several worlds.")
        s.add_argument("--checkpoint-every", type=int, default=50_000, help="Transitions between checkpoints with a replay snapshot; only the latest snapshot is kept.")
        s.add_argument("--garment-curriculum-interval", type=int, default=0, help="Transitions between admitting one more garment to the draw, easiest first; 0, the reference launcher's setting, admits all.")
        s.add_argument("--garment-curriculum-order", default=",".join(WANG_GARMENT_ORDER), help="Comma-separated garments, easiest first.")
        s.add_argument("--horizon", type=int, default=HORIZON, help="Decisions per episode.")
        s.add_argument("--dt", type=float, default=REFERENCE_DT, help="Simulation step [s]; the action repeat, cuff strength and settle follow it (matched_physics) unless passed explicitly.")
        s.add_argument("--seed", type=int, default=1)
        s.add_argument("--device", default="cuda:0")
        s.add_argument("--work-dir", default="output/uipc_manip")
        s.add_argument("--run-name", default=None)
        s.add_argument("--log-interval", type=int, default=20, help="Vector steps between log rows.")
        s.add_argument("--save-trajectories", action="store_true", help="Store every evaluation episode as an .npz file.")
    r = stages.add_parser("resume", allow_abbrev=False, help="continue a run from its latest checkpoint")
    r.add_argument("run_dir", help="The run directory holding checkpoints/state.json.")
    r.add_argument("--transitions", type=int, default=None, help="New total budget; default the saved one.")
    return p


def reserved_flags(extra: list[str]) -> list[str]:
    """Pass-through flags that the protocol sets itself."""
    names = [token[2:].split("=", 1)[0].replace("-", "_") for token in extra if token.startswith("--")]
    return sorted({name for name in names if name in RESERVED_TRAINER_FLAGS})


def run_name(args) -> str:
    if args.run_name:
        return args.run_name
    if args.stage == "teacher":
        return f"wang_teacher_r{args.region}_s{args.seed}"
    return f"wang_student_r{'-'.join(str(r) for r in sorted(set(args.regions)))}_s{args.seed}"


def trainer_argv(args, extra: list[str]) -> list[str]:
    """The ``train_sac`` flags of this stage: the trainer settings every shared helper reads.

    The action repeat, cuff strength and, away from 1/60 s, the settle come from
    :func:`matched_physics`; a pass-through flag given explicitly comes later and wins.
    """
    physics = matched_physics(args.dt)
    argv = [
        "--task", "dressing", "--cell-source", "live", "--garments", *args.garments,
        "--num-envs", str(args.num_envs), "--horizon", str(args.horizon), "--dt", repr(float(args.dt)),
        "--action-repeat", str(physics["action_repeat"]), "--cuff-strength", repr(physics["cuff_strength"]),
        "--replay-capacity", str(args.replay_capacity), "--batch-size", str(args.batch_size), "--updates-per-step", "0",
        "--seed", str(args.seed), "--device", str(args.device), "--work-dir", str(args.work_dir), "--run-name", run_name(args),
    ]
    if abs(float(args.dt) - REFERENCE_DT) > 1.0e-12:
        argv += ["--settle-steps", str(physics["settle_steps"])]
    if args.stage == "student":
        argv += ["--teacher-checkpoints", *args.teacher_checkpoints, "--distill-weight", repr(float(args.distill_weight))]
    return argv + list(extra)


def stage_plan(args) -> dict:
    """Everything that defines the run's protocol; a resume must reproduce it exactly."""
    regions = [int(args.region)] if args.stage == "teacher" else sorted({int(r) for r in args.regions})
    garments = list(dict.fromkeys(str(g) for g in args.garments))
    train_poses, eval_poses = sorted({int(p) for p in args.train_poses}), sorted({int(p) for p in args.eval_poses})
    if set(train_poses) & set(eval_poses):
        raise ValueError(f"Poses {sorted(set(train_poses) & set(eval_poses))} are both trained on and held out")
    split = str(args.replay_split)
    keys = {"none": [0], "garment": garments, "region": regions}[split]
    if args.temperatures == "per-buffer" and split == "none":
        raise ValueError("--temperatures per-buffer needs --replay-split garment or region")
    order = curriculum_order(str(args.garment_curriculum_order).split(","), garments)
    return {
        "protocol": "wang_rss2023",
        "stage": args.stage,
        "regions": regions,
        "garments": garments,
        "train_poses": train_poses,
        "eval_poses": eval_poses,
        "eval_configs": [list(c) for c in region_configs(regions, garments, eval_poses)],
        "replay_split": split,
        "buffer_keys": keys,
        "temperatures": str(args.temperatures),
        "temperature_count": len(keys) if args.temperatures == "per-buffer" else 1,
        "num_envs": int(args.num_envs),
        "rotate_every_episodes": int(args.rotate_every),
        "eval_every_transitions": int(args.eval_every),
        "eval_slots": int(args.eval_slots),
        "checkpoint_every_transitions": int(args.checkpoint_every),
        "garment_curriculum_interval": int(args.garment_curriculum_interval),
        "curriculum_order": order,
        "horizon": int(args.horizon),
        "dt": float(args.dt),
    }


def eval_summary(records: list[dict], cells: list[tuple[str, int]]) -> dict:
    """Wang's evaluation round over the unseen configurations: every slot is held out.

    ``cellplan.summarize`` gives the per-cell, per-garment and ``heldout_`` keys that
    ``checkpoint_score`` ranks; per-region means follow ``train.py``'s ``region_*`` logs.
    """
    out = summarize(records, cells, range(len(cells)))
    out["mean_max_upperarm_ratio"] = float(np.nanmean([r.get("max_upperarm_ratio", np.nan) for r in records]))
    out["sim_errors"] = int(sum(bool(r.get("sim_error")) for r in records))
    for region in sorted({pose_region(b) for _, b in cells}):
        rows = [r for r in records if pose_region(cells[int(r["slot"])][1]) == region]
        out[f"mean_final_upperarm_ratio_region_{region}"] = float(np.nanmean([r["final_upperarm_ratio"] for r in rows]))
        out[f"success_rate_region_{region}"] = float(np.mean([r["success"] for r in rows]))
    return out


# ------------------------------------------------------------------------ the run
class WangRun:
    """One teacher or student run: rotating training worlds, fixed evaluation worlds, one agent."""

    def __init__(self, args, argv: list[str], targs, plan: dict, resume: dict | None = None) -> None:
        from . import train_sac

        self.args, self.argv, self.targs, self.plan, self.resume = args, list(argv), targs, plan, resume
        self.train_sac = train_sac
        self.run_dir = Path(targs.work_dir) / targs.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.base_cfg = train_sac.dressing_config(targs)
        # Genesis comes up before anything touches CUDA: the cell factory generates bodies on the GPU.
        _ensure_genesis(self.base_cfg.logging_level)
        missing = sorted(set(plan["garments"]) - set(available_garments(self.base_cfg.live)))
        if missing:
            raise ValueError(f"Garments {missing} have no raw mesh, pull schedule or index tables")
        self.factory = LiveCellFactory(self.base_cfg.live)
        self.pool = ConfigPool(plan["regions"], plan["garments"], plan["train_poses"])
        self.rng = np.random.default_rng(int(args.seed))
        self.env = None
        self.eval_worlds: list[tuple[object, list[tuple[str, int]]]] = []
        self.eval_dropped: list[dict] = []
        self.counters = {"vector_step": 0, "episodes": 0, "rotations": 0, "sim_errors": 0, "build_failures": 0}
        self.timing = {"act_s": 0.0, "env_s": 0.0, "update_s": 0.0, "rebuild_s": 0.0, "eval_s": 0.0, "checkpoint_s": 0.0}
        self.next_eval = 0
        self.next_checkpoint = int(plan["checkpoint_every_transitions"])
        self.best_score = None
        self.updates_started = False
        if resume is not None:
            self.rng.bit_generator.state = resume["rng"]
            state = resume["np_random"]
            np.random.set_state((state[0], np.asarray(state[1], dtype=np.uint32), *state[2:]))
            for entry in resume["dropped"]:
                self.pool.drop(entry["garment"], entry["body"], entry["reason"])
            self.counters.update(resume["counters"])
            self.timing.update(resume["timing"])
            self.next_eval, self.next_checkpoint = int(resume["next_eval"]), int(resume["next_checkpoint"])
            self.best_score = tuple(resume["best_score"]) if resume["best_score"] is not None else None
            self.updates_started = bool(resume["updates_started"])
            targs._eval_round = int(resume["eval_round"])

    # -------------------------------------------------------------- worlds
    def admitted_garments(self, transitions: int) -> list[str]:
        order = self.plan["curriculum_order"]
        stage = garment_curriculum_stage(transitions, interval=self.plan["garment_curriculum_interval"], garment_count=len(order))
        return order[:stage]

    def placeable(self, garment: str, body: int) -> str | None:
        """Why this configuration cannot be placed, or None when it can."""
        try:
            self.factory.clearance_for(garment, body)
        except NoClearPlacement as exc:
            return str(exc)
        return None

    def draw_cells(self, transitions: int) -> list[tuple[str, int]]:
        while True:
            cells = self.pool.draw(self.plan["num_envs"], self.rng, self.admitted_garments(transitions))
            refused = {(g, b): self.placeable(g, b) for g, b in dict.fromkeys(cells)}
            refused = {cell: reason for cell, reason in refused.items() if reason}
            if not refused:
                return cells
            for (g, b), reason in refused.items():
                print(f"[wang] dropped {g} on body {b} from the training pool: {reason}", flush=True)
                self.pool.drop(g, b, reason)

    def build_world(self, cells) -> tuple[object, float]:
        t0 = time.time()
        env = GenesisIPCDressingEnv(replace(self.base_cfg, cells=tuple(cells)), num_envs=len(cells), cell_factory=self.factory)
        return env, time.time() - t0

    def close_world(self, env) -> None:
        env.close()
        # The libuipc world is freed only once nothing references it; collect before the next build.
        gc.collect()

    def rotate(self, transitions: int):
        """Tear down the training world and build the next one on a fresh draw; return its first observation."""
        t0 = time.time()
        if self.env is not None:
            self.close_world(self.env)
            self.env = None
        n = int(self.plan["num_envs"])
        seeds = [int(self.args.seed) * 1_000_000 + self.counters["rotations"] * n + i for i in range(n)]
        for _ in range(3):
            cells, env = self.draw_cells(transitions), None
            try:
                env, build_s = self.build_world(cells)
                obs = env.reset(seeds)
                break
            except RuntimeError as exc:
                # A world libuipc refuses or cannot settle is drawn again rather than ending the run.
                self.counters["build_failures"] = self.counters.get("build_failures", 0) + 1
                print(f"[wang] a world of {cells} failed to build or reset and is drawn again: {str(exc)[:300]}", flush=True)
                if env is not None:
                    self.close_world(env)
                gc.collect()
        else:
            raise RuntimeError("Three worlds in a row failed to build; see the messages above")
        self.env = env
        self.slot_cells = cells
        self.slot_keys = [buffer_key(self.plan["replay_split"], g, b) for g, b in cells]
        self.slot_regions = [pose_region(b) for _, b in cells]
        self.counters["rotations"] += 1
        spent = time.time() - t0
        self.timing["rebuild_s"] += spent
        print(f"[wang] world {self.counters['rotations']}: {n} cells built in {build_s:.1f}s (scene build "
              f"{getattr(env, 'build_seconds', float('nan')):.1f}s, the rest cells and settle; {spent:.1f}s with placement, "
              f"teardown and reset) garments={sorted({g for g, _ in cells})} regions={sorted(set(self.slot_regions))}", flush=True)
        return obs

    def build_eval_worlds(self) -> None:
        cells = []
        for g, b in (tuple(c) for c in self.plan["eval_configs"]):
            reason = self.placeable(g, b)
            if reason:
                print(f"[wang] held-out {g} on body {b} cannot be placed and is not evaluated: {reason}", flush=True)
                self.eval_dropped.append({"garment": g, "body": int(b), "reason": reason})
            else:
                cells.append((g, int(b)))
        if not cells:
            raise RuntimeError("No held-out configuration can be placed")
        size = max(1, int(self.plan["eval_slots"]))
        for start in range(0, len(cells), size):
            chunk = cells[start:start + size]
            env, build_s = self.build_world(chunk)
            self.eval_worlds.append((env, chunk))
            print(f"[wang] evaluation world {len(self.eval_worlds)}: {len(chunk)} held-out configurations built in {build_s:.1f}s", flush=True)

    # -------------------------------------------------------------- learning
    def make_agent(self, env):
        from . import sac
        from .replay import FlatReplayBuffer, ReplaySet
        from .sac import wang_equivalent_reward_scale

        targs, plan = self.targs, self.plan
        import torch

        torch.manual_seed(int(self.args.seed))
        sac_cfg = self.train_sac.build_sac_config(targs)
        sac_cfg.temperature_count = int(plan["temperature_count"])
        self.privileged = sac_cfg.critic_input == "privileged"
        if self.privileged:
            sac_cfg.privileged_dim = int(env.privileged_dim)
        self.spec = ObsSpec(targs.point_budget)
        self.agent = sac.SACAgent(self.spec, env.action_dim, sac_cfg, targs.device)
        self.labelled = plan["stage"] == "student"
        priv_dim = sac_cfg.privileged_dim if self.privileged else 0
        if plan["replay_split"] == "none":
            self.replay = FlatReplayBuffer(env.obs_dim, env.action_dim, targs.replay_capacity, targs.batch_size, targs.device,
                                           priv_dim=priv_dim, labelled=self.labelled)
        else:
            self.replay = ReplaySet(plan["buffer_keys"], env.obs_dim, env.action_dim, targs.replay_capacity, targs.batch_size,
                                    targs.device, priv_dim=priv_dim, labelled=self.labelled)
        self.reward_scale = wang_equivalent_reward_scale(sac_cfg.discount)
        self.teacher_regions = sorted(self.teachers)
        if self.teachers:
            self.agent.set_teachers(self.teachers)
        if self.resume is not None:
            payload = self.agent.load(self.resume["checkpoint"], load_optimizers=True)
            self.reward_scale = float(payload.get("metadata", {}).get("reward_scale", self.reward_scale))
            meta = self.replay.load(self.resume["replay"])
            if int(meta.get("transitions", -1)) != int(self.resume["transitions"]):
                raise ValueError("The replay snapshot does not come from the checkpoint it is resumed with")

    def policy(self, obs, deterministic):
        return self.agent.act(obs, deterministic)

    def add(self, i: int, obs, action, reward, next_obs, **kw) -> None:
        if self.labelled:
            kw["label"] = self.slot_regions[i]
        if self.plan["replay_split"] == "none":
            self.replay.add(obs, action, reward, next_obs, False, **kw)
        else:
            self.replay.add(self.slot_keys[i], obs, action, reward, next_obs, False, **kw)

    def replay_sizes(self) -> dict:
        if self.plan["replay_split"] == "none":
            return {"buffer_0": self.replay.size}
        return {f"buffer_{k}": s for k, s in zip(self.plan["buffer_keys"], self.replay.sizes(), strict=True)}

    def metadata(self) -> dict:
        """Checkpoint metadata. ``cells`` is the training pool with no held-out slot, so
        ``train_sac.load_teachers`` reads a teacher's region from it as from any run."""
        return {
            "task": "dressing",
            "protocol": "wang_rss2023",
            "stage": self.plan["stage"],
            "env": self.env.descriptions[0]["config"] if self.env is not None else self.base_cfg.to_dict(),
            "sac_config": self.agent.cfg.to_dict(),
            "reward_scale": self.reward_scale,
            "seed": int(self.args.seed),
            "num_envs": int(self.plan["num_envs"]),
            "cells": [[g, int(b)] for g, b in self.pool.configs()],
            "heldout_slots": [],
            "eval_cells": [[g, int(b)] for _, cells in self.eval_worlds for g, b in cells],
            "slot_cells": [[g, int(b)] for g, b in getattr(self, "slot_cells", [])],
            "regions": self.plan["regions"],
            "garments": self.plan["garments"],
            "teacher_regions": self.teacher_regions,
            "dropped_configs": self.pool.dropped,
            "dropped_eval_configs": self.eval_dropped,
            "transitions": int(self.replay.total_added),
            "plan": self.plan,
        }

    # -------------------------------------------------------------- evaluation and checkpoints
    def on_schedule(self, transitions: int) -> None:
        """Evaluate and checkpoint when the transition count has passed the next mark of either."""
        plan = self.plan
        if plan["eval_every_transitions"] > 0 and transitions >= self.next_eval:
            self.evaluate()
            while self.next_eval <= transitions:
                self.next_eval += int(plan["eval_every_transitions"])
        if plan["checkpoint_every_transitions"] > 0 and transitions >= self.next_checkpoint:
            self.checkpoint()
            while self.next_checkpoint <= transitions:
                self.next_checkpoint += int(plan["checkpoint_every_transitions"])

    def evaluate(self) -> dict:
        t0 = time.time()
        transitions = int(self.replay.total_added)
        self.agent.train(False)
        records, offset = [], 0
        for chunk, (env, cells) in enumerate(self.eval_worlds):
            trajectories = self.run_dir / "trajectories" / f"eval_{transitions:08d}_{chunk}" if self.args.save_trajectories else None
            result = self.train_sac.evaluate(env, self.policy, self.spec, self.targs, env.num_envs, trajectories,
                                             slot_cells=cells, heldout_slots=range(env.num_envs))
            records += [{**r, "slot": int(r["slot"]) + offset} for r in result["records"]]
            offset += env.num_envs
        self.agent.train(True)
        cells = [cell for _, chunk in self.eval_worlds for cell in chunk]
        summary = eval_summary(records, cells)
        spent = time.time() - t0
        self.timing["eval_s"] += spent
        self.eval_logger.log({"step": self.counters["vector_step"], "transitions": transitions, "eval_s": round(spent, 1), **summary})
        print(f"[wang] eval transitions={transitions} unseen upper-arm ratio {summary['heldout_mean_final_upperarm_ratio']:.3f} "
              f"success {summary['heldout_success_rate']:.3f} worst cell {summary['heldout_worst_cell_success_rate']:.3f} "
              f"({len(records)} episodes, {spent:.0f}s)", flush=True)
        score = checkpoint_score(summary)
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.agent.save(self.run_dir / "checkpoints" / "best.pt", self.counters["vector_step"], {**self.metadata(), "eval": summary})
        return summary

    def checkpoint(self) -> None:
        """Agent, replay snapshot and loop state; the state file is written last and names the other two."""
        t0 = time.time()
        transitions = int(self.replay.total_added)
        ckpt_dir = self.run_dir / "checkpoints"
        path = self.agent.save(ckpt_dir / f"checkpoint_{transitions:08d}.pt", self.counters["vector_step"], self.metadata())
        staging, latest = ckpt_dir / "replay_staging", ckpt_dir / "replay_latest"
        shutil.rmtree(staging, ignore_errors=True)
        self.replay.save(staging, metadata={"transitions": transitions, "reward_scale": self.reward_scale})
        shutil.rmtree(latest, ignore_errors=True)
        staging.rename(latest)
        state = np.random.get_state()
        payload = {
            "argv": self.argv,
            "checkpoint": str(path),
            "replay": str(latest),
            "transitions": transitions,
            "plan": self.plan,
            "rng": self.rng.bit_generator.state,
            "np_random": [state[0], state[1].tolist(), *state[2:]],
            "dropped": self.pool.dropped,
            "counters": self.counters,
            "timing": self.timing,
            "next_eval": self.next_eval,
            "next_checkpoint": self.next_checkpoint,
            "best_score": list(self.best_score) if self.best_score is not None else None,
            "updates_started": self.updates_started,
            "eval_round": int(getattr(self.targs, "_eval_round", 0)),
        }
        staging_state = ckpt_dir / "state.json.tmp"
        staging_state.write_text(json.dumps(payload, indent=1, default=float) + "\n")
        staging_state.replace(ckpt_dir / "state.json")
        self.timing["checkpoint_s"] += time.time() - t0
        print(f"[wang] saved {path} and the replay snapshot at {transitions} transitions ({time.time() - t0:.0f}s)", flush=True)

    # -------------------------------------------------------------- the loop
    def train(self) -> None:
        from .sac import gradient_update_budget

        args, targs, plan = self.args, self.targs, self.plan
        start_transitions = int(self.resume["transitions"]) if self.resume is not None else 0
        self.teachers = {}
        if plan["stage"] == "student":
            # Every region needs its teacher; find that out before any world is built.
            self.teachers = self.train_sac.load_teachers(
                targs.teacher_checkpoints, ObsSpec(targs.point_budget), GenesisIPCDressingEnv.action_dim, targs.device
            )
            uncovered = sorted(set(plan["regions"]) - set(self.teachers))
            if uncovered:
                raise ValueError(f"Regions {uncovered} have no teacher among {sorted(self.teachers)}")
        obs = self.rotate(start_transitions)
        self.build_eval_worlds()
        env = self.env
        self.make_agent(env)
        target = int(args.transitions)
        (self.run_dir / "config.json").write_text(json.dumps(
            {"argv": self.argv, "trainer_args": vars(targs), **self.metadata()}, indent=2, default=str) + "\n")
        self.logger = self.train_sac.CsvLogger(self.run_dir / "train_log.csv")
        self.eval_logger = self.train_sac.CsvLogger(self.run_dir / "eval_log.csv")
        print(f"[wang] {plan['stage']} regions={plan['regions']} garments={plan['garments']} pool={len(self.pool.configs())} "
              f"held-out={sum(len(c) for _, c in self.eval_worlds)} replay={type(self.replay).__name__}{plan['buffer_keys']} "
              f"temperatures={plan['temperatures']} discount={self.agent.cfg.discount:.6f} reward_scale={self.reward_scale:.3f} "
              f"budget={target} transitions", flush=True)
        if self.resume is None:
            # The reference evaluates the untrained policy at step 0.
            self.evaluate()
            self.next_eval = int(plan["eval_every_transitions"])
        priv = self.env.privileged() if self.privileged else None
        episode_return = np.zeros(len(self.slot_cells))
        recent_returns: list[float] = []
        recent_success: list[float] = []
        stats: dict = {}
        t_start = time.time()
        n_log = max(1, int(args.log_interval))
        while self.replay.total_added < target:
            env = self.env
            self.counters["vector_step"] += 1
            step = self.counters["vector_step"]
            t = time.time()
            if step <= targs.init_steps:
                actions = np.random.uniform(-1.0, 1.0, size=(env.num_envs, env.action_dim)).astype(np.float32)
            else:
                actions = self.agent.act(obs, deterministic=False).astype(np.float32)
            self.timing["act_s"] += time.time() - t
            t = time.time()
            next_obs, rewards, dones, infos = env.step(actions)
            self.timing["env_s"] += time.time() - t
            if any(info.get("sim_error") for info in infos):
                # The reference ends the episode on a simulator error and draws the next one; so does a rotation.
                # The schedule still runs: the decision watchdog can cut every episode of a run short.
                self.counters["sim_errors"] += 1
                print(f"[wang] simulator error at step {step}; the step's transitions are dropped and the world rebuilt: "
                      + str(next(info.get("error", "") for info in infos if info.get("sim_error")))[:300], flush=True)
                episode_return[:] = 0.0
                transitions = int(self.replay.total_added)
                if transitions >= target:
                    break
                self.on_schedule(transitions)
                obs = self.rotate(transitions)
                priv = self.env.privileged() if self.privileged else None
                continue
            next_priv = env.privileged() if self.privileged else None
            for i, info in enumerate(infos):
                # Time limits are not terminal: bootstrap from the true final observation.
                terminal_obs = info.get("terminal_obs", None)
                kw = {}
                if self.privileged:
                    terminal_priv = info.get("terminal_privileged", None)
                    kw = {"priv": priv[i], "next_priv": next_priv[i] if terminal_priv is None else terminal_priv}
                self.add(i, obs[i], actions[i], float(rewards[i]) * self.reward_scale, next_obs[i] if terminal_obs is None else terminal_obs, **kw)
                episode_return[i] += float(rewards[i])
                if dones[i]:
                    recent_returns.append(float(episode_return[i]))
                    recent_success.append(float(info.get("success", False)))
                    episode_return[i] = 0.0
            obs, priv = next_obs, next_priv
            sizes = self.replay.sizes() if plan["replay_split"] != "none" else [self.replay.size]
            budget, self.updates_started = gradient_update_budget(
                transitions_added=env.num_envs, replay_size=max(sizes), batch_size=targs.batch_size,
                updates_started=self.updates_started, updates_per_step=targs.updates_per_step,
            )
            t = time.time()
            if step > targs.init_steps:
                for _ in range(budget):
                    stats = {**stats, **self.agent.update(self.replay)}
            self.timing["update_s"] += time.time() - t
            transitions = int(self.replay.total_added)
            if step % n_log == 0:
                alpha = self.agent.alpha.detach()
                row = {
                    "step": step,
                    "transitions": transitions,
                    "updates": self.agent.updates,
                    "elapsed_s": round(time.time() - t_start, 1),
                    **{k: round(v, 1) for k, v in self.timing.items()},
                    **{k: v for k, v in self.counters.items() if k != "vector_step"},
                    "episode_return": float(np.mean(recent_returns[-40:])) if recent_returns else float("nan"),
                    "episode_success": float(np.mean(recent_success[-40:])) if recent_success else float("nan"),
                    "active_garments": len(self.admitted_garments(transitions)),
                    "alpha": round(float(alpha.mean()), 6),
                    **self.replay_sizes(),
                    **{k: round(v, 5) for k, v in stats.items()},
                }
                self.logger.log(row)
                print("[wang] " + " ".join(f"{k}={v}" for k, v in row.items()), flush=True)
            if not dones.any():
                continue
            if not dones.all():
                raise RuntimeError("Slots of one world finished their episodes at different steps; the fixed horizon keeps them together")
            self.counters["episodes"] += 1
            if transitions >= target:
                break
            self.on_schedule(transitions)
            if self.counters["episodes"] % max(1, int(plan["rotate_every_episodes"])) == 0:
                obs = self.rotate(transitions)
                priv = self.env.privileged() if self.privileged else None
                episode_return = np.zeros(len(self.slot_cells))
        self.evaluate()
        self.checkpoint()
        self.close()

    def close(self) -> None:
        for env, _ in self.eval_worlds:
            self.close_world(env)
        self.eval_worlds = []
        if self.env is not None:
            self.close_world(self.env)
            self.env = None


def prepare(argv: list[str]):
    """Parse a stage command line into the protocol args, the generated trainer settings and the plan."""
    from . import train_sac

    args, extra = build_parser().parse_known_args(argv)
    if args.stage == "resume":
        raise ValueError("prepare() takes a teacher or student command line")
    reserved = reserved_flags(extra)
    if reserved:
        raise ValueError(f"The protocol sets {['--' + r.replace('_', '-') for r in reserved]} itself; use this CLI's own flags")
    targs = train_sac.build_parser().parse_args(trainer_argv(args, extra))
    train_sac.resolve_defaults(targs)
    return args, targs, stage_plan(args)


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else list(argv)
    parsed, _ = build_parser().parse_known_args(argv)
    resume = None
    if parsed.stage == "resume":
        run_dir = Path(parsed.run_dir)
        resume = json.loads((run_dir / "checkpoints" / "state.json").read_text())
        argv = list(resume["argv"])
        if parsed.transitions is not None:
            argv += ["--transitions", str(parsed.transitions)]
        # The run continues where it was saved, whatever its directory is called now.
        argv += ["--work-dir", str(run_dir.parent), "--run-name", run_dir.name]
    args, targs, plan = prepare(argv)
    if resume is not None:
        saved = json.loads(json.dumps(resume["plan"]))
        if saved != json.loads(json.dumps(plan)):
            changed = sorted(k for k in set(saved) | set(plan) if saved.get(k) != json.loads(json.dumps(plan)).get(k))
            raise ValueError(f"The resumed run's protocol differs from the saved one in {changed}")
    WangRun(args, argv, targs, plan, resume).train()


if __name__ == "__main__":
    main()
