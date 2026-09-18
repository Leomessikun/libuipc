"""Drive the dressing task from a program an agent writes, the way agentic-robotics demos do.

The published agent demonstrations give a general model a small tool surface — look,
compute, command the arm, look again — and let it write and revise the program that
uses them. This is that surface for our dressing environment, so an agent can be run
as a baseline beside a trained policy and the two can be compared on the same task,
seeds, controller and metric.

What the agent may read is what a robot could read: the segmented point cloud in the
tool's frame, the tool's own pose, the direction to the goal, and how far the garment
moved for the motion it commanded. Task metrics live behind :meth:`DressingSession.report`
and are recorded for the evaluator, not returned to the program. A rendered view is
available on request and is a geometric drawing of the simulator state, not a camera
image.

Commands are in metres and radians and are clipped to the controller's own
per-decision limits, so a program cannot move faster than the policy it is compared
with. Every call is logged, so a run can be audited afterwards.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import decision_features as df


class DecisionBudgetExhausted(RuntimeError):
    """Raised when a program asks for more decisions than the episode allows."""


class DressingSession:
    """One episode of the dressing task, driven by an agent's program.

    The program receives this object and calls :meth:`observe`, :meth:`move`,
    :meth:`render` and optionally :meth:`run_policy`. It cannot reach the simulator
    directly.
    """

    def __init__(self, env, *, slot: int = 0, max_decisions: int = 300, agent=None,
                 image_dir: Path | None = None, response_window: int = 8, render_every: int = 0):
        self.env, self.slot, self.max_decisions = env, int(slot), int(max_decisions)
        self._agent, self._image_dir = agent, image_dir
        self._response_window = int(response_window)
        self._render_every = int(render_every)
        self.decisions_used = 0
        self.calls: list[dict] = []
        self.trace: list[dict] = []
        self._obs = None
        self._centroids: list[np.ndarray] = []
        self._commanded: list[float] = []
        if image_dir is not None:
            image_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- lifecycle
    def reset(self, seed: int) -> dict:
        self._obs = self.env.reset([seed + i for i in range(self.env.num_envs)])
        self.decisions_used = 0
        self._centroids.clear()
        self._commanded.clear()
        self._log("reset", dict(seed=seed))
        return self.observe()

    @property
    def decisions_left(self) -> int:
        return self.max_decisions - self.decisions_used

    # ------------------------------------------------------------------- tools
    def observe(self) -> dict:
        """What the robot can see: the segmented cloud in the tool frame and the recent response."""
        budget = self.env.spec.point_budget
        flat = self._obs[self.slot]
        features = df.observation_features(flat, budget)
        named = dict(zip(df.OBSERVATION_FEATURE_NAMES, (round(float(v), 5) for v in features), strict=True))
        pos, feat, valid, extra = df.unpack_observation(flat, budget)
        out = dict(decisions_left=self.decisions_left, tool_position=[round(float(x), 5) for x in extra[0:3]],
                   goal_from_tool=[round(float(x), 5) for x in extra[3:6]],
                   garment_points=int((valid & (feat[:, 0] > 0.5) & (feat[:, 1] < 0.5)).sum()),
                   arm_points=int((valid & (feat[:, 1] > 0.5)).sum()), **named)
        out["garment_response"] = self.response()
        self._log("observe", None, out)
        return out

    def points(self) -> dict:
        """The segmented point cloud the robot sees, in the tool's frame, as arrays.

        ``garment`` and ``arm`` are ``[N, 3]`` in metres relative to the tool, ``goal``
        is the shoulder relative to the tool. A program that needs the fingertip, the
        arm's axis or the garment's shape computes them from these.
        """
        budget = self.env.spec.point_budget
        pos, feat, valid, extra = df.unpack_observation(self._obs[self.slot], budget)
        out = dict(garment=pos[valid & (feat[:, 0] > 0.5) & (feat[:, 1] < 0.5)],
                   arm=pos[valid & (feat[:, 1] > 0.5)], goal=np.asarray(extra[3:6], dtype=float),
                   tool=np.asarray(extra[0:3], dtype=float))
        self._log("points", None, {k: (len(v) if v.ndim == 2 else None) for k, v in out.items()})
        return out

    def response(self) -> float | None:
        """Metres the garment centroid moved per metre commanded, over the recent window.

        ``None`` before the window fills. Well below one means the garment has stopped
        following the commands, which is what a stall looks like from the robot's side.
        """
        window = self._response_window
        if len(self._centroids) < window + 1:
            return None
        # ``_centroids[i]`` and ``_commanded[i]`` are both recorded after decision ``i``, so the
        # motion between consecutive centroids pairs with the later decision's command.
        moved = np.linalg.norm(np.diff(np.stack(self._centroids[-window - 1:]), axis=0), axis=1)
        commanded = np.asarray(self._commanded[-window:])
        keep = commanded > 1e-4
        return round(float(moved[keep].sum() / commanded[keep].sum()), 4) if keep.any() else None

    def move(self, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0,
             rx: float = 0.0, ry: float = 0.0, rz: float = 0.0, repeat: int = 1) -> dict:
        """Command a tool motion for ``repeat`` decisions; metres and radians per decision.

        The request is clipped to the controller's limits, the same ones the trained
        policy is bound by. Returns the observation after the last decision.
        """
        cfg = self.env.cfg
        repeat = int(repeat)
        if repeat < 1:
            raise ValueError("A command must last at least one decision")
        if repeat > self.decisions_left:
            raise DecisionBudgetExhausted(f"{repeat} decisions requested, {self.decisions_left} left")
        translation = np.asarray([dx, dy, dz], dtype=np.float64) / float(cfg.max_translation)
        rotation = np.asarray([rx, ry, rz], dtype=np.float64) / float(cfg.max_rotation)
        action = np.clip(np.concatenate([translation, rotation]), -1.0, 1.0)
        requested = dict(dx=dx, dy=dy, dz=dz, rx=rx, ry=ry, rz=rz, repeat=repeat)
        clipped = bool(np.any(np.abs(np.concatenate([translation, rotation])) > 1.0 + 1e-9))
        for _ in range(repeat):
            self._step(np.tile(action, (self.env.num_envs, 1)), "agent")
        out = self.observe()
        self._log("move", dict(requested=requested, clipped=clipped), out)
        return out

    def run_policy(self, decisions: int) -> dict:
        """Hand the next ``decisions`` decisions to the trained policy, if one was supplied."""
        if self._agent is None:
            raise RuntimeError("This session was started without a policy")
        decisions = int(decisions)
        if decisions < 1:
            raise ValueError("Ask the policy for at least one decision")
        if decisions > self.decisions_left:
            raise DecisionBudgetExhausted(f"{decisions} decisions requested, {self.decisions_left} left")
        for _ in range(decisions):
            self._step(self._agent.act(self._obs, deterministic=True), "policy")
        out = self.observe()
        self._log("run_policy", dict(decisions=decisions), out)
        return out

    def render(self, name: str | None = None) -> str:
        """Draw the garment, the arm and the tool from two viewpoints; returns the file path."""
        if self._image_dir is None:
            raise RuntimeError("This session was started without an image directory")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cell = self.env.cells[self.slot]
        cloth = self.env.positions()[self.slot]
        tool = self.env._anchor[self.slot]
        path = self._image_dir / (name or f"decision_{self.decisions_used:04d}.png")
        figure, axes = plt.subplots(1, 2, figsize=(11, 5), subplot_kw=dict(projection="3d"))
        for ax, (elev, azim) in zip(axes, ((20, -60), (80, -90)), strict=True):
            ax.scatter(*cell.arm_points[::4].T, s=1, c="0.6", label="arm")
            ax.scatter(*cloth[::6].T, s=1, c="tab:blue", label="garment")
            ax.scatter(*cloth[cell.opening_idx].T, s=14, c="tab:red", label="opening")
            ax.scatter(*tool[None, :].T, s=40, c="k", marker="x", label="tool")
            for name_, point in (("finger", cell.finger), ("elbow", cell.elbow), ("shoulder", cell.shoulder)):
                ax.scatter(*np.asarray(point)[None, :].T, s=25, marker="^")
                ax.text(*point, name_, fontsize=7)
            ax.view_init(elev=elev, azim=azim)
            ax.set_xlabel("x"), ax.set_ylabel("y"), ax.set_zlabel("z")
        axes[0].legend(loc="upper left", fontsize=7)
        figure.suptitle(f"{cell.garment} on body {cell.human}, decision {self.decisions_used}")
        figure.tight_layout()
        figure.savefig(path, dpi=110)
        plt.close(figure)
        self._log("render", dict(path=str(path)))
        return str(path)

    def report(self) -> dict:
        """Task metrics, for the evaluator; a program that calls this is reading privileged state."""
        row = self.trace[-1] if self.trace else {}
        self._log("report", None, row)
        return dict(row)

    # ----------------------------------------------------------------- private
    def _step(self, actions, source: str) -> None:
        cfg = self.env.cfg
        obs, _, done, infos = self.env.step(np.asarray(actions, dtype=np.float32), reset_on_done=False)
        self.decisions_used += 1
        info = infos[self.slot]
        if info.get("sim_error"):
            raise RuntimeError("Simulator failure during an agent episode")
        self._obs = obs
        budget = self.env.spec.point_budget
        features = df.observation_features(obs[self.slot], budget)
        _, _, _, extra = df.unpack_observation(obs[self.slot], budget)
        self._centroids.append(np.asarray(features[2:5]) + np.asarray(extra[0:3]))
        self._commanded.append(float(info["commanded_translation_m"]))
        self.trace.append(dict(decision=self.decisions_used, source=source,
                               **{k: (float(v) if isinstance(v, (int, float, np.floating)) else bool(v))
                                  for k, v in info.items()
                                  if k in ("upperarm_ratio", "forearm_ratio", "tracking_error", "grasp_valid",
                                           "commanded_translation_m", "accepted_anchor_translation_m",
                                           "collision_rejected_substeps", "tether_rejected_substeps")}))
        if self._render_every and self._image_dir is not None and self.decisions_used % self._render_every == 0:
            self.render()
        if bool(np.any(done)) and self.decisions_used < self.max_decisions:
            raise RuntimeError("Episode ended before its decision budget")

    def _log(self, call: str, arguments=None, result=None) -> None:
        self.calls.append(dict(decision=self.decisions_used, call=call, arguments=arguments,
                               result=result if call in ("move", "run_policy", "render") else None))


def run_program(program: Path, cfg, out: Path, *, cell: str, seed: int, checkpoint: Path | None,
                max_decisions: int, render: bool, render_every: int = 0) -> dict:
    """Execute ``program``'s ``policy(session)`` for one episode and record everything."""
    from dataclasses import replace

    from .dressing_env import GenesisIPCDressingEnv
    from .dressing_live import LiveCellFactory
    from .recovery_intervention import episode_summary

    out.mkdir(parents=True)
    started = time.perf_counter()
    source = program.read_text()
    env = GenesisIPCDressingEnv(replace(cfg, workspace=str(out / "assets")), num_envs=1,
                                cell_factory=LiveCellFactory(cfg.live))
    result = dict(completed=False, program=str(program), program_source=source, cell=cell, seed=seed,
                  max_decisions=max_decisions, env=cfg.to_dict())
    try:
        agent = None
        if checkpoint is not None:
            from .physics_gradient_actor import load_agent

            agent = load_agent(checkpoint, env.spec.point_budget, env.action_dim, "cuda")
        session = DressingSession(env, max_decisions=max_decisions, agent=agent, render_every=render_every,
                                  image_dir=out / "images" if (render or render_every) else None)
        namespace: dict = {}
        exec(compile(source, str(program), "exec"), namespace)
        if "policy" not in namespace:
            raise ValueError("The program must define policy(session)")
        session.reset(seed)
        error = None
        try:
            namespace["policy"](session)
        except DecisionBudgetExhausted as exhausted:
            error = f"budget: {exhausted}"
        if session.decisions_left:
            # An episode that ends early is padded by holding still, so every arm is scored
            # over the same horizon.
            hold = np.zeros((env.num_envs, env.action_dim), dtype=np.float32)
            while session.decisions_left:
                session._step(hold, "hold")
        result.update(program_error=error, decisions_used=session.decisions_used,
                      calls=session.calls, trace=session.trace, **episode_summary(session.trace))
        result["completed"] = True
        return result
    finally:
        result["seconds"] = time.perf_counter() - started
        (out / "result.json").write_text(json.dumps(result, indent=1) + "\n")
        env.close()


def main():
    import argparse
    from dataclasses import replace

    from . import train_sac
    from .sac import SACAgent

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--program", type=Path, required=True, help="A file defining policy(session)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cell", default="tshirt_26:14046")
    p.add_argument("--seed", type=int, default=9901)
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Supplies the environment contract, and the policy if the program calls run_policy")
    p.add_argument("--no-policy", action="store_true", help="Refuse run_policy: the agent must drive alone")
    p.add_argument("--max-decisions", type=int, default=300)
    p.add_argument("--render", action="store_true")
    p.add_argument("--render-every", type=int, default=0, help="Also draw the state every N decisions")
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    payload = SACAgent.read_checkpoint(args.checkpoint)
    targs = train_sac.build_parser().parse_args(["--eval-only"])
    train_sac.restore_resume_args(targs, ["--eval-only"], payload)
    train_sac.resolve_defaults(targs)
    garment, body = args.cell.rsplit(":", 1)
    cfg = replace(train_sac.dressing_config(targs), cells=((garment, int(body)),),
                  contact_force_readout=False, decision_watchdog=False)
    result = run_program(args.program, cfg, args.out, cell=args.cell, seed=args.seed,
                         checkpoint=None if args.no_policy else args.checkpoint,
                         max_decisions=args.max_decisions, render=args.render, render_every=args.render_every)
    print(json.dumps({k: v for k, v in result.items()
                      if k in ("completed", "program_error", "decisions_used", "success", "sustained_coverage",
                               "final_coverage", "max_coverage", "max_tracking_error",
                               "whole_episode_grasp_valid", "seconds")}), flush=True)


if __name__ == "__main__":
    main()
