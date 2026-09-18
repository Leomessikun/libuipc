"""Is a teacher's dressing action reachable inside the learned behavior prior?

Reverses the trained behavior flow on saved (observation, action) pairs and groups the
result by whatever distinguishes them — garment, body, episode outcome, the phase of
the episode — so that the prior's support can be compared where dressing succeeds and
where it fails. Nothing is trained and no simulator runs.

Read the three columns together. A group whose latent norms sit in the bulk of the
prior's own normal and whose reconstruction error is at the integration error's scale
is a group the prior can already produce, and steering its noise is worth trying. A
group in the far tail, or one the forward pass cannot reproduce, is a group the prior
does not contain, and no amount of steering will find it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from uipc_manip import flow_reversal as fr  # noqa: E402


def load_episodes(dataset: Path):
    """Every reconstructed episode with observations, actions and its own outcome."""
    rows = []
    for path in sorted(dataset.glob("episodes/*.npz")):
        data = np.load(path, allow_pickle=True)
        record = data["record"].item()
        record = json.loads(record) if isinstance(record, str) else record
        privileged = data["privileged"].astype(np.float64)
        rows.append(dict(path=path, observations=data["obs"], actions=data["actions"],
                         garment=record["garment"], human=int(record["human"]),
                         upperarm=privileged[:, 25], forearm=privileged[:, 24],
                         tracking=privileged[:, 29],
                         final_upperarm=float(privileged[-1, 25]),
                         max_tracking_cm=float(privileged[:, 29].max())))
    if not rows:
        raise FileNotFoundError(f"No episodes under {dataset}")
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True, help="An FQL checkpoint with a behavior flow")
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--success-coverage", type=float, default=0.7)
    p.add_argument("--grasp-limit-cm", type=float, default=2.0)
    p.add_argument("--late-from", type=int, default=120, help="Decisions from here on count as the contact phase")
    p.add_argument("--train-bodies", default="14045,14046,14047",
                   help="Bodies the prior was fitted on; the rest are reversed out of sample")
    args = p.parse_args()
    from uipc_manip.fql import FQLAgent

    agent, metadata = FQLAgent.load(args.checkpoint, device=args.device)
    episodes = load_episodes(args.dataset)
    groups: dict[str, dict[str, list]] = {}

    def add(name, observations, actions):
        entry = groups.setdefault(name, dict(observations=[], actions=[]))
        entry["observations"].append(np.asarray(observations))
        entry["actions"].append(np.asarray(actions))

    train_bodies = {int(b) for b in args.train_bodies.split(",") if b.strip()}
    for episode in episodes:
        split = "in_sample" if episode["human"] in train_bodies else "held_out"
        dressed = episode["final_upperarm"] >= args.success_coverage
        held = episode["max_tracking_cm"] <= args.grasp_limit_cm
        outcome = "succeeded" if (dressed and held) else ("dressed_but_slipped" if dressed else "failed")
        late = slice(args.late_from, None)
        add("all", episode["observations"], episode["actions"])
        add(outcome, episode["observations"], episode["actions"])
        add(f"{outcome}/contact_phase", episode["observations"][late], episode["actions"][late])
        add(f"garment/{episode['garment']}", episode["observations"], episode["actions"])
        add(f"split/{split}", episode["observations"], episode["actions"])
        add(f"split/{split}/{outcome}", episode["observations"], episode["actions"])
        # The decisions where this episode's sleeve is on the upper arm are the ones a
        # prior has to contain if it is ever to finish a dressing.
        on_upper = episode["upperarm"] > 0.05
        if on_upper.any():
            add(f"{outcome}/on_upper_arm", episode["observations"][on_upper], episode["actions"][on_upper])
    print(f"{len(episodes)} episodes, {sum(len(e['actions']) for e in episodes)} transitions, "
          f"behavior flow of {agent.cfg.flow_steps} Euler steps over {agent.action_dim} action dimensions")
    print(f"\n{'group':34s} {'n':>6s} {'|z| median':>10s} {'percentile':>10s} {'>99th':>6s} "
          f"{'recon median':>12s} {'recon p90':>10s} {'integration':>11s}")
    summary = {}
    for name, entry in groups.items():
        report = fr.reversal_report(agent, np.concatenate(entry["observations"]),
                                    np.concatenate(entry["actions"]))
        s = fr.summarize(report)
        summary[name] = s
        print(f"{name:34s} {s['count']:6d} {s['latent_norm']['median']:10.3f} "
              f"{s['latent_percentile']['median']:10.3f} {s['fraction_above_99th']:6.2f} "
              f"{s['reconstruction_error']['median']:12.4f} {s['reconstruction_error']['p90']:10.4f} "
              f"{s['noise_round_trip']['median']:11.4f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(checkpoint=str(args.checkpoint), dataset=str(args.dataset),
                                        checkpoint_updates=int(agent.updates), checkpoint_metadata=metadata,
                                        flow_steps=int(agent.cfg.flow_steps), action_dim=int(agent.action_dim),
                                        episodes=[{k: v for k, v in e.items()
                                                   if k in ("garment", "human", "final_upperarm", "max_tracking_cm")}
                                                  for e in episodes],
                                        groups=summary), indent=1) + "\n")
    print(f"\nwritten {args.out}")


if __name__ == "__main__":
    main()
