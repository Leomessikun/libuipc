"""Join several behavior datasets into one the flow trainer can read.

A prior needs both halves of what has been collected: the teacher's dressing actions,
and the recoveries measured to help at the states a policy actually visits. They live
in separate collections with the same observation and action contract, so this merges
them by reference, keeping each episode's own provenance in the merged metrics.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sources", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--copy", action="store_true", help="Copy the tapes instead of linking them")
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    (args.out / "episodes").mkdir(parents=True)
    records, manifests = [], []
    for source in args.sources:
        manifest = json.loads((source / "manifest.json").read_text())
        if not manifest.get("completed") or manifest.get("transition_schema") != "explicit_successors_v1":
            raise ValueError(f"{source} is not a completed explicit-successor dataset")
        manifests.append(dict(source=str(source), **{k: manifest[k] for k in
                                                     ("obs_dim", "action_dim", "transition_schema")}))
        if any(m["obs_dim"] != manifests[0]["obs_dim"] or m["action_dim"] != manifests[0]["action_dim"]
               for m in manifests):
            raise ValueError("Sources disagree on the observation or action contract")
        for record in json.loads((source / "episode_metrics.json").read_text()):
            index = len(records)
            target = Path("episodes") / f"episode_{index:05d}.npz"
            origin = (source / record["path"]).resolve()
            if args.copy:
                shutil.copy2(origin, args.out / target)
            else:
                (args.out / target).symlink_to(origin)
            merged = dict(record)
            merged.update(index=index, path=str(target), source_dataset=str(source),
                          source_path=str(origin), source_sha256=record.get("source_sha256", str(origin)))
            records.append(merged)
    manifest = dict(completed=True, transition_schema="explicit_successors_v1",
                    obs_dim=manifests[0]["obs_dim"], action_dim=manifests[0]["action_dim"],
                    merged_from=manifests, episodes=len(records),
                    usable_for="behavior model only; a source may carry labels that do not match its successors")
    (args.out / "episode_metrics.json").write_text(json.dumps(records, indent=2) + "\n")
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(dict(episodes=len(records), sources=[str(s) for s in args.sources])), flush=True)


if __name__ == "__main__":
    main()
