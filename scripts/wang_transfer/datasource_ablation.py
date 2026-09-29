"""Data-source ablation: flow trained on the released checkpoint's rollouts vs on r1's, same bodies.

``indexes`` writes two indexes next to ``--out``:
* ``index_fmvp.json``: accepted episodes of the fmvp_sim collection (all training bodies), split ``train``;
* ``index_r1_matched.json``: r1 episodes of the main index, split ``train``;
restricted to the (body, garment) units both sources have, with equal episodes per unit;
both carry the main index's ``val`` episodes, so checkpoint selection is identical. The flows are then
trained with ``train_flow_policy.py`` (frozen encoder features, same settings as the main flow).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path



def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True, help="fmvp_sim collection directory")
    p.add_argument("--index", type=Path, required=True, help="main body-split index (r1 rollouts)")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    main_index = json.loads(a.index.read_text())
    val = [e for e in main_index["episodes"] if e["split"] == "val"]
    held = {e["body"] for e in main_index["episodes"] if e["split"] in ("val", "test")}
    fmvp, seen = [], set()
    for line in (a.run / "attempts.jsonl").read_text().splitlines():
        r = json.loads(line) if line.strip() else {}
        if not r.get("accepted"):
            continue
        path = Path(r["log"]).with_suffix("") / r["path"]
        if not path.exists() or str(path) in seen or r["body"] in held:
            continue
        seen.add(str(path))
        fmvp.append(dict(path=str(path.resolve()), run=a.run.name, garment=r["garment"], body=int(r["body"]),
                         region=int(r["body"]) // 1000 - 1, transitions=int(r.get("transitions") or 0),
                         stage=r.get("stage"), seed=r.get("seed"), split="train"))
    r1_all = [dict(e) for e in main_index["episodes"] if e["split"] == "train"]
    # Keep only (body, garment) units present in both sources, with the same number of episodes per unit.
    by = {}
    for src, eps in (("fmvp", fmvp), ("r1", r1_all)):
        for e in eps:
            by.setdefault((e["body"], e["garment"]), {}).setdefault(src, []).append(e)
    units = sorted(u for u, d in by.items() if "fmvp" in d and "r1" in d)
    fmvp, r1 = [], []
    for u in units:
        k = min(len(by[u]["fmvp"]), len(by[u]["r1"]))
        fmvp += sorted(by[u]["fmvp"], key=lambda e: e["path"])[:k]
        r1 += sorted(by[u]["r1"], key=lambda e: e["path"])[:k]
    a.out.mkdir(parents=True, exist_ok=True)
    for name, eps in (("index_fmvp.json", fmvp), ("index_r1_matched.json", r1)):
        (a.out / name).write_text(json.dumps(dict(episodes=eps + val, source=str(a.run if "fmvp" in name else a.index)),
                                             indent=1))
        print(f"{name}: {len(eps)} train episodes, {sum(e['transitions'] for e in eps)} transitions, "
              f"{len({e['body'] for e in eps})} bodies; + {len(val)} val", flush=True)
    print(f"shared (body, garment) units {len(units)}", flush=True)


if __name__ == "__main__":
    main()
