#!/usr/bin/env python3
"""Convert one local GRAB sequence on CPU; no simulation or policy training."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
from uipc_manip.grab_motion import convert_grab


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--body", type=int, default=14046)
    p.add_argument("--start", type=float, default=0.)
    p.add_argument("--duration", type=float, default=3.)
    p.add_argument("--fps", type=float, default=30.)
    p.add_argument("--amplitude", type=float, default=.35)
    p.add_argument("--ramp", type=float, default=.3)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    motion = convert_grab(args.source, body_id=args.body, start_s=args.start,
                          duration_s=args.duration, fps=args.fps,
                          amplitude=args.amplitude, ramp_s=args.ramp)
    motion.save(args.out)
    print(json.dumps(dict(output=str(args.out.resolve()), **motion.metadata), indent=2))


if __name__ == "__main__":
    main()
