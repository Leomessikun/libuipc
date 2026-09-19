# The sleeve stays on if the tool stops — 2026-09-19

Status: complete, controlled, 2 runs of 4 cells. Nothing was trained and no solver,
reward or controller setting was changed.

`2026-09-19-what-actually-fails.md` found that four of the scripted teacher's
twenty-five cells reach an upper-arm coverage of 0.997-1.000 and end the episode at
0.000, three of them holding the grip comfortably inside the 2 cm rule throughout.
Whether that is worth anything turns on a question nobody had asked: once the sleeve is
on, does it stay there, or does it come off whatever the controller does?

Script: `scripts/stop_when_covered.py`. Artifacts:
`output/uipc_manip/stop_when_covered_20260919/{stop,drive}/`.

## Protocol

The scripted teacher drives all four cells from the same seeds. The two arms differ in
one thing and nothing else: in `drive` the teacher keeps control to the horizon, as it
always has; in `stop` a slot whose upper-arm coverage crosses 0.7 stops commanding —
zero action — and stays stopped. Scored by the project's rule: the minimum coverage
over the last twelve decisions at or above 0.7, and a whole-episode grasp tracking
error at or below 2 cm.

## Result

| cell | arm | crossed at | peak | final | sustained last 12 | worst tracking | success |
|---|---|---:|---:|---:|---:|---:|:--:|
| tshirt_4 / 14047 | drive | 168 | 1.000 | 0.000 | 0.000 | 0.51 cm | no |
| tshirt_4 / 14047 | **stop** | 170 | 0.862 | **0.862** | 0.858 | 0.49 cm | **yes** |
| tshirt_26 / 14045 | drive | 154 | 0.999 | 0.000 | 0.000 | 1.42 cm | no |
| tshirt_26 / 14045 | **stop** | 153 | 0.837 | **0.837** | 0.830 | 1.50 cm | **yes** |
| tshirt_26 / 14048 | drive | 159 | 0.996 | 0.000 | 0.000 | 4.04 cm | no |
| tshirt_26 / 14048 | stop | 170 | 0.829 | 0.829 | 0.820 | 3.68 cm | no (grasp) |
| tshirt_68 / 14049 | drive | 181 | 1.000 | 0.000 | 0.000 | 1.38 cm | no |
| tshirt_68 / 14049 | **stop** | 177 | 0.813 | **0.813** | 0.808 | 1.95 cm | **yes** |

**0 of 4 against 3 of 4.** The peak and the final coverage of every stopped slot are
the same number: after the tool stops, the coverage does not move at all. The sleeve
does not slide off. The one stopped cell that still fails does so on the grasp rule,
whose 3.68 cm was already spent before the crossing.

## What it settles

* The end-of-episode coverage is keepable. A controller that stops when it is covered
  keeps what it earned, so the benchmark is not asking for something the physics
  refuses.
* The scripted teacher goes from 6 of 25 to 9 of 25 on this rule by stopping, without
  changing a single command before the crossing.
* That headroom was available to every learner ever run here, and none of them took
  it. The best learned result on the full grid is 7 of 25 on coverage alone and is not
  reproducible within its own run; the reference SAC checkpoint scores 0 wherever it
  has been measured under coverage-and-grasp. A policy that learned nothing except
  when to stop would have beaten all of them.

That is a statement about the learners and not about the task, which is the direction
the feasibility question needed.

## Reproduce

```bash
python scripts/stop_when_covered.py --checkpoint <ckpt> --arm stop  --out <dir>/stop
python scripts/stop_when_covered.py --checkpoint <ckpt> --arm drive --out <dir>/drive
```
