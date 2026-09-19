# What actually fails on the 25 dressing cells — 2026-09-19

Status: complete, and free. It re-reads `output/uipc_manip/expert_r13_heldout_s0/records.json`,
the scripted teacher's own 25-cell evaluation. Nothing was simulated for it.

The feasibility question is whether "train dressing fast on one workstation" has any
hope. Before spending a training run on it, it is worth knowing what the failures on
this benchmark actually are, because the answer decides which of "the learner cannot
find the motion", "the learner cannot hold the grip" and "the learner does not know
when to stop" is the thing to work on.

## The decomposition

| | cells | |
|---|---:|---|
| succeeds: final coverage >= 0.7 and whole-episode grasp <= 2 cm | 6 | |
| **reaches >= 0.7 at some decision, then loses it** | 4 | peak 0.997-1.000, final 0.000 |
| never reaches 0.7 | 10 | |
| **reaches >= 0.7 at some point, total** | **15** | a solution exists and was executed |

The four that reach it and lose it:

| cell | peak | final | worst tracking |
|---|---:|---:|---:|
| tshirt_4 / 14047 | 1.000 | 0.000 | 0.45 cm |
| tshirt_26 / 14045 | 0.999 | 0.000 | 1.14 cm |
| tshirt_26 / 14048 | 0.998 | 0.000 | 3.42 cm |
| tshirt_68 / 14049 | 0.997 | 0.000 | 1.39 cm |

Three of the four hold the grip comfortably inside the 2 cm rule the whole time. The
sleeve goes fully onto the upper arm and then comes off again while the controller
keeps driving. This is not a manipulation failure.

The ten that never reach it carry the opposite signature: nine have a worst tracking
error of 1.84 cm or more and eight are past the 2 cm rule, up to 3.17 cm. All five
`tshirt_392` bodies are here, peaking between 0.23 and 0.50 with tracking 2.87-3.17 cm.

## What it means

A 733-second scripted controller already puts the sleeve fully on the upper arm in 15
of 25 cells. The task is not a needle in a 300-decision haystack. What separates 6 from
15 is **stopping**, and what separates 15 from 25 is **not pulling hard enough to lose
the grip** — with one garment, `tshirt_392`, failing that way on every body.

Both are properties a reward can express, and the reward here does express them: Wang's
task term is the instantaneous distance the opening has travelled along the arm, paid
at every decision, so coming off again costs reward immediately, and losing the grip
stops the cloth following the tool and so stops progress. Across the 25 cells the two
track each other: worst tracking is 0.45-1.14 cm on the cells that succeed and
1.84-3.17 cm on the cells that never reach threshold.

So the reward is not obviously misaligned with the score, and the failures are not
exotic. That sharpens the feasibility question rather than answering it: if the motion
exists, the reward pays for holding it, and a script finds it 15 times out of 25, the
open question is why no learner in this repository has matched the script.

## The one learned result that beats the script

`tshirt_68 / 14049` is one of the four "reached it and lost it" cells: the teacher
peaks at 0.997 and ends at 0.000. FQL, with that cell **withheld from its training
split**, ends at coverage 1.000 with a valid grasp
(`output/uipc_manip/fql_pretrain_20260917/summary.json`, actor, round 0, split
`validation`). Its behaviour-cloning reference fails every cell. One seed, two rounds,
and every FQL episode trips the `early_turn` flag, so this is a signal and not a
result — but it is the only case in this repository of a learner finishing a cell the
script does not.

## Reproduce

```bash
python - <<'EOF'
import json
rows = json.load(open('output/uipc_manip/expert_r13_heldout_s0/records.json'))
reach = [r for r in rows if r['max_upperarm_ratio'] >= 0.7]
final = [r for r in rows if r['final_upperarm_ratio'] >= 0.7 and r['max_tracking_error'] <= 0.02]
print(len(final), len(reach), len(rows))
EOF
```
