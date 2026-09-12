# Where the training run's wall clock actually goes

Date: 2026-09-12. Measured from `wang_teacher_r13_s1/train_log.csv`, the full 26.2-hour region-13 run.

Two figures this project has been planning against are wrong, and both matter.

| | Seconds | Share | Per transition |
|---|---:|---:|---:|
| Environment | 59,316 | **63.0 %** | 0.220 s |
| **Evaluation** | 24,080 | **25.6 %** | — |
| Updates | 9,694 | **10.3 %** | 0.036 s |
| World rebuilds | 938 | 1.0 % | — |
| Checkpoints | 610 | 0.6 % | — |

270,216 transitions and 270,151 updates over 94,162 s: **10,331 transitions per hour overall, 13,881
per hour if evaluation is excluded**, at one update per transition.

## The corrections

- **Updates are a tenth of the run, not two fifths.** An earlier figure of 40 per cent came from a
  different, smaller run (`onepolicy_t26-t68`). One update costs **35.9 ms**, so the cost model is
  `seconds per transition = 0.220 + k × 0.036` for an update-to-data ratio of k: 0.256 s at k=1
  (14.1k per hour), 0.364 s at k=4 (9.9k), 0.508 s at k=8 (7.1k). **Raising the update ratio is
  cheaper than we have been assuming** — k=4 costs 29 per cent of throughput, not the 74 per cent a
  0.31 s simulator step would imply.
- **The 0.31 s per transition quoted throughout this project is the all-in figure including
  evaluation.** The simulator itself costs 0.220 s.

## The largest recoverable cost is evaluation

A quarter of the run is evaluation, and **14 of its 27 rounds measured nothing**, because a decision
watchdog trip raises for a whole world and ends all 25 episodes at once
(`2026-09-11-pretraining-infrastructure-review.md`). Two fixes for that are already written and
committed and take effect at the next start: evaluation worlds build without the watchdog, and a
round carrying a simulator error can no longer take `best.pt`.

Recovering that quarter is worth more than any change to the learner: at 13,881 transitions per hour
instead of 10,331, the same 265,584 transitions take 19.1 hours instead of 26.2.
