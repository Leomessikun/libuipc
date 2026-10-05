# Sim-to-real post-training: measurement and research decision

The requested prior-art map and 210-attempt robustness screen are complete.
There is no verified new algorithm or hardware transfer result yet. The
measurement has an initialization limitation that prevents a strong conclusion
about retention of the historical sim gains.

## What the new measurement establishes

One worker ran 30 batches in **3.44 worker-hours**, with a recorded peak total GPU
allocation of **88.16 GiB**. All 210 planned outcomes are saved. The original
post-settling byte-hash pairing check admitted **zero pairs**: independent
settling changes vertices, and changing material also changes the settled drape.
That primary analysis is retained as failed, rather than silently relaxed.

A **secondary matched-reset analysis** verifies identical static body/garment/
grasp geometry, placement, seed, and configurations except declared interventions.
All cases pass. Within-condition post-settle discrepancies are at most 0.759 mm.
Representative descriptive results:

| Condition | fmvp_sim | r1 | flow |
|---|---:|---:|---:|
| Nominal | 3/7 | 3/7 | 4/7 |
| Bending stiffness ×2 | 4/7 | 1/7 | 2/7 |
| Density ×1.5 | 5/7 | 3/7 | 3/7 |
| Point dropout 30% | 3/7 | 1/7 | 1/7 |
| Identical nominal repeat | 4/7 | 3/7 | 3/7 |

The checkpoint ordering varies, but this seven-body/single-garment screen does
not reproduce the nominal r1 advantage. Flow's one-case nominal edge also
disappears in the repeat. Therefore it **cannot establish retention or collapse
of the historical 201-case gains**. It does not identify a dominant real material
parameter, prove simulator exploitation, or measure hardware transfer. All
within-condition exact McNemar p-values are at least 0.25. Full numbers, intervals,
reset checks and repeatability: [results](2026-10-05-sim2real-robustness-results.md).

## Prior art that constrains the contribution

The [primary-source map](2026-10-05-sim2real-posttraining-study.md) covers brief
questions 1–3. DPPO provides genuine sim post-training → hardware evidence but
uses estimated object state. FMVP uses real fine-tuning data; Dressing in Motion
explicitly does not test sim-to-real transfer. EPOpt/robust baseline regret,
DROPO/SimOpt, S-HCI-GIBO and Sim2Act already cover ensemble robustness, target
calibration and decision/ranking-aware improvement. A KL anchor, stronger flow
network or more physics randomization is not a new method contribution.

Follow-up searches also found ties-aware reward learning, A2D/BIG, CoPlanner,
Set-Supervised Diffusion Policy and feedback action chunking. These remove broad
claims about ties, observable teachers, shared prefixes, corrective sets or
within-chunk feedback being new.

## Two candidates, with a research decision

| Candidate | Falsifiable prediction | Strongest objection | Smallest test and cost |
|---|---|---|---|
| Calibrate base-relative improvement/tie/regression labels with few target comparisons. **Deprioritize.** | At equal target budget, halve harmful updates while retaining useful positive gain. | Sim2Act/S-HCI-GIBO already calibrate decision-relevant improvement; ties-aware preference models already exist. | Completed CPU-only leave-one-condition-out screen: calibrated selection 19/32 successes versus frozen base 20/32; same one harmful outcome as simple calibration. No extra IPC. The held-out bank has no successful alternative to a base failure, so this does not test calibration when repairs exist. |
| Post-train from **complete observable feedback repairs**, retaining repair identity through prefix/branch/continuation. **Next method hypothesis.** | Same-bank/query/capacity student retains at least 10 pp more unseen-domain success than independent action/branch supervision; shuffling repair associations removes the benefit. | May be robust POMDP planning plus latent-mode/set-supervised imitation. Trees, mixture optimization and causal feedback are established. | First reject the teacher mechanism with 84 new candidate continuations and 28 actual held-domain controller runs: **112 attempts, roughly 2–4 single-worker hours**, four-hour ceiling. This is not another old-checkpoint benchmark. Then matched actor absorption and real comparison are required; costs are additional. |

The [second candidate](2026-10-05-feedback-repair-posttraining.md) includes the
finite mixture operator, actor losses, source/target protocol, missing-data rules,
baselines and stopping criteria. Its CPU compiler is implemented and three
analytic checks pass. Those checks are not dressing success or novelty evidence.
The physical candidate pilot is now running under the owner-authorized
continuation: supervisor 659533, output `output/uipc_manip/feedback_repairs_20261005/`.
The frozen source gate compares against a robust flat-repair mixture before
spending target attempts. Actor training has not been launched.

The next study must produce **new successful, observable repairs** and transfer
them into an actual actor. Reweighting weak labels or repeatedly evaluating the
same three checkpoints does not supply that capability. If ordinary history-flow
distillation matches the proposed mechanism, drop the method claim. If a teacher
works only with hidden simulator state, it is not the needed supervision source.

Before claiming real transfer, compare on a static manikin with randomized matched
reset blocks: 12 pairs ×2 repetitions = **48 real attempts**, about **1.6–3.2
robot-hours** assuming 2–4 minutes including reset. Separate actor evaluation,
unseen ClothesNet garments and GRAB motions come afterward. Simulation stress
cannot substitute for that hardware evidence.
