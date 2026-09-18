# Is the action that helps inside the behavior prior? — 2026-09-19

Status: measurement complete on the branch `research/flow-latent-steering`. Nothing was
trained and no simulator ran. It answers, for this project's own flow prior, the
question that decides whether steering a prior's latent noise is worth building at
all: whether the actions that improve dressing are ones the prior could already
produce.

The method is the reversal of Tang, Chen, Wagenmaker, Finn and Levine's *Improving
Robotic Generalist Policies via Flow Reversal Steering*
([project page](https://flow-reversal-steering.github.io/)), applied to the behavior
flow of our own FQL checkpoint. A flow policy integrates noise forward through a
velocity field in a fixed number of Euler steps; carrying a reference action back
along the same discretization gives the noise that would have produced it. Three
numbers follow: how far that noise sits in the prior's own standard normal, whether a
forward pass reproduces the action, and how much the inversion itself drifts.

Code: `uipc_manip.flow_reversal`, `scripts/flow_prior_support_audit.py`,
`scripts/flow_reversal_macros.py`. Artifacts under
`output/uipc_manip/flow_reversal_20260919/`. The prior is the behavior flow of
`fql_pretrain_20260917/continue_a100_s17_30k/final.pt`: ten Euler steps over a
six-dimensional action, fitted to 7,500 transitions from 25 scripted-expert episodes,
bodies 14045–14047 in training and 14048–14049 withheld.

## The prior reproduces its own data, and only its own data

A six-dimensional standard normal has a median length of about 2.31; the percentile
column places each reversed action's noise in that distribution.

| Group | Transitions | Noise length, median | Its percentile | Fraction beyond the 99th | Reconstruction error, median | Inversion drift |
|---|---:|---:|---:|---:|---:|---:|
| Bodies the prior was fitted on | 4,500 | 1.120 | 0.025 | 0.00 | 0.034 | 0.379 |
| … of those, the successful episodes | 2,100 | 1.037 | 0.017 | 0.00 | 0.028 | 0.323 |
| Withheld bodies | 3,000 | 2.493 | 0.602 | 0.13 | 0.172 | 1.706 |
| … of those, the successful episodes | 600 | 1.986 | 0.316 | 0.22 | 0.131 | 1.566 |

On the bodies it was fitted on, the prior places the expert's actions at noise shorter
than 97 % of its own samples and reproduces them to 0.034 on actions whose median norm
is 0.924. On bodies it never saw, the same expert behaviour needs noise at the 60th
percentile, a fifth of the successful actions need the extreme tail, the
reconstruction error grows fivefold, and the inversion's own drift grows by four and a
half — the field itself is rougher out of sample, so the numbers there are both worse
and less trustworthy.

## The recovery that actually helps is outside the prior

The [counterfactual branch study](2026-09-18-recovery-decisions.md) found fixed
eight-decision macros that beat the policy's own continuation at 43 of 48 states.
Those actions come from outside the prior's data. Reversed at the same observations,
on bodies the prior *was* fitted on:

| Macro | States | Mean gain over the policy | Noise length, median | Its percentile | Fraction beyond the 99th | Reconstruction error |
|---|---:|---:|---:|---:|---:|---:|
| `lift` | 48 | +0.0166 | 5.527 | 1.000 | 1.00 | 0.294 |
| `forward` | 48 | +0.0103 | 5.430 | 1.000 | 0.98 | 0.265 |
| `retreat_outward` | 48 | +0.0038 | 5.244 | 1.000 | 1.00 | 0.226 |
| `outward` | 48 | −0.0025 | 5.436 | 1.000 | 1.00 | 0.243 |
| `retreat` | 48 | −0.0027 | 5.244 | 1.000 | 1.00 | 0.226 |

The macro that wins at its own state needs noise of length 5.378, beyond every sample
the prior draws, and the forward pass returns something 0.282 away from it.

**This is not a magnitude effect.** Each macro commands 8 mm of translation, which is
0.924 in normalized units; the expert data's own action norm has a median of 0.924 and
exceeds 0.9 in 75 % of its decisions. The macros are exactly the size of the actions
the prior was fitted on. They differ in direction, and that is enough to put them
outside it.

**Nor is it an artifact of the ten-step discretization.** Refining the inversion drives
the reconstruction error down while the noise stays in the same place (`tshirt_26`,
24 states, 120 pairs):

| Euler steps | Noise length, median | Its percentile | Fraction beyond the 99th | Reconstruction error | Inversion drift |
|---:|---:|---:|---:|---:|---:|
| 2 | 3.869 | 0.979 | 0.04 | 1.368 | 3.896 |
| 5 | 5.131 | 1.000 | 1.00 | 0.575 | 2.117 |
| 10 | 5.483 | 1.000 | 1.00 | 0.247 | 0.833 |
| 20 | 5.576 | 1.000 | 1.00 | 0.111 | 0.320 |
| 40 | 5.616 | 1.000 | 1.00 | 0.053 | 0.151 |

At two steps the inversion is too coarse to say anything: it lands in the bulk of the
prior and reproduces nothing. From five steps on the answer is stable, and as the
scheme converges the flow *can* reproduce the macro — the error falls to 0.053 — but
only from noise of length 5.6, which is beyond anything the prior draws. So the useful
action is representable by this flow and unreachable by sampling it.

## Reading

For this prior the answer is the second of the three that were possible: the flow can
represent the behaviour that improves dressing, but only from noise it would never
draw. That is worse than it sounds for latent steering. A noise-space policy may output
any vector, so nothing forbids a latent of length 5.6; but the critic and the actor
would then be working where the flow was never fitted, and the regularization toward a
standard normal that the published method relies on pulls directly against it. The
practical consequence is the same as if the action were absent: widen the prior with
data that contains those directions, and re-run this audit, before building a latent
learner on it. The audit costs seconds and is the natural acceptance test for a
retrained prior.

Two secondary readings are worth keeping. The prior's extrapolation to unseen bodies
is measurably poor in a way that is independent of any task metric, which is a cheap
diagnostic to reuse whenever a prior is retrained. And in the successful episodes'
contact phase the expert's median action norm is 0.000: what the prior has learned to
do there is hold still, which is what its data does.

## Limits

One prior, fitted to 7,500 transitions from one scripted controller; a wider prior may
well contain these directions, and this measurement is the way to check. The action
space here is one six-dimensional decision, not the action chunk the original method
reverses, so nothing is said about chunk-level latent structure. The reversal is the
explicit reverse of the explicit forward scheme, so it is approximate by construction;
the inversion-drift column is the scale its errors must be read against. Reconstruction
error is reported in normalized action units.
