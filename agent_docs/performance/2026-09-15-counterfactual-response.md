# IPC counterfactual response experiments

## Scope and implemented comparisons

The owner requested trying response prediction, explicit Jacobian supervision,
symmetric counterfactual supervision, partial/history observations, and response
conditioning of dense Q. `uipc_manip.response_experiment` implements a bounded
native-IPC diagnostic. It reuses PointNet++ and normalized residual trunks; it
is not a second production dressing trainer. Existing `pretrain_offline` uses
sequence replay without IPC D/vertex correspondence and cannot silently consume
these labels. Diagnostic checkpoints are explicitly not production actor weights.

Data: eight reset episode groups, four cloth slots, twenty decisions per episode,
640 transitions per corpus. The encoder sees a fixed subset of 64/400 cloth
positions over four frames and past commands, not full state, velocities or D.
The decoder receives fixed material-coordinate queries. All spatial response
labels stay privileged. This known-mesh assumption is not evidence of garment
transfer. A second removal of half the visible points is a synthetic stress test,
not rendered RGB-D occlusion. Split entire reset groups, including all slots,
before augmentation; normalize responses with training-only scalar RMS.

Implemented objectives, with common batches/directions/optimizer budgets:

- response-only;
- explicit network JVP matching (`create_graph=True`);
- literal paired endpoint regression at `a +/- epsilon v`;
- normalized central-difference regression;
- no-history controls for both counterfactual forms;
- frozen random encoder reference.

Each representation gets frozen velocity/Jacobian probes and matched trainable
dense action-conditioned Q probes, with and without predicted per-point response.
Q targets are observed discounted returns under the random collection policy,
with remaining time included. These are offline prediction probes, not online
policy improvements. Constant response, constant tangent, commands-only velocity
and constant value baselines guard against misleading metrics. True perturbed
rollouts, not Taylor pseudo-labels, score counterfactual prediction.

## Why normalize the difference?

For a locally linear predictor and squared endpoint error, the average of the
positive and negative endpoint losses is approximately nominal error squared
plus `epsilon^2 * ||J v - D v||^2`. A smaller radius weakens literal supervision.
We retain this original objective as an arm; the alternative uses
`||(R(a+epsilon v)-R(a-epsilon v))/(2 epsilon) - D v||^2`.
The latter is finite-radius supervision, not exact equivalence to the explicit
Jacobian objective. Both use ordinary backward; only the explicit-JVP arm needs
mixed derivatives of the decoder. Tests verify that all differential objectives
actually update the encoder/history. More perturbations do not create independent
physical observations; they reuse one local linear model.

At bounded actions, shrink the symmetric radius using the available margin.
Never clamp perturbed actions while keeping an unadjusted Taylor target. Labels
are detached. Static vertex sampling masks do not differentiate a changing
contact-region selection. Dense, uniform, sensitivity/uniform and grasp-biased
sampling are available; evaluation always covers all vertices.

## Measured linearization validity

Real IPC accepted-state raw derivatives, including exported half-plane friction
history for friction .6. Radii are normalized actions; the physical decision
translation scale is 6 mm. Results below pool both perturbation signs at two
anchor decisions per episode/slot, 128 evaluations per radius.

| Friction | Radius | Median relative error | 90th percentile |
|---|---:|---:|---:|
| 0 | .005 | .056% | .985% |
| 0 | .02 | .088% | .429% |
| 0 | .05 | .136% | .374% |
| 0 | .1 | .247% | .727% |
| .6 | .005 | .199% | .489% |
| .6 | .02 | .706% | 1.927% |
| .6 | .05 | 1.706% | 4.603% |
| .6 | .1 | 3.417% | 9.385% |

Recovering and replaying the nominal action is not bitwise deterministic at the
solver's stopping tolerance. The no-friction maximum coordinate discrepancy is
0.387 micrometers; `repeatability.json` records it. Collector rejects discrepancy
over 10 micrometers. Small-radius errors must be interpreted against this floor.
These tests do not cover cloth-body simplex friction, stick/slip classification,
or jam states.

## Initial no-friction results

Three network training seeds, 500 pretraining steps each; 200 steps per value
probe; fixed episode split. Means, not statistical claims:

| Objective | Response RMSE (mm) | JVP cosine | Relative JVP RMSE | Dense Q MSE | Response Q MSE |
|---|---:|---:|---:|---:|---:|
| Response | .823 | .860 | .438 | .725 | .725 |
| Literal counterfactual | .823 | .860 | .438 | .724 | .725 |
| Explicit Jacobian | .833 | .883 | .368 | .721 | .721 |
| Normalized difference | .803 | .947 | .236 | .727 | .726 |
| Normalized difference, no history | 1.074 | .961 | .208 | .735 | .735 |

Crucially, the **training-mean tangent** has test cosine **.988** and relative
RMSE **.095**. This corpus has too little variation in D to use high JVP cosine as
proof of a mechanics-aware representation. Literal counterfactual supervision
adds almost nothing at epsilon .05. Normalization improves the decoder's slope,
but frozen velocity and dense value probes do not show corresponding gains.
History helps response prediction even when the no-history slope metric looks
better. This directly contradicts treating derivative accuracy alone as success.

Measured pretraining times on this small CUDA model average about 2.31 s for
response, 3.03 s literal CF, 3.46 s explicit Jacobian, and 2.95 s normalized
difference. These are not PointNet++ production-scale memory/cost benchmarks.

Artifacts (ignored run data): `output/iaql/response_data_20260915`,
`response_data_friction06_20260915`, `response_comparison_20260915`.
Further friction/OOD/sampling results are recorded below when complete.

## Interpretation and outstanding work

A functioning decoder need not force informative encoder features: a nearly
constant Jacobian can be represented in decoder parameters. Partial observations
can also be insufficient even with history. The current evidence does not meet
the proposed gate for expensive dressing RL. Do not substitute a learned tangent
into the actor; the separately implemented fresh actor continues to use IPC.

Needed for a stronger claim: varied contact regimes with annotated slide/jam,
material and garment variation, real partial observations, held-out human/garment
splits, and a transferable encoder protocol. Pullback metrics and counterfactual
wording are not novelty proofs. The contribution must be assessed against
[DiffSRL](https://arxiv.org/abs/2110.12352),
[Sobolev Training](https://arxiv.org/abs/1706.04859), and visual Jacobian methods;
this record makes no first-of-its-kind claim.

## Friction and out-of-distribution follow-up

Same budgets, three network seeds, whole-episode splits. Within friction .6:

| Objective | Response RMSE (mm) | JVP cosine | Relative JVP RMSE |
|---|---:|---:|---:|
| Response | .373 | .866 | .483 |
| Literal counterfactual | .370 | .869 | .476 |
| Explicit Jacobian | .423 | .867 | .429 |
| Normalized difference | .365 | .926 | .338 |
| Normalized difference, no history | .712 | .915 | .434 |
| Literal counterfactual, no history | .734 | .838 | .604 |
| Training-mean tangent | — | .939 | .419 |

There is a positive result: normalized difference improves tangent magnitude
accuracy over the constant teacher on this frictional corpus. However, its
frozen-encoder tangent MSE (.2818) is essentially the response-only value (.2820),
and velocity probes are no better than commands alone (~4.04e-5). A linear probe
cannot establish absence of all useful encoded information; it simply does not
supply the requested evidence. Response-conditioned Q remains inconsistent,
with no reliable advantage over the random-encoder dense value reference.

Train at friction 0, evaluate at .6 with no OOD training rows: response-only has
1.140 mm response RMSE; normalized difference 1.100 mm. JVP cosine rises from
.756 to .848, but the training-mean tangent reaches .894. The learned relative
JVP error (.802) also trails the constant teacher (.762). This is not evidence
of robust friction generalization.

Artifacts: `response_friction06_20260915`, `response_ood_friction06_20260915`,
and `response_report_20260915` under `output/iaql`. The report contains all
seed means/spreads, JSON and PNG/PDF plots; regenerate using
`scripts/summarize_response_experiment.py RUN... --out REPORT`.

**Current decision:** keep normalized finite-difference supervision as a useful
experimental objective, retain literal CF and explicit JVP as controls, and do
not promote this prototype into production dressing RL on the present evidence.
History is useful for response prediction here; derivative accuracy alone is
not the representation go/no-go criterion. Neither the method nor its novelty
has been established by this pilot.

## Vertex sampling control

Friction .6, seed 0, 500 steps, 64 sampled targets per batch row; held-out scoring
still uses all 400 vertices. This is a single-seed diagnostic, not a ranking claim.

| Sampling | Response-only RMSE (mm) | Difference RMSE (mm) | Difference relative JVP RMSE |
|---|---:|---:|---:|
| Uniform | .364 | .359 | .343 |
| Half sensitivity / half uniform | .362 | .384 | .344 |
| Sensitivity / grasp / uniform | .406 | .454 | .423 |

Grasp-biased supervision is not justified as the default by this test. Increasing
importance on selected vertices changes the training objective; reporting only
those vertices would hide the whole-cloth degradation.

At epsilon .05, the frictional linearization error has 99th percentile 9.26%
and maximum 10.10% (one of 128 signed probes exceeds 10%). A good median does
not authorize trusting every contact configuration at this radius.

Verification: 122 related CPU tests pass (one existing empty-distance warning),
including analytical loss scaling, teacher detach, encoder/history gradients,
a complete offline run and source-separated OOD splitting. Native collection
and every reported objective/value probe completed without a runtime failure.
Original strict bitwise nominal replay checking was replaced by recorded
repeatability and a 10-micrometer rejection threshold, not silently ignored.

Reproduce the OOD comparison after collecting both corpora with the same seed:

```bash
python -m uipc_manip.response_experiment run --data DATA_FRICTION0 \
  --ood-data DATA_FRICTION06 --out NEW_OOD_RUN --steps 500 --probe-steps 200 \
  --seeds 0 1 2 --variants response jacobian counterfactual difference --device cuda
```

Use `--data DATA_FRICTION06` without `--ood-data` for the within-friction split.
The two no-history variants are `no_history` (normalized difference) and
`counterfactual_no_history` (literal endpoint loss). Sampling controls use
`--target-sampling uniform|sensitivity|task --variants response difference --seeds 0`.
