# Dressing research: reliable recovery supervision before another RL loss

Status: research and bounded native diagnostic complete; proposed learning method
below is **not implemented or validated**. No policy training was started.

## Decision

Keep IPC, existing demonstrations, and the dense action-conditioned SAC baseline.
First isolate restored-rollout variability and make existing expert data usable
under one physical-quality/evaluation contract. Then test whether IPC can supply
successful **alternative recovery routes at learner-visited states**, with equal
movement and compute controls. Only train on these corrections if the teacher
actually improves sustained dressing. Do not repeat an unvalidated local gradient
loss or assume a longer action sequence is itself the missing contribution.

The objective is robust completed dressing per total wall time, with an observation
policy that runs without IPC on hardware. It is not high GPU utilization, a better
critic loss, or a successful short geometric probe.

## What the evidence establishes

### New repeatability decomposition

Script: [diagnose_dressing_repeatability.py](../../scripts/diagnose_dressing_repeatability.py).
Artifacts: `output/uipc_manip/dressing_research_20260917/repeatability/{result.json,traces.npz}`.
Checkpoint: `dressing_redesign_20260916/warm_sac/checkpoints/checkpoint_00127416.pt`.
One development cell, tshirt_26/body 14046, decision 60, seed 1097. Saved architecture
and environment settings are retained; observation augmentation is disabled.

After a 60-decision approach, take one native snapshot. Test deterministic actor
inference 32 times on a cached observation, then restore/query observations three
times. Run three closed-loop and three fixed-action 12-decision continuations,
interleaving order. The fixed sequence is recorded from the first closed-loop run.
No solver tolerances, rewards, controller rules, or policy weights are changed.

| Measurement | Result |
|---|---|
| Cached-observation action spread, 32 calls | Exactly 0 |
| Restored observation maximum component spread | 1.19e-7 |
| Restored cloth position errors | Exactly 0 for all six branches |
| Fixed-action final upper-arm coverage | .22954, .24619, .19703 |
| Closed-loop final upper-arm coverage | .27598, .20929, .21931 |
| Fixed-action final coverage range | .04916, or 4.916 percentage points |
| Closed-loop final coverage range | .06669, or 6.669 percentage points |
| Fixed-action maximum cloth-coordinate spread after one / 12 decisions | .00538 / .14550 m |
| Fixed-action command spread at every decision | Exactly 0 |
| Grasp / controller | All valid; zero collision or tether rejections |
| Native cost | 132 simulator decisions, 61.41 s; completed |

Cloth-coordinate spread is the maximum component difference across corresponding
vertices and runs, **not mean displacement or Euclidean RMS**. Flattened point-cloud
observation differences also include changed sampling/order/flags; they are not
physical displacements of corresponding points.

Variability exists without policy feedback. Its source lies somewhere in forward
simulation, recovered state, or the numerical execution path. This does not prove
a specific restore bug, nor that stochastic dynamics prevent RL. It makes small
single-pair advantages and finite differences unreliable in this case.

Source inspection found FEM positions, velocities and previous positions are
serialized (`finite_element_method.cu`, dump/recover near line 1109). Global vertex
positions/previous positions and PT/EE/PE/PP trajectory candidates also have
recovery paths. Do not claim velocities are simply omitted. Unrecorded caches,
adaptive state, convergence and parallel numerical ordering remain hypotheses.

The next causal test is fixed-action replay of the **entire saved prefix from a
fresh world versus continuation after snapshot recovery**. Compare full state at
the branch and each substep, including velocities and relevant solver/cache state.
If only recovery diverges, repair recovery. If both diverge, examine convergence
and numerical sensitivity under matched states; quantify outcome distributions.
The requirement is reliable ranking at a meaningful effect size, not indefinite
pursuit of bitwise reproducibility.

### Existing expert trajectories: feasibility and data compatibility

Audited `output/uipc_manip/expert_r13_heldout_s0/`: 25 episodes, five garments and
bodies 14045–14049. This is **one diagnostic subset, not a complete corpus inventory**.
Summary: `output/uipc_manip/dressing_research_20260917/expert_audit.json`.

| Predicate | Episodes |
|---|---:|
| Final geometric success, upper-arm coverage >= .7 | 11 / 25 |
| Ever reach that threshold | 15 / 25 |
| Final success and maximum tracking error <= .02 m | 6 / 25 |
| Existing paper-style admission filter | 0 / 25 |

The early-turn predicate rejects this subset. Physical success, a reference
teacher's path convention, and admission to a training dataset are different
questions. Do not silently relax a criterion after seeing results. Define and
version the intended deployment/evaluation contract, retaining the original
paper predicate as a separately reported comparison.

Four episodes reach approximately full coverage then finish at zero: tshirt_26
with bodies 14045 and 14048; tshirt_68/14049; tshirt_4/14047. Inspect cloth geometry
and the metric around these transitions before attributing them to slipping,
overpulling, or a geometric-intersection branch change. Not every failure is an
elbow snag. Episode return and final coverage have Spearman correlation .904 in
this subset: reward irrelevance or reward hacking is not established.

Successful tshirt_26/14046 and /14047 trajectories have final coverage .965 and
.973 and maximum tracking errors approximately .010 and .006 m. The existence
of feasible motions therefore is supported for some configurations. It does not
prove robust reachability for every garment, body, initialization or perturbation.

The sampled episode NPZ has `privileged[300,35]`, `actions[300,6]`, rewards, stages
and a record. It has no policy observation sequence or full cloth state. The 35D
privileged vector is a geometry summary, not a complete Markov physical state.
The current distillation loader expects `obs`, `actions` and collector metadata.
This diagnostic subset cannot directly enter that loader. Locate compatible
records in the existing corpus first; if observations must be reconstructed,
replay and store the **actual new trajectory**, since replay divergence means the
new observations cannot automatically inherit the old success labels.

The checked expert and current SAC manifests agree on the examined physical and
controller settings, including 6 substeps, friction .3, Young's modulus 6000,
bending .1, shear ratio .01, constraint strength 1e4 and 48 anchors. Equal config
values do not establish equal cold state or build, or calibration to real cloth.

The previous 2,000-update IQL/BC/replay-SAC controls used the 125k **policy replay**,
not all collected expert demonstrations. Their 0/4 development success results
cannot reject expert pretraining or establish a general offline-to-online failure.

### Earlier sequence optimization already tested the obvious next suggestion

See [historical gradients, Level 3](2026-09-13-physics-gradients.md) and
`output/uipc_manip/physics_gradient_trajopt/{samestate,control,lagcheck}`.
The 12-action adjoint optimization produced similar gradient rows across time,
mostly shifting/scaling the sequence together. Its inertia-only temporal chain
does not capture the complete friction/contact history.

| Same-state control | Scaled proxy, no optimization | Optimized sequence |
|---|---:|---:|
| tshirt_26/14049 elbow | .163 +/- .001 | .170 +/- .001 |
| tshirt_392/14046 elbow | .107 +/- .001 | .105 +/- .001 |

These are short continuation coverage results, not completed dressing. Much of
the advantage over the original heuristic was larger motion. A new proposal must
compare **different paths with matched command budgets**, including the scaled
proxy, rather than claim all improvements from longer sequences are gradient gains.

The latest [actor audit](2026-09-17-dressing-actor-audit.md) gave each of six
proposal types 0/6 strict improvement gates. That is six tested states, not six
training seeds or proof that IPC cannot help. Corrected task geometry derivatives
often agree with finite differences, while the critic-derived directions do not
pass the required agreement at both scales. Production `PhysicsActorSignal`
still uses the historical last-frame surrogate; the diagnostic chain is not a
validated replacement wired into production training.

## First-principles diagnosis

Let physical state be x, controller C, IPC transition F, observation O, and policy
history h. A deployed decision follows a = pi(h), x' = F(x,C(x,a)), o' = O(x').
An IPC actor derivative has to cross every relevant link, not just the cloth
position solve. Friction memory, six substeps, controller acceptance/clipping,
tool-relative coordinates, discrete visibility/voxel/neighborhood selection,
and a learned continuation value all matter. A correct local task derivative is
neither a correct long-horizon policy gradient nor a successful recovery plan.

There are several distinct obstacles:

1. **Comparison reliability:** the new fixed-action experiment demonstrates an
   unresolved source of trajectory variation. Small correction labels need
   repeated comparisons; repairing actor mathematics alone cannot remove it.
2. **Exploration across contact transitions:** a sleeve may need clearance,
   rotation or partial retreat before forward progress becomes possible. This is
   a route-selection hypothesis consistent with local-gradient failures, not
   proof that every elbow state requires one particular maneuver.
3. **Observation ambiguity:** an instantaneous partial surface view can hide folds,
   velocity and contact history. Test the already available history machinery;
   do not assume a bigger network or the 35D summary resolves that ambiguity.
4. **Supervision and objective contracts:** existing successful motions are useful
   only if observations, executed actions, episode boundaries and physical-quality
   labels align. Brief coverage and stable completed dressing differ.
5. **Interaction cost:** prior profiles put 96–98% of environment time in simulation;
   a short timer window attributed 56% to collision-candidate detection. Increasing
   learner updates or network size does not fix expensive or uninformative rollouts.

IPC's contact guarantees do not imply real material calibration or better policy
optimization. A precise simulator can represent a hard-to-learn task accurately.
Contact forces can be useful diagnostics without being sufficient policy inputs.
Do not infer that raw force features must work, or that their failure excludes
all other uses of mechanics.

## Literature and novelty boundary

| Primary source | Relevant result and limitation for this project |
|---|---|
| [Wang et al., Learning to Dress People with Diverse Poses and Garments](https://arxiv.org/abs/2306.12372) | Dense action-conditioned Q, regional teachers and distillation are part of the reference method. Different solver/material/controller behavior matters. Its mean arm coverage is not a success-rate number. |
| [DiffCloth](https://arxiv.org/abs/2106.05306) | Differentiable dry-friction cloth simulation supports dressing optimization; this is positive precedent, not validation of our IPC/SAC derivative chain. |
| [Suh et al., Do Differentiable Simulators Give Better Policy Gradients?](https://proceedings.mlr.press/v162/suh22b.html) | Stiffness and discontinuities complicate first-order gradient quality. An analytic derivative is not automatically the best policy learning estimator. |
| [Guided Policy Search](https://proceedings.mlr.press/v28/levine13.html), [DAgger](https://proceedings.mlr.press/v15/ross11a.html) | Trajectory-guided learning and supervision at learner-visited states are established. Our teacher must actually find useful recoveries. |
| [RLPD](https://proceedings.mlr.press/v202/ball23a.html) | Existing data can assist online off-policy learning; using it does not require making a separate large offline-RL project the core contribution. |
| [Q-chunking](https://arxiv.org/abs/2507.07969) | The critic conditions on an action chunk, supporting coherent exploration and chunk-level backups. A critic for a full committed chunk is not interchangeable with one for its first action. |
| [Adaptive Q-Chunking](https://arxiv.org/abs/2605.05544) | Adaptive action horizons are already prior art; contact-dependent duration alone is not a sufficient novelty claim. |
| [Diffusion Policy](https://arxiv.org/abs/2303.04137) | Multimodal sequence imitation and receding-horizon execution are strong established controls. The current WangFlowActor is not a generative flow-matching policy. |
| [Unbiased Asymmetric Reinforcement Learning under Partial Observability](https://arxiv.org/abs/2105.11674) | Privileged information does not automatically make a state-only critic valid for a history-dependent actor; its conditioning must match the policy/value semantics. |
| [MPC Scaffolding for Dexterous Manipulation](https://arxiv.org/abs/2609.14878), September 2026 preprint | MPC seed trajectories, pretraining, intermittent online MPC guidance and SAC have already been combined. Its rigid-object hardware results do not establish cloth performance. |
| [Dressing in Motion](https://arxiv.org/abs/2609.04759), September 2026 preprint | A motion-aware diffusion dressing policy provides a recent deployment baseline; diffusion, arm-relative observations and dressing are not new in themselves. |
| [Constraint demo methods](https://qinengwang-aiden.github.io/demos/constraint_demos/methods.html) | Useful pattern: propose paths, simulate, inspect failures and revise. The claw uses a reference geometric path; the rope uses staged controls and ideal grasp constraints. These are not evidence of a general deployed dressing policy. |

A defensible research question is whether **selective IPC evaluation of different
contact recovery routes provides better policy supervision per unit total cost**
than ordinary demonstrations, SAC, and planner distillation. Neither novelty nor
performance is established yet. If the benefit is only imitation or extra compute,
report it as that rather than a new RL breakthrough.

## Proposed minimal integration with SAC

1. Reuse the existing successful demonstrations for a policy prior. Preserve
   multiple valid routes rather than averaging incompatible action sequences.
   First compare ordinary BC and a standard sequence model on the same data.
2. Let the current learner visit training states. At a bounded selection of
   stalled/ambiguous contact states, retrieve or propose a few different short
   routes: current SAC, a matched scaled proxy, a successful demonstrated route,
   and a geometrically different clearance/reorientation route. The exact route
   families must come from observed failures, not a universal elbow heuristic.
3. Evaluate candidates with IPC under the actual controller. Keep equal travel,
   rotation and decision budgets. Use repeated comparisons when uncertainty is
   comparable to the proposed gain. Gradient-free candidates are mandatory;
   validated analytic gradients may refine them locally as a separate ablation.
4. Accept supervision only for physical validity and repeatable, sustained
   progress. Retain full continuations to check for later loss of coverage.
   An unvalidated critic must not be the only judge immediately inside a jam.
5. Add actual executed transitions to SAC replay and teach the actor accepted
   corrections at its own visited observation histories. If the teacher's first
   action only works when followed by a specific sequence, label/evaluate that
   continuation too; one isolated action label is insufficient evidence.

For the minimal **primitive-action** implementation, keep the current SAC Bellman
update. A candidate actor objective is

$$
L_\pi = \mathbb{E}_{h,a\sim\pi_\theta}
[\alpha\log\pi_\theta(a\mid h)-Q_\phi(h,a)]
 + \lambda\mathbb{E}_{(h,a^*)\sim D_{\rm verified}}
[-\log\pi_\theta(a^*\mid h)].
$$

Here h is deployable observation/action history, a* an accepted executed teacher
action, D_verified the correction set, alpha SAC's entropy coefficient, and lambda
the supervised weight. This is an established style of auxiliary supervision,
**not the novelty claim**. Conflicting teacher actions for indistinguishable h
require better history or multimodal modeling; changing the loss weight will not
resolve missing information.

A separate chunked-RL extension must condition Q on the actually committed action
sequence A. For k executed actions its return target has the form

$$
y = \sum_{j=0}^{k-1}\gamma^j r_{t+j} + \gamma^k V(h_{t+k}).
$$

Gamma is the per-decision discount; V must use the chosen chunk policy's consistent
entropy convention, and terminal transitions remove bootstrapping. If only the
first action is executed before replanning, do not use the originally proposed
H-action return as an unbiased target for Q(h,a0). Interruptible options need
their own consistent termination/value contract. Do not introduce this additional
algorithmic boundary before proving the recovery teacher useful.

## Experiments and stop rules

| Stage | Concrete deliverable | Decision |
|---|---|---|
| Restore/numerics | Cold-prefix versus recovered fixed-action replay, state comparison and bounded convergence ablation if needed | Fix a confirmed discrepancy; otherwise quantify variation and use repeated outcome comparisons |
| Existing-data contract | Inventory compatible corpus records; align observations/actions/quality and version splits; inspect high-to-zero coverage cases | Reuse valid data; reconstruct only missing records and reassess reconstructed outcomes |
| Prior control | Same-data single-frame/history BC and a standard sequence imitation baseline, full closed-loop episodes | Identify whether fitting, observation ambiguity or distribution shift is the remaining problem |
| Recovery teacher | Reuse existing probe infrastructure; compare distinct routes against SAC/scaled proxy under equal budgets | No policy training from a teacher that fails to beat its controls |
| Learning ablation | SAC + same existing data versus + verified IPC corrections; gradient-free versus gradient refinement | Require full-episode gains at matched total cost; distinguish teacher quality from student fitting |
| Generalization | New body/garment/pose/material/control settings; subsequently other contact tasks | Dressing development gains alone do not establish a general RL algorithm |

A first route pilot could use 12 training states, four candidates, eight decisions
and three repeats: 1,152 branch decisions, **plus approach and full-continuation
costs**. This is a proposed cap, not a sufficient statistical sample size. Freeze
candidate definitions, split, validity rules and meaningful effect size before
running. If uncertainty remains too large at the cap, record a tie/inconclusive
result rather than picking the best noisy sample or retuning thresholds. States
already used repeatedly, including 14046/14049 development cases, are not fresh
held-out evaluation.

For a subsequent training comparison, report at least several independent seeds
(three as an initial practical minimum, not a power guarantee), identical source
architectures/data, optimizer handling and full evaluation cells. Plot sustained
grasp-valid success against **total wall time**, including data conversion, teacher
rollouts, resets, backward solves and evaluation. Also report interactions, coverage,
tracking failures, solver failures and variation. A solver-fidelity advantage
requires a matched lower-cost teacher comparison; beating SAC alone does not
attribute the gain to IPC's fidelity.

For speed, prior measurements were 7.02/18.29/17.71 transitions/s at 1/8/24
environments. Two eight-environment MPS workers gave 1.61x aggregate throughput,
not yet a shared-learner speedup. Use this as a profiling lead, preserve bounded
policy lag and measure end-to-end learner throughput. Do not launch more workers
blindly or add redundant learner updates just to warm the GPU.

## Real-world interface

Deployment remains camera observations plus robot proprioception/history -> actor
-> bounded robot controller. IPC and privileged teacher state are training tools.
The student must be evaluated using only the deployed sensor interface; successful
privileged planning is not successful end-to-end deployment. Before hardware
transfer, match material/compliance/friction and observation/controller behavior,
then test robustness to their measured uncertainty. Solver accuracy alone does not
calibrate these. This research turn establishes no hardware success.

## Reproduction

```bash
env PYTHONPATH=build_raw/python/src:python OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 LD_LIBRARY_PATH=build_raw/Release/bin:/home/ge47gax/Toolchain/uipc_cuda128/lib /home/ge47gax/kun/genesis-world/.venv/bin/python scripts/diagnose_dressing_repeatability.py \
  --checkpoint output/uipc_manip/dressing_redesign_20260916/warm_sac/checkpoints/checkpoint_00127416.pt \
  --out output/uipc_manip/dressing_research_20260917/repeatability
```

The output path must be fresh; the completed artifact already occupies the path
above. The native run completed all six branches and validates the diagnostic's
actual execution. It does not validate the proposed learning method.
