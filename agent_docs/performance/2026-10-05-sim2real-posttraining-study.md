# Post-training whose gains survive transfer — study opened 2026-10-05

Request: [the brief at 67e0c5fa](2026-10-05-sim2real-posttraining-research-brief.md).
Questions 1–3 were investigated before launching question 4. This record separates
published evidence, our measurements, and proposed methods. No new policy has been
trained for this study. The cancelled M4/composite-teacher queues remain stopped.

**Completed measurement update:** all 210 predeclared attempts finished in 3.44
worker-hours. The frozen exact-state pairing check admitted **zero pairs** because
independent settling changes cloth vertices. A disclosed secondary matched-reset
analysis verifies identical spawn geometry/configuration and reports the outcomes.
Nominal success is fmvp_sim/r1/flow **3/7, 3/7, 4/7**; bending ×2 is **4/7, 1/7, 2/7**.
The nominal r1 advantage is absent in this small screen, so it cannot establish
retention or loss of the historical 201-unit gain. No real transfer was measured.
See the [decision memo](2026-10-05-sim2real-posttraining-decision.md) for the current
recommendation and [complete result table](2026-10-05-sim2real-robustness-results.md).

## 1. What the transfer literature actually establishes

“Works on a real robot” and “an improvement learned in simulation transfers” are
different experimental claims. D = dynamics gap, O = observation gap, A = action/
controller gap. An entry marked “not established” is not evidence of impossibility.

| Primary source | Mechanism; gap addressed | Target real data | Real evidence and limits |
|---|---|---|---|
| [ResiP, 2024](https://arxiv.org/html/2407.16677v2), §IV-C, App. C/E | Frozen BC policy plus closed-loop residual RL in simulation; distill and co-train visual policy. D/A through corrective behavior; O through rendering and co-training. | 10 or 40 real demonstrations, mixed with synthetic demonstrations. | One-leg assembly improves from roughly 20–30% to 50–60%. This is real-plus-sim co-training, not demonstration-free visual transfer. Local residuals also remain tied to the base policy's support. |
| [Policy Decorator, 2024](https://policydecorator.github.io/) | Bounded residual RL and progressive exploration around a frozen base. Primarily action exploration and preservation of useful BC behavior. | None for the reported simulated benchmarks. | ManiSkill/Adroit evidence; the source inspected does not establish that sim improvements survive deployment. Residual clipping alone is not a transfer guarantee. |
| [DPPO, 2024/2025](https://arxiv.org/html/2409.00588v1), §6, App. D.7 | PPO on an environment/denoising two-level MDP; structured exploration preserves useful action support. D/A randomization plus robust action generation. | No real fine-tuning or co-training for the hardware transfer experiment; AprilTags/calibration for perception. | **16/20** real One-leg successes after simulated fine-tuning; Gaussian comparison 0/20 despite 88% sim success. Hardware policy uses estimated object state, not an end-to-end raw point-cloud policy. A particularly relevant counterexample to “higher sim success is sufficient.” |
| [ReinFlow, 2025](https://reinflow.github.io/) | Inject learnable noise into the flow integration chain, exposing tractable probabilities for online policy gradients. | Offline pretraining data and online simulated interaction in the evaluated suites. | Gym, Kitchen, Robomimic benchmarks. Hardware transfer of gains is not established by the inspected source. An optimizer, not a dynamics calibration mechanism. |
| [DSRL, 2025](https://arxiv.org/html/2506.15799v1) | Learn the noise/latent input distribution of a frozen diffusion/flow policy using an action critic and latent actor. Preserves the pretrained generator. | Real interaction: some adaptations use fewer than 50 episodes; the generalist experiments use 100–150 episodes. | Real improvements are demonstrated, including 20% to 90% in one setting. Those gains are learned with **real** feedback; they do not establish sim-only gain transfer. |
| [TRPO](https://proceedings.mlr.press/v37/schulman15.html); [KL pathologies, 2022](https://arxiv.org/abs/2212.13936) | Penalize/constrain change from a reference policy. A policy change bound controls optimization/distribution drift in the assumed MDP. | No real data is intrinsically required by the objective. | A KL bound does not bound the error between cloth simulators and hardware; an inaccurate reference can also limit learning. Anchoring is a baseline, not our novelty. |
| [RLinf-Co, 2026](https://arxiv.org/html/2602.12628v1), §IV, Table IV | Sim/real supervised initialization, followed by simulated RL plus a supervised loss on real data. Anchors visual/action capabilities while sim supplies interaction. | 20–50 real demonstrations per evaluated task. | Real tabletop manipulation with OpenVLA and a flow VLA. Already covers “sim RL with real BC anchoring.” No cloth contact evidence. |
| [RialTo, 2024](https://real-to-sim-to-real.github.io/RialTo/) | Scan a real scene; transfer real demonstrations into a digital twin using inverse distillation; improve with simulated RL. D/O/A alignment and robustification. | Real scene scan and task demonstrations. | Real manipulation robustification. Building and validating a deformable digital twin remains additional work, not an automatic consequence of this pipeline. |
| [Real-is-Sim, 2025](https://real-is-sim.github.io/) | Continuously synchronize a dynamic digital twin; policy acts on simulator representations and the physical robot follows simulated joints. O/A mediation. | Real observations during operation and synchronized setup. | Physical PushT demonstration. An always-in-the-loop simulator differs from exporting a post-trained dressing policy. No demonstrated cloth twin in this source. |
| [ASAP, 2025](https://arxiv.org/html/2502.01143v1) | Fit a delta-action model from target data; insert it into the source simulator and fine-tune the policy against the aligned dynamics. D/A. | Target motion tracking rollouts and locomotion data. | IsaacGym→IsaacSim/Genesis and Unitree G1 hardware; better tracking than tested SysID/DR/delta-dynamics baselines. Does not resolve garment perception or establish that cloth mismatch is action-equivalent. |
| [TRANSIC, 2024](https://arxiv.org/html/2405.10315v2) | Learn a residual policy and gate from human online corrections; regularize a transferred visual policy. Joint O/D/A errors. | 20/100/90/17 corrected real trajectories across four tasks, containing 62/434/489/58 corrections. | Contact-rich furniture skills, aggregate 81% in the reported ablation table. Shows the value of targeted real corrections, rather than zero-target-data transfer. |
| [DFP, 2026](https://arxiv.org/html/2605.07727v1); [Drifting, 2026](https://arxiv.org/html/2602.04770v2) | Move a one-pass generator toward data and critic-selected actions using a stopped-gradient drifting target; original drifting also supports feature-space transport. | DFP's evaluated offline/online simulator data. | DFP reports simulated RL, not real transfer. Its critic can favor simulator-specific actions. One network evaluation is not equivalent to critic-free deployment when best-of-N selection is used. Merely replacing flow matching with drifting does not address this study's gap. |
| [World and behavior grounding, 2026](https://arxiv.org/html/2610.00821), §4 | Independently ground the simulated world and simulated demonstration behavior; co-train flow/foundation policies. | 100 real demonstrations plus about 1,500 simulated trajectories per configuration; calibration/System ID also used. | Dynamic pick-and-sort: real-only 52% versus grounded co-training 86%; 50 real trials per policy. World grounding contributes a larger average effect than behavior grounding in this setting. Gains do not establish target-data-free transfer or which cloth coefficient matters. |

**Correction to earlier framing:** the brief points to §14 of the October 3
report for DFP. That section concerns **Discrete Forcing**, a different paper.
The DFP/drifting comparison above is the relevant record for this study.

## 2. Cloth and dressing: direct evidence versus suspected causes

| Primary source | How transfer/deployment was obtained | What the evidence does and does not isolate |
|---|---|---|
| [One Policy to Dress Them All, RSS 2023](https://www.roboticsproceedings.org/rss19/p008.pdf) | Sim-trained RL with partial, segmented point clouds and policy distillation; static arm observed before dressing. Real study: 510 trials, 17 participants. | Shows that an appropriate geometry interface can transfer. Does not rank friction, Young's modulus, thickness, bending and voxel errors by an independent controlled ablation. |
| [FMVP, 2025](https://arxiv.org/html/2509.12741v1), §5 and App. A | Sim visual pretraining; **192 real trajectories** with visual/force feedback and learned preference reward; offline real fine-tuning with force modulation. D435i plus Sawyer force sensing. | Explicitly identifies occlusion/segmentation, unseen arm motion, and inaccurate simulated cloth force as limitations. This is evidence for those problems, not an isolated numerical ranking of each material parameter. Real improvements require real data. |
| [Dressing in Motion, 2026](https://arxiv.org/html/2609.04759v1), §V-B | Static-demonstration diffusion policy plus arm-region registration/trajectory adaptation. **210 real expert demonstrations**, separately trained from the 180-demo simulation policy. | **Explicitly says there is no sim-to-real transfer.** Its hardware success cannot validate our simulated post-training. Motion registration and occlusion remain relevant baselines. |
| [Garment Diffusion Models, RA-L 2025](https://spiral.imperial.ac.uk/bitstreams/0da51527-b0f6-4c04-a2c5-0203c612982e/download), §IV | Learn a garment-opening diffusion dynamics model in simulation; use sampled MPC with iterative data aggregation. Partial point clouds; two grasp points; static manikin. | Real zero-shot tests, 45 trials per baseline. The paper reports similar performance **without domain randomization**, while the dynamics-model choice matters. This argues against assuming that more DR is automatically the missing contribution. No per-parameter dominance ranking. |
| [Cloth Funnels, 2022/2023](https://clothfunnels.cs.columbia.edu/) | Sim RL for canonical alignment, with geometry-based actions and downstream keypoint-conditioned manipulation. | Real garment manipulation, but flattening/alignment and subsequent folding are different contact regimes from sleeve entry. Cannot import its robustness conclusion as a dressing result. |
| [Foldsformer, 2023](https://arxiv.org/abs/2301.03003) | Goal-conditioned sequential cloth manipulation with depth-based spatial/temporal representation. | Reports zero-shot transfer without extra training/DR. Useful evidence that representation/action abstractions can matter; not a study of post-training gain retention. |
| [Benchmarking the Sim-to-Real Gap in Cloth Manipulation, 2024](https://arxiv.org/html/2310.09543v1) | Replay measured real cloth motions; tune multiple engines and compare geometric errors, separating free motion from contact. | Shows residual error and simulator/frequency tradeoffs despite tuning. The contact phase and collision handling must be tested separately. Does not prove that IPC or any one material parameter dominates our policy gap. |

**Finding:** no inspected source justifies asserting “friction is the dominant
gap” (or stiffness, thickness, noise, or voxel size) for our current interface.
Those are candidate axes to measure. Force repeatability in our IPC study is not
the same quantity as real force accuracy. Geometric repeatability is likewise not
evidence of geometric accuracy against hardware.

## 3. Prior-art attack on robust post-training

| Existing idea | Primary source | Consequence for a proposed contribution |
|---|---|---|
| Train on the worst-return portion of a physics ensemble; adapt its distribution with target data | [EPOpt, 2017](https://arxiv.org/abs/1610.01283) | CVaR/ensemble training, including adaptation, is established. |
| Improve relative to a baseline under model uncertainty; fall back when improvement is unsupported | [Robust baseline regret, NeurIPS 2016](https://arxiv.org/abs/1607.03842) | Even **relative** robust improvement is established: maximize the worst-case difference, not merely absolute return. Adding a base-policy comparison is insufficient novelty. |
| Expand randomization ranges automatically | [ADR / Rubik's Cube, 2019](https://arxiv.org/abs/1910.07113) | A curriculum of wider material/noise ranges is an implementation baseline. |
| Fit parameter distributions to limited offline target trajectories | [DROPO, 2022](https://arxiv.org/abs/2201.08434) | “Use a few real trajectories to calibrate the ensemble” is already covered. |
| Infer a posterior over simulator parameters, rather than one best fit | [BayesSim, 2019](https://arxiv.org/abs/1906.01728) | Parameter uncertainty and posterior DR are established. |
| Update the randomization distribution by matching simulated and real behavior | [SimOpt, 2018/2019](https://arxiv.org/abs/1810.05687) | Alternating real measurements, system identification and retraining is established. |
| Optimize a pessimistic dynamics model consistent with data | [RAMBO-RL, 2022](https://arxiv.org/abs/2204.12581) | Model pessimism is established and can become overly conservative; need a budget-matched robust baseline. |
| Fit models for decision/value accuracy instead of reconstructing every state coordinate | [VAML, 2017](https://proceedings.mlr.press/v54/farahmand17a.html); [calibrated VAML, ICML 2025](https://proceedings.mlr.press/v267/voelcker25a.html) | Task-aware calibration is also established. A contact-feature loss or value-aware model is not independently new. |
| Validate that simulation preserves real policy rankings and behavioral sensitivities | [SIMPLER, CoRL 2024](https://arxiv.org/html/2405.05941v1) | Predictive simulation is an empirical claim requiring paired real tests; simulator fidelity alone does not establish it. |
| Actively probe task-relevant parameters | [Task-oriented exploration, RSS 2020](https://arxiv.org/html/2006.01952v1); [SPI-Active, 2025](https://arxiv.org/abs/2505.14266) | Choosing informative real experiments, including task-aware probing, is established. |
| Learn simulation bias and query simulation/robot for high-confidence policy improvements | [S-HCI-GIBO, 2024/2025](https://arxiv.org/html/2411.14246v1) | Particularly strong prior art: dual-source Gaussian processes and gradient-information acquisition already do this for black-box policy parameters. Its smoothness/GP assumptions do not directly describe contact-event labels, but simply removing those assumptions is not a novelty proof. |
| Calibrate residual simulator error and perceived-environment uncertainty, then adapt policies | [Neural Fidelity Calibration, 2025](https://arxiv.org/html/2504.08604v1) | A learned residual fidelity distribution plus selective adaptation is already covered, with real navigation evidence. |
| Preserve decision-critical action rankings through simulator calibration and grouped perturbations | [Sim2Act, 2026](https://arxiv.org/html/2603.09053v1) | Explicitly overlaps even with “calibrate rankings, not average state error.” Its experiments concern supply chains, but that application difference does not give us algorithmic novelty. |

For a target-model set M, the familiar robust improvement objective is

`max_pi min_{m in M} [J_m(pi) - J_m(pi_base)]`,

possibly with a KL constraint. An ensemble acceptance test approximates this
objective. **We will not claim that expression as a new algorithm.** Also, if the
real system is outside M, a positive worst-case simulated gain provides no real
guarantee. Observation/controller mismatch belongs in M alongside cloth physics.

## 4. Question-4 protocol and status

The latest owner constraint supersedes the brief's older memory limit: **one IPC
worker, total GPU use below 90 GiB**. At inspection total use was 91,723 MiB
(89.57 GiB); unrelated jobs account for that usage. No dressing simulation was
running. A new worker needs admission headroom; do not terminate other workloads.

Measurement implementation and a fixed pilot manifest follow this literature map.
Use a contemporary nominal control, the frozen `fmvp_sim`, `r1`, and `flow_r1_h16`
checkpoints, identical development cases and initial placements. Bodies after the
first fourteen remain sealed. Distinguish task failure (including loss of grasp)
from infrastructure termination. Do not filter out failed episodes because their
`sim_error` describes grasp tracking failure.

Report success counts, paired win/loss counts and exact McNemar p-values. For
perturbation z, estimate both the gain
`Delta_pi(z) = mean[Y_pi(z) - Y_fmvp(z)]`
and its change from nominal `Delta_pi(z) - Delta_pi(nominal)`; a nonsignificant
p-value is not proof that the gain disappeared. Repeated units are not independent
new bodies. A fixed pilot can reveal sensitivity but cannot certify real transfer.

Two code-level qualifications discovered before measurement:

1. The existing flow checkpoint consumes simulated gripper force; fmvp_sim and
   r1 receive zero force. Preserve the deployed checkpoint interfaces for Q4 and
   state this confound. A later vision-only method comparison must address it.
2. Density/thickness alter both areal cloth mass and the mass-scaled grasp penalty.
   Their intervention is on this coupled implemented system, not exclusively on
   free-cloth dynamics. The original physical success criterion stays unchanged.

Shrinking gain under a selected perturbation establishes **fragility to that
shift**. It alone does not prove simulator exploitation, identify the real-world
error distribution, or establish the cause of a hardware failure.

### 4.1 Fixed first screen (prepared before any new rollout)

Executable: `scripts/wang_transfer/run_sim2real_audit.py --run`.
Manifest and live status: `output/uipc_manip/sim2real_audit_20261005/`.

- Seven first development bodies: 1032, 1041, 2034, 2035, 3041, 3047, 4038.
- Original primary garment `tshirt_26`; all three frozen policies.
- **210 attempts, 30 sequential batches**, one world of seven slots at a time.
  This is a single-garment screen, not the entire 70-unit development benchmark.
- Nominal and one identical nominal repeat; eight one-factor interventions below.
  Other parameters, hang, body sizing, placement offset `[0,5,0]` mm, horizon and
  success criterion are fixed. No adaptive choice of a better start.
- Separate CPU inference clients per slot prevent another body's termination
  from changing the flow random-number stream. The contemporary nominal control
  uses the same clients; historical aggregate scores are context, not controls.
- The bridge voxel is **62.5 mm**, not 6.25 mm. The voxel experiment shifts its
  grid origin by half a cell; it does not translate the point cloud.

| Intervention | Nominal → perturbed | Interpretation |
|---|---|---|
| Friction | 0.3 → 0.6 | Global IPC contact-table coefficient; includes cloth/body contact, not an isolated body-only coefficient. |
| Young's modulus | 6,000 → 12,000 Pa | Membrane material change, other coefficients fixed. |
| Bending coefficient | 0.1 → 0.2 | Discrete-shell bending coefficient. |
| Density | 750 → 1,125 kg/m³ | Coupled mass/grasp change noted above. |
| Half-thickness/contact radius | 0.15 → 0.225 mm | Coupled contact/mass/grasp change; not a solver-model replacement. |
| Point noise | 0 → 3 mm SD per coordinate | Independent Gaussian camera-point noise, not a calibrated D435i noise model. |
| Dropout | 0 → 30% | Independent point removal, not a full structured occlusion model. |
| Voxel origin | `[0,0,0]` → `[31.25,31.25,31.25]` mm | Same cell width and first-point selection; tool point always retained. |

The material values are stress levels, not estimates of the real garments.
No downward parameter sweep or unseen garment is included in this first screen.
The runner requires total usage below 84 GiB to reserve 6 GiB for startup; it
terminates only its own process group if total usage reaches 89.75 GiB. Sampling
cannot prevent instantaneous allocations by unrelated jobs. Admission and
termination events, hashes, worker cost, missing cases and initialization hashes
are recorded. No infrastructure-censored run is converted to a task failure.
The worker-time cap is 24 hours; historical median batch cost suggests about
18 hours, excluding resource waiting. The estimate is not a completion promise.

Validation: three CPU integrity tests pass (protected observation fields,
zero-shift equivalence to the actual bridge, and paired statistics including
grasp failures). Collector CLI parsing and Python compilation pass.

### 4.2 Completed CPU diagnostic

While the GPU admission condition was unmet, replayed **49 preserved r1 states**
(seven fixed times on each of the seven bodies) through all three policies on
CPU. Nominal and perturbed flow calls use identical sampled noise. The following
are **median / 90th-percentile changes in raw translation proposal, in mm**:

| Camera perturbation | fmvp_sim | r1 | flow |
|---|---:|---:|---:|
| 3 mm coordinate noise | 0.72 / 3.63 | 0.51 / 3.30 | 0.53 / 3.29 |
| 30% dropout | 1.22 / 4.53 | 1.18 / 3.44 | 1.19 / 3.39 |
| Half-cell voxel-origin shift | 1.31 / 3.34 | 1.04 / 3.20 | 1.05 / 3.19 |

This costs 7.86 CPU wall-seconds for inference/analysis after loading preserved
states, and zero new simulator interaction. In this sample r1/flow are not more
action-sensitive than fmvp_sim. It **does not** show retention of success gains:
states come from r1 trajectories, responses can compound in closed loop, and
within-body states are correlated. Do not turn these 49 states into 49 independent
task trials. Reproduce with `audit_observation_sensitivity.py` in the `curl` env;
raw output is `observation_sensitivity.json` beside the protocol.
The committed [evidence snapshot](2026-10-05-sim2real-posttraining-evidence.json)
records hashes, intervention settings, and these completed CPU results.

### 4.3 All physical attempts finished; primary pairing failure disclosed

Supervisor 2288111 finished at **2026-10-05 02:06 UTC / 04:06 Berlin**. There are
30 completed batches and 210 recorded outcomes; no automatic actor training
follows. Charged cost: **12,389.26 worker-seconds = 3.44 hours**. Recorded peak total
GPU usage: **90,276 MiB = 88.16 GiB**, below the owner limit.

The original analysis required identical post-reset cloth/TCP/landmark byte
hashes. Every body's hash differed across runs, so **zero cases pass that frozen
primary endpoint**. This was an overly strict/inappropriate initial-state check:
each world performs 30 settling substeps and two hold substeps before state zero.
Floating-point nondeterminism changes vertices even with the same controls;
material interventions also legitimately change the settled drape. Do not call
the zero admitted cases zero task successes, or silently discard the check.

`analyze_sim2real_completed.py` adds a **secondary, post hoc matched-reset block
analysis**, preserving `summary.json` and its primary rejection. It verifies
exact equality of static human/arm geometry, garment faces/hang, tool position,
grasp indices/initial offsets, landmarks, placement and seed. Every environment
configuration must equal nominal plus its predeclared material intervention.
All 210 cases pass those checks. No tolerance was selected using success outcomes.
The largest post-settle vertex discrepancy among policies within a condition is
0.759 mm; nominal is 0.220 mm. Material-dependent initial drape is part of this
secondary intervention; it is not held fixed by this design.

| Condition | fmvp_sim | r1 | flow |
|---|---:|---:|---:|
| Nominal | 3/7 | 3/7 | 4/7 |
| Friction ×2 | 4/7 | 3/7 | 4/7 |
| Young's modulus ×2 | 4/7 | 4/7 | 4/7 |
| Bending ×2 | 4/7 | 1/7 | 2/7 |
| Density ×1.5 | 5/7 | 3/7 | 3/7 |
| Half-thickness ×1.5 | 5/7 | 5/7 | 5/7 |
| Point noise 3 mm | 5/7 | 3/7 | 3/7 |
| Dropout 30% | 3/7 | 1/7 | 1/7 |
| Half-cell voxel origin | 5/7 | 4/7 | 4/7 |
| Identical nominal repeat | 4/7 | 3/7 | 3/7 |

The [generated full table](2026-10-05-sim2real-robustness-results.md) includes
block win/loss counts, exact McNemar p-values, gain-change bootstrap intervals,
repeat agreement and initialization diagnostics. None of the within-condition
exact tests is below 0.25. A coarse negative bootstrap interval is not a
multiple-comparison discovery; its sampling assumptions and small sample matter.
Nominal-repeat agreement is **4/7 base, 7/7 r1, 6/7 flow**.

**Interpretation:** checkpoint ordering varies in this pilot, with bending and
dropout unfavorable to r1/flow. However nominal r1 has no gain and nominal flow's
one-case edge disappears in the nominal repeat. This screen cannot establish
that the historical sim gain either survives or collapses. Do not generalize
the seven-body counts to the 201-unit benchmark. Do not claim a dominant real
physics gap or hardware transfer. The initial protocol's exact-state endpoint
failed, and the secondary result is a sensitivity warning, not a causal verdict.

## 5. One concrete candidate: post-training with calibrated comparison labels

**Updated status: deprioritized candidate; its small calibration component was
tested without new IPC and gave no added success.** It remains a falsifiable
proposal, not an established new algorithm. The measured failure mechanism is
not yet known. In particular the CPU diagnostic
does not support claiming that r1's encoder became more noise-sensitive. Proceed
with this candidate only if Q4 or a small target-domain check reveals unreliable
improvement ordering beyond repeatability noise. If all selected simulated gains
persist, first test that ordering on hardware instead of inventing a robustness
defect.

### Mechanism and proposed distinction

Keep the pretrained policy as an explicit fallback. IPC supplies alternative
entry action chunks; a small amount of target data calibrates **whether each
alternative really improves on that fallback**. Train the actor from supported
comparisons; preserve the base output on inconclusive comparisons.

The proposed technical unit is a **three-outcome comparison law** (+1 improvement,
0 tie, -1 regression), conditioned on deployable observation history. Both failure
and both success are ties, not discarded examples or an arbitrary winner. Estimate
it using repeated paired complete rollouts, including the continuation after the
entry chunk. This addresses finite-margin misranking without treating a noisy
one-step progress score or inaccurate absolute force as an action oracle.

This has two concrete differences from the closest inspected implementations:

1. Calibrate the discrete **relative outcome distribution** directly from target
   comparisons; do not fit next-state errors as Sim2Act does, or a differentiable
   global return surface over controller parameters as S-HCI-GIBO does.
2. Carry both ties and replica uncertainty into the distillation weights. Test
   whether doing so saves target trials near an irreversible entry boundary,
   rather than merely making the policy more conservative.

Neither difference alone is a novelty claim. Their value must be demonstrated
against calibrated score models and ordinary robust improvement under matched
data/query budgets. In particular, a reward-preference model or Bayesian ranking
model could be an equivalent implementation. This is the strongest objection,
not something resolved by giving the method a new name.

### Minimal formulation

Let h be causal point-cloud/tool/action history, m a model of physics and sensing,
and pi_0 the frozen fallback. Let u be a candidate 16-decision entry chunk from the
existing flow policy; after it, use the same pi_0 continuation. All evaluation
endpoints are full task outcomes. This is an initial data-generation choice, not
a requirement to execute 16 steps open-loop on hardware.

For each independently reset paired replicate r:

`d_{m,r}(h,u) = Y_{m,r}(u then pi_0) - Y_{m,r}(pi_0) in {-1,0,+1}`.

Starting states and perturbations are matched. Random seeds do not make IPC
bitwise deterministic; the repeated pair distribution, not one pair, is the
quantity estimated. For real experiments use randomized execution order and
matched reset blocks; never claim identical hidden cloth state from similar
point clouds alone.

With modest pseudocount smoothing, aggregate a simulated prior p_sim(d | h,u).
Fit a low-capacity, regularized multinomial correction from target comparisons:

`p_psi(d | h,u) = softmax_d[log p_sim(d | h,u) + b_psi,d(z(h,u))]`,

`L_cal = -sum_real log p_psi(d_real | h,u) + lambda_cal ||psi||^2`.

Initially z should be a small, fixed feature vector (simulated three-way
probabilities, between-model disagreement, and observable recent garment motion),
not a new large point-cloud network trained on a few dozen trials. Keep reset
blocks/garments separated in fitting and validation. A lower confidence estimate
L(h,u) for `p_psi(+1) - p_psi(-1)` determines positive imitation weights:

`w(h,u) = max(0, L(h,u))`;

`L_actor = E[w L_flow(h,u)] + lambda_keep E[L_keep(pi_theta(h), pi_0(h))]`.

Use the existing flow-matching loss and an explicit reference-output retention
loss; no IPC gradients or accurate force targets are assumed. Confidence is
empirical and must be checked on held-out reset blocks; it is not a theorem for
unseen real garments. Deployed pi_theta gets observations only. Distillation
error can destroy a teacher advantage, so evaluate the actual resulting actor.

For expensive target-query selection, prioritize comparisons where plausible
models disagree on the **sign of improvement**, weighted by their expected
effect on the proposed update. Compare this optional acquisition rule with
uniform queries; task-oriented active probing itself is established prior art.

### Falsifiable prediction and smallest decisive experiment

**Prediction:** at the same target-comparison budget, the calibrated comparison
method selects fewer harmful entry updates than uniform DR and a calibrated
absolute-score model, while retaining at least as much positive gain. In a pilot,
predeclare a meaningful effect as halving the harmful-update fraction and improving
target success by at least 10 percentage points; report uncertainty rather than
treating these thresholds as a significance test. A method that succeeds only by
rejecting almost every change has failed the prediction.

**Cheapest rejection screen: no new IPC calls.** Once Q4 finishes, treat each
perturbed condition in turn as a hidden synthetic target. Reveal the first three
bodies' two base-relative comparisons for calibration (six paired labels), then
predict harmful versus helpful updates on the remaining four bodies. Compare
the ternary calibrator with uncalibrated ensemble selection, a simple scalar
gain calibrator, and ordinary robust-baseline-regret selection. Reuse only the
existing three checkpoints and Q4 outputs. This costs CPU minutes and can reject
the calibration premise. It cannot establish a new trained policy, and the small
number of independent bodies means a weak or noisy result remains inconclusive.

**First physical test of candidate chunks, if the free screen survives:** use 8 development entry
cases, pi_0 and two fixed flow-generated candidate chunks, 3 source physics models,
and 2 independent replicas. This is **144 full continuations** (8×3×3×2). Use a
fourth, predeclared held-out model (a new joint shift, not a Q4 condition selected
after seeing its results) as a synthetic target: reveal at most **12 paired comparisons
(24 rollouts)** to calibrate; score the frozen selection rule on held-out cases
with **36 more rollouts** (6×3×2). Cases, source/target models and chunks are frozen
before viewing target outcomes. Count all prefix replay/setup/retries. Do not
reset a deformed cloth into another material and pretend its history is valid;
replay the physical prefix from the common initial state.

Compare: uncalibrated nominal selection; uniform-domain robust baseline-regret
selection; DROPO-style parameter calibration; a regularized absolute-outcome
calibrator using exactly the same target data; the proposed ternary calibrator;
direct regression on signed paired differences with uncertainty; and the
tie-discarding ablation. All share the candidate bank, rollouts and target
budget. Selection regret and harmful-update frequency use unrevealed target
outcomes. Use block-level resampling; do not count action candidates or replicas
as independent bodies. The 12 calibration pairs are too few for a strong asymptotic
claim—this test is intended to reject an unpromising mechanism cheaply.

**Cost ceiling:** 204 complete simulated attempts, at most 157,080 decisions if
all reach the existing 770-decision maximum, plus explicitly charged prefix
replays if starting from intermediate states. The current seven-body historical
batch rate suggests roughly **12–24 single-worker hours**, to be re-estimated
from Q4 before execution. Calibration itself is CPU minutes. This is a proposed
experiment, **not launched by this study**. Existing Q4 trajectories may reduce
the nominal-control cost, but cannot substitute for new candidate continuations.

If this test survives, first verify the comparison mechanism on a static
manikin using **12 randomized matched pairs with two repetitions = 48 real
attempts**, then test a trained actor on a separate garment/body set. At 2–4
minutes per attempt including reset, the comparison pilot alone is roughly
1.6–3.2 robot-hours; actor evaluation costs extra. No amount of simulated
calibration establishes transfer without this physical check.

**Failure criteria:** stop this candidate if the base-relative sign is already
stable, if target comparisons are mostly indistinguishable ties, if a simple
absolute-score calibrator is as good, if the tie/uncertainty terms do not help,
or if corrected labels fail to improve the deployed student. If the only winning
component is ordinary DR or real BC anchoring, report that finding and use the
known method; do not advertise a new post-training algorithm.

### What would count as a contribution

The defensible prospective contribution is a demonstrated reduction in **real
data required to obtain a transferable policy update**, using a specific
comparison estimator and its handling of ties/model disagreement. The task,
IPC engine, pretrained checkpoint, flow parameterization, residual formulation,
KL anchor, and worst-case ensemble check are not independently new. Broadly
describing the proposal as “rank-aware sim-to-real post-training” would overlap
with existing work. The narrow claim remains unproven pending the experiment.

### 5.1 Completed free rejection screen

`check_comparison_calibration.py` used the disclosed matched-reset outcomes.
For each of eight stress targets it fit six target comparison labels from the
first three bodies, then chose base/r1/flow on the remaining four. Constants and
methods were fixed before running this screen. These 32 decisions reuse four
independent bodies; they are not 32 independent targets.

| Selection rule | Actual target successes / 32 | Harmful updates | Selected updates |
|---|---:|---:|---:|
| Keep base | 20 | 0 | 0 |
| Nominal or ensemble-mean selection | 19 | 1 | 8 |
| Worst-model gain | 20 | 0 | 0 |
| Scalar bias or ridge calibration | 19 | 1 | 11 |
| Three-outcome logit correction | 19 | 1 | 13 |

There is **no positive oracle opportunity** among the two alternatives on these
held-out cases: neither repairs a base failure. The ternary estimator's smaller
harmful fraction comes from choosing more ties, not fewer harmful outcomes.
Therefore this screen provides no reason to scale that estimator. It cannot
reject the full future chunk-calibration method when successful alternatives
exist; that situation is absent here. Costs: CPU seconds, zero new IPC calls.

The novelty objection is also stronger after the follow-up search:
[Reward Learning From Preference With Ties](https://arxiv.org/html/2410.05328)
already models tie-related preference bias. Treat tie handling as a control,
not as a new post-training principle.

## 6. Second candidate and current research decision

The [complete observable feedback-repair proposal](2026-10-05-feedback-repair-posttraining.md)
specifies a different supervision object: a robust distribution over complete
feedback corrections, with causal observation routing and repair identity retained
through prefix/continuation. Its mathematical operator, actor losses, objections,
failure criteria and proposed **112-attempt / 2–4-worker-hour** mechanism pilot
are concrete. A CPU compiler and three analytic checks are implemented. No new
physical pilot or student training has been launched.

This is a hypothesis about efficient supervision/absorption, not an assertion
that trees, mixtures, belief-aware teachers or shared-prefix diffusion are new.
A2D, BIG, LCEOPT, CoPlanner, SDP and ACPPO-Corr are required comparisons. If ordinary
history-flow distillation preserves the same gain, the proposed association
mechanism has no demonstrated method value. A real paired comparison remains
necessary before claiming sim-to-real improvement.

The requested prior-art map, bounded measurement, failure disclosure and concrete
candidate specifications are delivered. No validated novel algorithm or transferable
actor exists from this study. The next scientific step is a finite experiment on
**new executable repairs**, not more data from the unchanged weak labeling bank.
