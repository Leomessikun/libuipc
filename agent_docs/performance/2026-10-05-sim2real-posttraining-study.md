# Post-training whose gains survive transfer — study opened 2026-10-05

Request: [the brief at 67e0c5fa](2026-10-05-sim2real-posttraining-research-brief.md).
Questions 1–3 were investigated before launching question 4. This record separates
published evidence, our measurements, and proposed methods. No new policy has been
trained for this study. The cancelled M4/composite-teacher queues remain stopped.

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

## 5. Method candidates

Pending the Q4 measurement. The target is a transferable improvement signal,
with uniform DR, robust-baseline-regret filtering, and real-data anchoring as
explicit alternatives. Newness must lie in a specific estimator, experiment
selection rule, or update mechanism that beats those alternatives at equal cost.
No new algorithm is declared successful or novel in this preliminary record.
