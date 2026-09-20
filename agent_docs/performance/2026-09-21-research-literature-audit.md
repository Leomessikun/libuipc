# Literature audit for the dressing research decision

Checked 2026-09-21 against primary papers, proceedings and author repositories.
This completes the interrupted thematic screen; it is a bounded literature search,
not proof that no other related work exists. Recommendations and remaining gaps
are our inferences. Local measurements are in the [synthesis](2026-09-21-research-direction-synthesis.md).

## 1. Boundary estimators: the mathematics and several query mechanisms already exist

| Primary source | What is already established | Implication for this proposal |
|---|---|---|
| [Peng, Fu, Hu, Heidergott, Operations Research 2018](https://doi.org/10.1287/opre.2017.1674), [author-hosted full text](https://research.vu.nl/ws/files/104903391/A_New_Unbiased_Stochastic_Derivative_Estimator_for_Discontinuous_Sample_Performances_with_Structural_Parameters.pdf), introduction and estimator construction | GLR handles discontinuous sample performance with structural parameters; its analysis includes surface corrections to IPA/LR. The introduction explains SPA's problem-dependent conditioning and possible inversion/additional simulation. | Neither identifying missing boundary terms nor combining pathwise and score estimators is new. Opaque solver nondeterminism does not automatically supply the density and derivative assumptions these estimators require. |
| [Lee, Yu, Yang, NeurIPS 2018](https://arxiv.org/abs/1806.00176) | Reparameterized gradients combine smooth-region contributions with sampled boundary contributions. | The scalar threshold example and interior-plus-boundary decomposition are motivation, not contributions. |
| [Li et al., TOG 2018, edge sampling](https://people.csail.mit.edu/tzumao/diffrt/) | Samples visibility discontinuities alongside smooth rendering derivatives. | Boundary-aware importance sampling predates this project. Cloth rollout boundaries lack the directly represented geometric edges available here. |
| [Bangaru et al., TOG 2021, Teg](https://people.csail.mit.edu/sbangaru/projects/teg-2021/teg-2021.pdf) | Differentiates integrals with parametric discontinuities, including distributional contributions. | A general differentiable programming treatment already exists; handling an expensive implicit outcome must be the additional result. |
| [Bangaru et al., TOG 2022, neural SDF rendering](https://arxiv.org/abs/2206.05344) | Handles visibility derivatives for surfaces without the convenient explicit edge parameterization of triangle meshes. | Even “implicit boundary” is too broad a novelty claim. The distinction must involve sequential policy learning, query access and cost. |
| [Suh et al., ICML 2022](https://proceedings.mlr.press/v162/suh22b.html) | Analyzes bias/variance under stiffness and discontinuities and combines first- and zeroth-order estimates. | Discontinuity does not imply abandoning all gradients; likelihood-score policy gradients remain available. |
| [AHAC, ICML 2024](https://proceedings.mlr.press/v235/georgiev24a.html); [AGPO](https://openreview.net/pdf?id=S9DV6ZP4eE) | Adaptive rollout horizons and adaptive gradient handling address difficult differentiable dynamics. | Horizon adaptation or gradient trust weighting needs comparison with these methods. |
| [Onoda et al., 2026](https://arxiv.org/html/2604.18161v1) | DDCG and IVW-H are two methods in **one** paper. Detection helps controlled examples; variance management performs strongly in differentiable robotics tasks. | Correct the proposal's apparent count of two independent works. No-extra-rollout weighting does not remove the cost of obtaining analytic gradients in the first place. |

The local branch library does not yet measure a boundary normal, boundary density,
or one-sided return limits under coupled disturbances. Bimodality and mixed
repeats are compatible with a stochastic continuous response. A valid boundary
experiment must specify the noise variable, a continuous action/segment family,
and shrinking perturbations before claiming a jump. An ordinary likelihood-score
estimator does not need an extra boundary term: it already differentiates the
policy distribution for the chosen return. Adding a boundary term to it without a
control-variate derivation would double-count part of the gradient.

## 2. Cheap feasibility and costly performance: a particularly close prior art

[Constrained Best Arm Identification with Tests for Feasibility, Cai and
Kandasamy, AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/39063)
explicitly chooses both an arm and whether to measure performance or a feasibility
constraint. It eliminates candidates using whichever evidence is easier and gives
sample-complexity upper/lower bounds. This is a direct precedent for the proposed
“cheap feasibility, expensive return” operator, not merely generic safe RL.
Sections 1–2 of its [paper](https://ojs.aaai.org/index.php/AAAI/article/download/39063/43025)
also explain why always testing feasibility first can waste samples.

[Optimal Multi-Fidelity Best-Arm Identification](https://arxiv.org/abs/2406.03033)
(2024 submission, revised 2025) optimizes sampling cost across fidelities, with an
instance-dependent lower bound and asymptotically optimal method. A truncated
cloth rollout is not automatically an admissible low-fidelity estimate: a useful
bias relation must be justified. The existing negative early/late coverage
correlations specifically warn against assuming such a relation.

These papers cover separate tests and cost-sensitive allocation, respectively.
Neither citation alone proves that the exact dressing setup is solved. Conversely,
combining their ideas with a neural actor does not establish a new RL contribution.
A substantive extension would need to account for changing continuation policies,
prefix-dependent first violations, shared trajectory observations and execution
error after policy fitting, and show a benefit over straightforward adaptations.

| Required baseline family | Existing mechanism and boundary |
|---|---|
| [CPO, ICML 2017](https://proceedings.mlr.press/v70/achiam17a.html) | Optimizes reward under expected-cost constraints. This is not a blanket per-trajectory guarantee. A first-violation cost can encode an episode failure probability. |
| [Recovery RL, RA-L 2021](https://arxiv.org/abs/2010.15920) | Learns risk from offline data and separates task and recovery policies. Separating progress from constraint satisfaction is established. |
| [Predictive safety filter, Automatica 2021](https://arxiv.org/abs/1812.05506) | Uses predictive control and a backup policy for constraint enforcement. Short empirical survival does not satisfy its model/feasibility assumptions by itself. |
| [Provably Optimal RL under Safety Filtering, 2025 preprint](https://arxiv.org/abs/2510.18082) | Analyzes optimal learning with a valid safety filter. This does not turn a learned cloth classifier into such a filter. |
| [SPO, NeurIPS 2024](https://papers.nips.cc/paper_files/paper/2024/file/01fb6de3360f9e32862665580e2c5853-Paper-Conference.pdf) | SMC search and amortized policy improvement. Planning followed by policy fitting is already a central mechanism. |
| [P3O, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/file/3afe351e7b99b4b2f03c2ec9ce7ef2a8-Paper-Conference.pdf) | SMC-based policy optimization for continuous POMDPs. Particle trajectories plus history-conditioned policy updates are not sufficient novelty. |
| [Prism-GRPO, August 2026 preprint](https://arxiv.org/abs/2608.17423) | Explicitly targets uninformative same-outcome groups in VLA policy optimization. | 

The last paper is a new overlap to include if pursuing sparse-success group
updates. It is not evidence of an equivalent cloth feasibility algorithm. The
older weighted-ensemble/GPS/DART comparisons remain recorded in the
[clean-slate screen](2026-09-20-clean-slate-rl-research.md); this pass does not
rebrand the previously unsuccessful score-aware branching proposal.

## 3. Active contact identification: capability needs an information test

| Primary source | What it already covers | What it does not by itself establish |
|---|---|---|
| [Haptic Simulation for Robot-Assisted Dressing, IROS 2017](https://repository.gatech.edu/bitstreams/f84800ca-2ffa-4310-a6bb-5635960910c1/download), building on Kapusta et al. 2016 | HMMs classify sleeve entry, missed opening and catches from haptic sequences. | Selecting a probe because its feedback changes the optimal continuation. |
| [Deep Haptic MPC, ICRA 2018](https://arxiv.org/abs/1709.09735) | Action-conditioned recurrent force predictions and control around fist/elbow catches. | An intervention-controlled measurement of the information value of probing. |
| [Task-Oriented Active Sensing via Action Entropy Minimization, 2019](https://kuscholarworks.ku.edu/server/api/core/bitstreams/2305b1d0-d4d0-4618-bdb8-e4b4ea79daf4/content) | Task/action uncertainty as the target of sensing instead of complete state reconstruction. Author-manuscript search text checked; direct PDF fetch failed in this pass. | Novelty for merely replacing state entropy with action uncertainty is not defensible. Detailed assumptions should be checked from the PDF before reproducing it. |
| [POMDP-Guided Active Force-Based Search for Robotic Insertion](https://arxiv.org/abs/2404.03943), IROS 2023 DOI, arXiv 2024 | Contact-informed primitives, POMDP reasoning and proprioceptive search for insertion. | Clothing topology and long-horizon cloth dynamics are not the evaluated setting. |
| [FCVP, 2024](https://arxiv.org/html/2311.04390v2) | Visual dressing policy plus learned force dynamics and constrained action optimization. | Predicting risk or filtering actions is not equivalent to selecting information-gathering probes. |

This screen supports a **testable distinction**, not a “first active dressing”
claim. A useful benchmark needs hidden states that (a) overlap under the actual
observation history and (b) require different feasible continuations. The same
physical probe must be executed in feedback-enabled and feedback-masked arms;
otherwise releasing a fold is confounded with learning about it. Train the masked
control under its permitted observations instead of masking a network only at
inference and mistaking distribution shift for information value. A privileged
oracle measures whether any meaningful action disagreement exists at all.

Our t26 direction-level agreement across eight similar slots currently supports a
shared intervention. It does not establish an observation aliasing problem.
The previous successful-controller/failed-student result motivates testing history
and execution error first, but does not prove either is the cause.

## 4. Dressing capability map, including an omitted counterexample

| Work | Demonstrated setting or metric | Consequence for claims |
|---|---|---|
| [Zhang and Demiris, Science Robotics 2022](https://pubmed.ncbi.nlm.nih.gov/35385294/), [author code](https://github.com/fan6zh/robot_dressing) | Learned garment manipulation in a pipeline from a gown on a rail through unfolding and dressing a medical manikin; reported success above 90%. Active pre-grasp manipulation and simulator calibration are included. | Retract the staged synthesis's broad “no learned dressing pipeline reports full-task success” claim. This is a modular pipeline, not evidence of a single end-to-end policy, and not the same task as t26. Publisher abstract checked; publisher full text returned 403. |
| [One Policy to Dress Them All, RSS 2023](https://arxiv.org/abs/2306.12372) | Partial point clouds, varied garments/poses, RL and distillation; 86% average arm-length coverage in human trials. | Coverage is not binary task success; these architectural ingredients are occupied. |
| [Garment Diffusion Models, RA-L](https://spiral.imperial.ac.uk/entities/publication/a0176f43-5eb7-4530-93d8-35c0118514df) | Action-conditioned future garment-opening point cloud, used in MPC/model-based RL. Online publication 2024-12-16, volume 10(2), 2025. Reported 91.2% on its sleeve insertion task with fewer than 100 sampled trajectories. | Diffusion **dynamics** is distinct from an action diffusion policy or differentiable simulation. This is not whole-garment success. |
| [FMVP, CoRL 2025](https://arxiv.org/html/2509.12741v1), Sections 3 and 6 | Visual/force adaptation under arm motion; assumes the garment has already been grasped. Appendix C documents occlusion-related elbow failures. | Multimodal feedback and moving arms are occupied. Its limitations motivate a narrower hypothesis, not a field-wide impossibility claim. |
| [DiffCloth, TOG 2022](https://arxiv.org/abs/2106.05306) | Differentiable cloth with frictional contact and dressing-related optimization. | First use of cloth gradients for dressing is unavailable as a claim. |
| [Wearing A Coat, July 2026 preprint](https://arxiv.org/abs/2607.10999), [full text](https://arxiv.org/html/2607.10999v1) | Dual-arm coat assistance using differentiable clothing simulation, global control and local constrained compensation. | Dual-arm differentiable dressing is also occupied; it is not a learned-policy training-throughput benchmark. |
| [Dressing in Motion, September 2026 preprint](https://arxiv.org/html/2609.04759v1), Sections III and V | Point-cloud action diffusion plus motion adaptation; pre-insertion arm is static. Reports dressing ratio and sleeve-insertion success separately; Diff-MPC is explicitly a diffusion garment-dynamics baseline. | Motion-adaptive diffusion is occupied. Its 89% success statistic cannot be substituted for grasp-valid full-episode success here. |

“No reliable initial grasp” alone is therefore too broad a capability claim: the
2022 work already handles pre-grasp uncertainty. A narrower possible capability
is recovery of uncertain **in-progress** grasp/insertion relationships, with
measured feedback value and maintained physical validity. It still needs a wider
benchmark and hardware evidence before a novelty claim. No current simulation
label certifies human safety.

## 5. Simulator budget and calibration: verified numbers, limited comparisons

[FLASH, 2026](https://arxiv.org/html/2604.17513v1), Table II, reports 50.43 ms per
vector step at 64 environments: 0.000788 s per environment-step. Its teacher is a
state-based finite-state controller distilled into a policy. The reported towel
training time is 50 minutes on an RTX 5090; its continuous real evaluation reports
91/106 successful towel folds. This is folding, not dressing.

[SAPO/Rewarped, ICLR 2025](https://arxiv.org/html/2412.12089v2), Appendix F,
Table 18, reports 6M HandFlip steps taking PPO 1.46 h, SAC 1.67 h, APG 7.43 h,
SHAC 7.66 h and SAPO 7.77 h. These are total algorithm runtimes. They do **not**
isolate backward-pass time or establish the backward/forward ratio of libuipc.
HandFlip is not a cloth-dressing task.

Our 0.0610 s/transition includes the measured local training pipeline. Comparing
it with another simulator's raw step time mixes tasks, meshes, simulated time,
substeps, hardware and learning overhead. The ~77x arithmetic ratio to FLASH is a
budget illustration, not a controlled simulator-speed result. Delete “slower than
any published cloth-RL setup,” “only work reporting timings,” and the unsupported
0.3–0.4 s local analytic-gradient projection from the staged synthesis.

[WSRL, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/504491292cb71e7681eedfe0e602b72f-Paper-Conference.pdf)
provides a relevant warm-start fine-tuning baseline. Its central issue is replay
initialization and value recalibration, not expensive cloth constraints. Small
online budgets do not erase prior data, pretraining, evaluation or reset costs.
The staged DSRL numerical example was not independently reverified from its full
text (OpenReview challenge); it is not used for the decision here.

[A Multi-Fidelity Control Variate Approach for Policy Gradient Estimation](https://arxiv.org/abs/2503.05696)
already covers combining high- and low-fidelity data for policy-gradient variance
reduction. [IsaacIPC](https://arxiv.org/abs/2605.24339) already integrates libuipc
with a robot-learning infrastructure. Neither “use two simulators” nor “use IPC
for learning” is a sufficient contribution.

[Calibrated Value-Aware Model Learning, ICML 2025](https://proceedings.mlr.press/v267/voelcker25a.html)
shows why common value-aware surrogate losses need not recover correct model/value
solutions. This is loss calibration; it is not by itself a guarantee that real
cloth action rankings or failure probabilities are calibrated. Direction 3 needs
held-out physical interaction data and repeated action consequences. Existing
simulator-to-itself checks cannot supply that evidence.

## Search coverage and residual uncertainty

Searches covered the named papers, IPA/SPA/GLR, parametric discontinuities,
rendering, active force insertion, haptic dressing, constrained/safety RL,
separate feasibility tests, multi-fidelity BAI, SMC policy optimization, and
learned garment preparation. The 2026 arXiv items are treated as preprints unless
a proceedings record was checked. Abstract-only and failed full-text access are
identified above. This screen establishes close prior art and corrects overly
broad claims; it does not prove the remaining intersection is novel.
