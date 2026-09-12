# Force-aware reinforcement learning beyond dressing

> **Status update, 2026-09-13:** This is a historical audit/proposal against `4bfe88c2`, not the current implementation status. Subsequent commit `024c5716` closed the proposed per-decision force-training direction after the reliability gate failed; see the [closure](2026-09-12-research-direction.md) and [measurements](2026-09-12-contact-force-calibration.md). No force-training result is established here. The dense action-per-point critic was subsequently implemented in `ef1b3c81`, with optional residual trunks in `01bf913e`. Any future experiment must use those changes as an explicit baseline and revalidate its labels. Literature findings remain reference material; implementation recommendations below are conditional.

The most practical use of a simulator-only force field is to improve training while keeping the actor's deployment inputs unchanged. The literature supports three distinct routes: privileged critic information, supervised prediction of hidden interaction state, and force objectives or constraints. Feeding exact simulated force directly to the actor is a different deployment contract. It requires either an equivalent real sensor or a separately validated estimator.

Six concrete methods below provide useful precedents. Results demonstrated on rigid manipulation, insertion or legged manipulation do not establish performance on cloth or human skin. Recommendations for IPC dressing are identified separately from reported evidence.

## 1. Asymmetric actor–critic: privilege the critic, preserve deployment inputs

Pinto et al., *Asymmetric Actor Critic for Image-Based Robot Learning* (RSS2018), train an image-input actor with a full-state critic. The actor receives RGBD observations; the critic receives simulator state. Combined with domain randomization, the method transfers picking, pushing and block-moving tasks to a real robot without training on real data. This paper is a precedent for asymmetric information access, **not a demonstration of distributed force-aware dressing**.^1

**Deployment requirement:** the actor's visual observations and robot control interface; simulator state is training-only. A force field can be added to the critic without requiring its measurement at deployment, provided it is not also passed to the actor. That extension is a proposal, not a result established by Pinto.

**IPC adaptation:** use a small vector of calibrated regional force statistics in the critic alongside observable state and action. Compare with a critic receiving the same non-force privileged geometry to determine whether gains are due to force information itself. A critic cannot eliminate ambiguity when identical actor observations correspond to different contact loads.

## 2. Concurrent force estimation and force-conditioned control without an F/T sensor

Portela et al., *Learning Force Control for Legged Manipulation* (ICRA2024), provide a direct force-privilege example. The actor observes histories of projected gravity, gait clock, joint positions/velocities, previous actions and task commands. A supervised estimator predicts privileged body velocity, gripper position and external gripper force; the actor uses predicted quantities, while the critic receives true privileged quantities. History length is30 and deployment runs at50Hz. Actions are17joint-position targets for PD controllers.^2

Training includes force tracking against a randomized spring-damper external field. The deployed UnitreeB1+Z1 does not use an F/T sensor for control; a dynamometer measures evaluation performance. Reported real force-tracking error is approximately5–10N over commanded downward forces up to70N. The authors observe force-estimation overshoot and suggest the policy partly disregards the estimate.^2

**Limitation and adaptation:** this supports learning hidden load from proprioceptive history, but its error scale is unsuitable as evidence for low-force human interaction. Predicting a distributed cloth–arm force map from point clouds is substantially harder than predicting one gripper force. Start with regional load/contact-event targets and explicit held-out calibration.

## 3. Multimodal self-supervision: use force to learn representations and predict contact

Lee et al., *Making Sense of Vision and Touch* (ICRA2019; expanded T-RO2020), fuse RGB images, wrist6-axis force/torque histories and end-effector position/velocity. Their self-supervised objectives include action-conditioned optical-flow prediction, future contact prediction, and whether visual and haptic streams are temporally aligned. The compact representation improves policy learning for peg insertion, evaluated in simulation and on a real KukaIIWA.^3

**Deployment requirement:** this demonstrated method continues to consume force/torque readings. It is not evidence that force can simply be removed after pretraining. Their contact labels arise from force-sensor events, so correct temporal alignment matters directly.

**IPC adaptation:** an auxiliary head on the existing observable encoder could predict next-action contact onset, regional force exceedance or an aggregate force vector using simulator labels. This sensor-free auxiliary-only variant is an extrapolation. Test whether it improves held-out task performance as well as prediction error; accurate labels do not imply those loads are identifiable from a single occluded image or point cloud. History and executed actions are plausible additional inputs.

## 4. Constrained policy optimization: separate task return and force budget

Achiam et al., *Constrained Policy Optimization* (ICML2017), optimize reward subject to expected cumulative auxiliary-cost constraints. The method targets near-constraint satisfaction across policy updates under its theoretical assumptions and is evaluated on simulated robot locomotion. It does not establish a hard bound on every instantaneous real-world contact force.^4

**Deployment requirement:** costs must be available during training; the policy need not observe each cost directly at deployment. Online enforcement is a separate mechanism.

**IPC adaptation:** retain dressing progress reward and add a separate regional-load cost, for example a time-integrated threshold exceedance. A cost budget is easier to interpret than repeatedly tuning a single combined reward weight. A Lagrangian extension to existing SAC may be a smaller engineering change than implementing CPO, but it is not CPO and inherits no CPO-specific guarantees. Report violation frequency, worst-case/upper-quantile load and task success separately. A cumulative budget alone can permit brief severe peaks or sacrifice a minority of episodes.

## 5. Online admittance residual adaptation: calibrate compliance with actual force feedback

Zhang et al., *Efficient Sim-to-real Transfer of Contact-Rich Manipulation Skills with Online Admittance Residual Learning* (CoRL2023), train motion and compliance in MuJoCo using SAC. For assembly, state includes relative peg pose,6Dvelocity and6Dwrist wrench; action includes6Ddesired velocity and6stiffness values. Online deployment optimizes residual admittance parameters using recent force measurements; positive parameter constraints maintain the chosen controller structure. This online step is optimization, not merely an extra residual RL action network.^5

On real assembly the proposed method reports10/10success versus3/10direct transfer; maximum force23.6±6.3N versus63.7±6.8N. Manual tuning also achieves10/10 and lower peak force10.3±2.2N. Thus improved transfer does not mean globally minimal force.^5

**Deployment requirement:** wrist wrench sensing and a compliant control interface, plus suitable object pose/velocity information. **IPC adaptation:** learn bounded trajectory or compliance residuals over a verified low-level controller. This cannot directly replace the current cuff target interface without defining how a real gripper realizes compliance; nor does wrist wrench expose distributed opposing cloth pressures.

## 6. Contact-aware low-level control beneath RL

Zhu, Kang and Chen, *A Contact-Safe Reinforcement Learning Framework for Contact-Rich Robot Manipulation* (arXiv2022), combine image/pose-based PPO with Cartesian variable impedance actions. A momentum observer estimates unexpected robot-arm contacts from joint measurements; a contact-aware null-space controller reduces contact loads while preserving the main task. End-effector wrench enters the observer formulation to separate intended tool contact from unexpected arm contact. Evaluation transfers surface wiping to a FrankaPanda and tests disturbances and unexpected collisions.^6

**Deployment requirement:** torque-capable robot control, dynamics/kinematics and joint measurements, with end-effector force handling as specified by the observer; this is substantially more than a vision-only RL policy.

**IPC adaptation:** a low-level force monitor and bounded motion/compliance correction can complement force-aware learning. This paper is evidence for separating slower policy learning from responsive contact control, not a certified human-safety guarantee. Its robot-arm collision observer does not measure pressure or shear under cloth on another person's arm.

## Comparison of implementation and deployment contracts

| Route | Training force role | Actor deployment input | Main additional burden |
|---|---|---|---|
| Privileged critic | Additional value-estimation information | Existing visual/proprioceptive observations | Correct transition-aligned force labels and ablation against non-force privileged state |
| Concurrent estimator | Supervised hidden-state target plus privileged critic | Observation/action history and predicted state | Quantified observability and sim-to-real estimator error |
| Multimodal representation | Contact/dynamics self-supervision | Demonstrated method uses wrist F/T; auxiliary-only variant is proposed | Temporal alignment and avoiding accidental dependence on unavailable sensors |
| Constrained RL | Separate cumulative cost signal | Can preserve existing observation | Define budgets, distinguish averages from peak constraints, evaluate violations |
| Admittance residual | Train compliance, adapt from real wrench | Force feedback through policy/controller | Real F/T and an executable compliant controller |
| Contact-aware controller | Physical control feedback beneath RL | Images/pose for policy; joint/dynamics sensing for controller | Model quality, controller rate, contact localization and hardware interface |

## Implications of the verified IPC export problem

The local calibration establishes that `-gradient/dt²` has the correct model-force scale. It does not make every exported vector a fresh final-state sensor reading. With tangential loading, the default loose Newton setting accepted a single correction and exported zero friction from the preceding assembly; tighter convergence recovered the expected friction and force balance. Source inspection shows the export uses cached last-assembled gradients and regularized lagged friction. This is local evidence, not a claim from the six external papers.

That distinction affects every route. A cost built from stale zero friction rewards the wrong behavior; a critic can learn a solver artifact; an auxiliary predictor can become excellent at predicting a stale label; and an admittance controller may react to a force from the wrong configuration. Moving the same faulty signal from actor to critic does not solve data validity.

A useful label contract should include simulator integration step, frame, sampling stage, convergence indicators and unavailable-versus-zero status. Collect forces at each simulator substep when computing short peaks or integrated exposure. Report model force rather than physical pressure unless an area/traction discretization and validation have been established. Preserve contact-region identity: opposite loads can cancel in a wrist net force. These are engineering recommendations derived from the local audit.

## Recommended experiment sequence

1. **Freeze a calibrated force readout and reproducible evaluation protocol.** Check loaded friction and motion balance across relevant timestep/tolerance settings. Do not change production physics to hide bad sensing.
2. **Test a training-only force cost and a privileged force critic independently.** Keep actor inputs, initial-state distribution, reward components, timing and training budget matched. Compare task success and regional force statistics; a lower force caused only by failing to dress is not improvement.
3. **Add auxiliary prediction only when it demonstrates predictive value on held-out garments/bodies.** Use histories and actions, report conditional error and high-load false negatives, and prevent privileged labels from leaking into deployment inputs.
4. **Consider force-conditioned/compliant execution when the real robot sensing contract is defined.** Sensor-free estimates and measured wrist wrench serve different roles; neither alone certifies distributed skin loads.
5. **Use an explicit constraint formulation when a meaningful force/exposure budget exists.** Keep worst-case monitoring separate from average-cost optimization. Thresholds must follow the actual force definition and downstream application, not be copied from rigid insertion papers.

## Sources

1. Pinto, Andrychowicz, Welinder, Zaremba and Abbeel. *Asymmetric Actor Critic for Image-Based Robot Learning*. RSS2018. [Original proceedings PDF](https://www.roboticsproceedings.org/rss14/p08.pdf).
2. Portela, Margolis, Ji and Agrawal. *Learning Force Control for Legged Manipulation*. ICRA2024. [Original paper, SectionsIV–V](https://arxiv.org/html/2405.01402v1); [IEEE publication](https://ieeexplore.ieee.org/document/10611066/).
3. Lee et al. *Making Sense of Vision and Touch: Self-Supervised Learning of Multimodal Representations for Contact-Rich Tasks*. ICRA2019; expanded T-RO2020. [Original author-hosted ICRA PDF](https://ai.stanford.edu/~yukez/papers/icra2019.pdf); [author laboratory explanation of objectives](https://www.robotics.stanford.edu/blog/selfsupervised-multimodal/); [T-RO publication](https://ieeexplore.ieee.org/document/9043710/).
4. Achiam, Held, Tamar and Abbeel. *Constrained Policy Optimization*. ICML2017. [Original proceedings](https://proceedings.mlr.press/v70/achiam17a.html); [PDF](https://proceedings.mlr.press/v70/achiam17a/achiam17a.pdf).
5. Zhang, Wang, Sun, Wu, Zhu and Tomizuka. *Efficient Sim-to-real Transfer of Contact-Rich Manipulation Skills with Online Admittance Residual Learning*. CoRL2023. [Original proceedings PDF, Sections3–4 and Table1](https://proceedings.mlr.press/v229/zhang23e/zhang23e.pdf).
6. Zhu, Kang and Chen. *A Contact-Safe Reinforcement Learning Framework for Contact-Rich Robot Manipulation*. arXiv2207.13438,2022. [Original paper, SectionsIII–IV](https://arxiv.org/html/2207.13438).

## Source-quality note for integration

An initial search snippet about fingertip-force privilege pointed to an OpenReview PDF, but the PDF was not accessible and could not be reliably attributed. It is not used above. Lin et al., *Sim-to-Real Reinforcement Learning for Vision-Based Dexterous Manipulation on Humanoids* (2025), is a useful additional asymmetric-critic precedent, but its accessible v1 AppendixC lists joint/object states and physical randomization scales, not the force-privileged state from that snippet. Do not attribute force privileges to it without verifying a later version: https://arxiv.org/html/2502.20396v1 .
