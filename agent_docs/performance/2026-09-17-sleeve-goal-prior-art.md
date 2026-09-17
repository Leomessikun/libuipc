# Sleeve-goal dressing: prior-art check — 2026-09-17

Status: literature review only; no code, simulation or training. It checks the
novelty of the [sleeve-motion transfer proposal](2026-09-17-dressing-transfer-direction.md)
after the [sleeve-path and reward audit](2026-09-17-sleeve-path-audit.md).
Two web surveys ran in parallel (object-motion interfaces across domains;
robot-assisted dressing 2020–2026). "Read" below says who read which source:
*verified* means this session fetched the primary source itself; *full text*
or *abstract* means the survey agent read it and this session did not re-check.

## The closest prior art: sleeve-state goals for dressing already exist

**Wearing A Coat: Dual-Arm Robot-Assisted Dressing with Differentiable Clothing
Simulation** — Yiming Liu, Lijun Han, Hesheng Wang, arXiv
[2607.10999](https://arxiv.org/abs/2607.10999), submitted 2026-07-13.
Verified (abstract and HTML full text):

- Control targets are sleeve state in arm coordinates: a progress scalar
  `s` with `s = 0` at the hand, 1 at the elbow and 2 at the shoulder, the
  sleeve-to-arm distance `l`, a rotation angle `θ`, plus the armhole centre
  and orientation. The objective drives `s` to 2 for both sleeves in a
  multi-phase strategy.
- Actions are end-effector velocities from gradient-descent MPC through the
  authors' explicit differentiable cloth simulator (not IPC), with a linear
  local model compensating at 10 Hz because "nonlinear optimization takes
  several seconds to complete".
- The problem statement takes "the material property of the coat (including
  mass, stiffness, and friction coefficient with human)" as a given input.
- The controller is not learned; learning is used for pose estimation, arm
  motion prediction and garment reconstruction. Two Franka arms, RTX 4060.
  The fetched text did not show trial counts or success rates; the dressing
  survey found qualitative results on a dummy and on people, and no handling of
  cloth self-collision.

Consequence: "control the sleeve opening relative to the arm instead of the
gripper" is not a contribution by itself. Our arm-frame privileged features
are the same idea as their dressing coordinates.

## What else constrains the claim

| Work | Read | What it already establishes |
|---|---|---|
| Kotsovolis, Demiris, ICRA 2024 ([DOI](https://doi.org/10.1109/ICRA57147.2024.10611478)) and RA-L 2025 "Garment Diffusion Models" ([DOI](https://doi.org/10.1109/LRA.2024.3518104)) | abstract | A learned **forward** model of the garment opening (graph network; diffusion over the opening's point cloud) inside MPC for sleeve insertion; sim-trained, 97.5 % on a manikin (2024) and 91.2 % (2025). Same object as ours; the direction is reversed: nothing learns what the opening *should* do |
| Dressing in Motion (Sun et al., arXiv [2609.04759](https://arxiv.org/abs/2609.04759), under review) | full text | Diffusion policy over end-effector action chunks plus a >50 Hz layer that re-targets the pending end-effector positions from arm registration; sleeve centre and arm centreline appear in the problem statement, not as the policy output; 180 simulated and 210 real teleoperated demonstrations, separately trained sim and real policies, no sim-to-real |
| TAX3D (Cai et al., CoRL 2024, [2410.19247](https://arxiv.org/abs/2410.19247)) | full text | Learned goal cloth geometry relative to an anchor, then open-loop PD control, for hanging a cloth hole on a peg (the geometric analogue of threading); the authors list closed-loop goal-conditioned execution as the missing piece |
| FabricFlowNet (Weng et al., CoRL 2021, [2111.05623](https://arxiv.org/abs/2111.05623)) | abstract | Flow to a cloth sub-goal, converted to pick-and-place; sim-to-real folding |
| DefGoalNet + DeformerNet (Thach et al., ICRA 2024, [2309.14463](https://arxiv.org/abs/2309.14463); [2305.04449](https://arxiv.org/abs/2305.04449)) | abstract | A goal shape learned from about ten demonstrations, realised by a separate learned shape-servo controller that generalises to unseen material stiffness; surgical retraction, not dressing |
| I2L (Gangwani, Peng, ICLR 2020, [2002.11879](https://arxiv.org/abs/2002.11879)) | full text | Under a transition-dynamics mismatch, expert states remain useful and expert actions do not; locomotion with changed gravity, density and joint friction |
| Im2Flow2Act (Xu et al., CoRL 2024, [2407.15208](https://arxiv.org/abs/2407.15208)) | full text | Object flow as a cross-embodiment interface; §4.3 reports that it does **not** absorb the cloth dynamics gap: success drops for cloth folding on the real robot because "the same robot action should lead to the same object flow" fails |
| ATM (Wen et al., RSS 2024, [2401.00025](https://arxiv.org/abs/2401.00025)) | full text | Point tracks transfer human video to robot; Fold Cloth 0 % for BC against 63 % |
| Force-Modulated Visual Policy (Hao et al., CoRL 2025, [2509.12741](https://arxiv.org/abs/2509.12741)) | full text | A FleX-trained dressing action policy reaches 0.36 upper-arm ratio in PyBullet and 0.64 after fine-tuning on 204 target trajectories (the shift also changes the body model and arm motion); trains with static arms because "existing simulators are not yet stable or accurate enough to simulate deformable garment interactions with moving human limbs" |
| One Policy to Dress Them All (Wang et al., RSS 2023, [2306.12372](https://arxiv.org/abs/2306.12372)) | full text | The reference pipeline: SAC teachers for 27 arm-pose regions in SoftGym/FleX, distilled into one point-cloud policy; zero-shot on a Sawyer, 17 participants, 425 trials with the method: whole-arm 0.86, upper-arm 0.71, binary success 0.57 |
| Deformable shape servoing with online Jacobians (Navarro-Alarcon et al., T-RO 2016/2018; Lagneau et al., RA-L 2020; Hu, Sun, Pan, RA-L 2018; McConachie et al., IJRR 2020) | mixed | Closed-loop control of a goal deformation without known material or friction is established on real hardware |
| TraKDis (Chen, Rojas, RA-L 2024, [2401.13362](https://arxiv.org/abs/2401.13362)) | abstract | A privileged full-cloth-state agent distilled into a vision agent, for folding |
| DiffCloth (Li et al., TOG 2022) | full text | Open-loop trajectory optimisation for simulated hat and sock dressing through Projective Dynamics with friction; not IPC |

Not accessed and able to change the IPC-Jacobian point below: DiffClothAI
(Yu et al., IROS 2023), which states that it combines Projective Dynamics with
Incremental Potential Contact; no dressing application of it was found. **No
dressing paper found uses IPC or C-IPC contact**: the simulators are FleX,
PyBullet/Assistive Gym, DART with PhysX, Isaac PBD/FEM or DiffCloth-style
integrators. No paper found varies garment-to-skin friction in dressing, or
identifies it from garment-on-arm contact. Neither survey could sweep IEEE
Xplore or Google Scholar; "not found" is not "does not exist".

**Novelty verdict of both surveys.** Learning a sleeve-opening trajectory
relative to the arm from demonstrations and executing it with a separate
closed-loop controller is not published for dressing. Every piece is: the
opening as the modelled object (Kotsovolis and Demiris), arm-frame sleeve
state tracked by a controller (Wearing A Coat), and learned goal geometry with
a separate executor (TAX3D, FabricFlowNet, DefGoalNet). A reviewer will read
the combination as incremental unless the experiment below carries it.

## What would still be new

1. **A controlled contact-dynamics experiment.** Same task, trajectories and
   observations; action imitation against a sleeve-goal interface; vary only
   garment-to-arm friction, cloth stiffness and then the simulator; report
   success and the target-domain data needed against shift size. No surveyed
   paper isolates this axis for deformable manipulation.
2. **A claim that survives Im2Flow2Act §4.3.** Not zero-shot invariance, but
   that a sleeve-goal interface confines the dynamics gap to the executor: the
   learned goal predictor is reused unchanged and only the executor adapts.
3. **Beyond Wearing A Coat:** learned goals instead of hand-designed phases,
   an executor that does not need friction as an input, and a policy with no
   simulator at run time instead of seconds-long optimisation.
4. **An IPC-accurate Jacobian of sleeve state with respect to the gripper**
   as the executor or its training signal, closed loop. The dressing actor
   audit found geometric task derivatives agreeing with finite differences;
   DiffClothAI must be read before claiming this.

## Traps reviewers will probe

- **Equal target access.** If the sleeve-goal executor uses target-domain
  Jacobians or rollouts, the action baseline gets the same target budget.
- **The goal may itself depend on friction.** Higher friction may call for a
  different sleeve path. Measure the shift of successful sleeve paths across
  friction before claiming the goal predictor transfers.
- **Confounded shifts.** Changing simulator also changes geometry and
  perception; hold observations fixed while varying contact parameters.
- **Metrics.** Report upper-arm and whole-arm dressed ratio as in the dressing
  literature, with the progress-metric fix from the audit.

## Bearing on the owner's single-workstation goal

The reference pipeline (27 region-wise SAC teachers, then distillation) and
Wearing A Coat (seconds of optimisation per control update) are both expensive,
one at training and one at run time. The testable single-workstation proposition is: a scripted or
Jacobian-based sleeve-goal teacher generates data on one GPU, a policy is fit
without simulation in the loop, and only the executor is adapted when contact
dynamics change. The first measurement it needs is cheap and needs no learning:
run one sleeve-servo teacher on the audit's 25 cells at several friction values
in IPC and measure whether successful sleeve paths shift and whether the
executor, not the goal, is what fails.
