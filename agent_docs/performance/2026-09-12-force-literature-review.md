# Force-field literature verification

Primary-source review, 2026-09-12. Scope: verify three supplied IDs and evidence relevant to distributed tangential traction; falsify overly broad novelty claims, not establish an exhaustive priority search. No repository code changed.

## Identity check

All supplied arXiv identifiers resolve to the claimed papers:

| ID | Verified title | Submission / status stated by arXiv |
|---|---|---|
| [2308.01696](https://arxiv.org/abs/2308.01696) | No Free Slide: Spurious Contact Forces in Incremental Potential Contact | 3 August 2023 |
| [2606.20426](https://arxiv.org/abs/2606.20426) | TaCauchy: An Extensible FEM Framework for Vision-Based Tactile Simulation | 18 June 2026; page says accepted IROS 2026 |
| [2605.24339](https://arxiv.org/abs/2605.24339) | IsaacIPC: Coupling High-Fidelity Simulation and Realistic Rendering for Contact-Rich Robotic Systems | 23 May 2026; technical report |

## What the papers establish

**No Free Slide.** Sections4.2–4.3 and6.1 analyze changing active primitive sets in IPC: superimposed primitive barriers can create energy variations and tangential resistance even when an object slides at constant separation above a perfectly planar, frictionless surface. Forward sliding and inverse-design benchmarks exhibit the resulting artifacts; smooth IMLS surface contact reduces them. The paper explicitly does not model physical friction. This is evidence of contact-discretization error, not of incorrect Coulomb-friction coefficients or insufficient friction lagging. It does not quantify error for our cloth/body mesh or prove every modern IPC variant equally affected. Relevant inference: a nonzero tangential projection of barrier force is not automatically physical skin shear; the deployed implementation needs a frictionless planar-slide negative control and sensitivity tests. [Full text, §§4.3,6.1](https://arxiv.org/html/2308.01696v1#S4.SS3)

**Original IPC lagged friction.** The original paper freezes normal-force magnitudes and tangent bases to construct an integrable friction potential, then optionally updates them through successive nonlinear minimizations toward momentum balance (Eq.17). It explicitly states no general convergence guarantee for these lagged updates and notes failures under large deformation/high-speed impact. Smaller friction regularization velocity reduces stiction error at higher computational cost. Inference: checking inner Newton convergence alone does not establish self-consistent friction, while converging friction updates still does not eliminate a discretization artifact in the underlying contact potential. [Original author-hosted paper, friction direction/magnitude discussion](https://cims.nyu.edu/gcl/papers/2020-IPC.pdf)

**TaCauchy.** Computes FEM Cauchy stress and surface traction, including normal/tangential decomposition. Its physical experiment uses six perpendicular pressing loads1.2556–4.7332N. Crucially, §IV-B adjusts simulated indentation until simulated Fz matches measured Fz within0.0800N, then compares tactile images; mean SSIM0.9377 measures image structure. This validates image response conditioned on matched total normal force, not independently predicted force error, shear traction, or distributed force calibration. §IV-C sliding/torsion fields are qualitative simulation analyses. Therefore cite its extraction method as a technical precedent, not measured proof of calibrated dressing-force labels. [Full text, §§III-C,IV-B,IV-C](https://arxiv.org/html/2606.20426v1#S4.SS2)

**IsaacIPC.** Introduces geometric mortar contact potential (GMCP), sampling/integrating a barrier on tactile surfaces. Contact patch and frictionless Hertzian benchmarks evaluate normal pressure transfer/distribution. Its conclusion explicitly leaves tangential traction, friction, stick–slip and shear deformation for further work. Useful pressure-discretization precedent; it does not supply a validated cloth–skin shear model or prove ordinary pointwise IPC reaction forces are accurate tractions. [Full text, §§4–6](https://arxiv.org/html/2605.24339v1#S6)

## Sweeping novelty claims fail

- **“Dressing only uses a wrist scalar” is false.** Deep Haptic MPC uses three-axis force and torque measurements plus kinematics, predicts27 force-magnitude taxels on the fist/forearm/upper arm, and controls using future force maps. Those scalar-valued spatial taxels are not a calibrated vector shear field, but they are distributed body-force reasoning. [Erickson et al., §III](https://arxiv.org/html/1709.09735v2#S3)
- **“No distributed dressing force prediction” is false.** Visual Haptic Reasoning (Wang, Held, Erickson, RA-L/IROS2022) predicts per-point contact and normal-force magnitude from visual and kinematic observations; dressing is an evaluated task. Its project page explicitly identifies the displayed quantity as normal-force magnitude. [Author project](https://sites.google.com/view/visualhapticreasoning/home)
- **“No prior distributed shear work” is false across robotics/pHRI.** Choi et al. use a tri-axial tactile skin on a human/robot arm and compare touch recognition with/without distributed shear. That is touch classification, not robot dressing; it cannot alone settle dressing-specific priority. [Primary paper](https://arxiv.org/abs/2210.00135)

A defensible proposed contribution is **experimentally validated distributed normal/tangential traction for cloth–body interaction and its demonstrated benefit to dressing control**. This is a research target, not a verified first-of-kind claim. Distinguish the claim from existing force-map prediction and tactile stress extraction; a focused dressing-specific shear literature search remains necessary before asserting priority.

## Validation implications (our recommendations, not paper results)

Keep three tests separate:

1. **Numerical consistency:** Newton/linear residuals, outer friction lag convergence, timestep and regularization sensitivity, force balance, action–reaction and integrated wrench consistency.
2. **Discretization consistency:** frictionless planar sliding, remeshing/orientation/density changes, barrier support sensitivity, contact patch area convergence. Define traction as force per deformed area with consistent normal/sign; a per-vertex force vector alone is not a pressure/shear stress field.
3. **Physical accuracy:** independently measured normal and tangential loading paths, material/friction identification on a calibration split, held-out predictions of wrench/contact area/traction; test sliding, stick–slip and hysteresis where relevant. Geometry plausibility and tactile-image SSIM cannot substitute for these force errors.

A policy learning from mechanically derived labels may still exploit simulator artifacts. Neither low numerical residual nor a visually plausible policy makes those labels physical ground truth. Compare calibrated scalar-wrench, distributed-normal and distributed-normal-plus-shear variants to identify the incremental control value.
