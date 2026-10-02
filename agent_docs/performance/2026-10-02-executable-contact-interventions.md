# Learning executable contact interventions — research proposal, 2026-10-02

## Decision and status

The candidate is to learn **how ordinary robot actions change a measured
dependence on a contact mechanism**. A simulated contact intervention diagnoses
a potential obstruction; a learned action-response map proposes a physical
recovery; an explicit residual measures failure to realize the requested change
within the tested local action family. All accepted action labels must improve
a continuation under the original physics.

This is a concrete proposed learning construction, not a validated algorithm or
an established novelty claim. The generic recipe of relaxing contact, planning,
and distilling is already covered by prior work. The proposed response-map
supervision must earn its place against direct recovery search and relaxed-state
inverse dynamics at matched total cost.

The existing [observed-motion teacher/student experiment](2026-10-02-observed-motion-student.md)
continues as the empirical baseline. This document does not change its queue,
training gate, candidates, scores, or resource limits. No experiment implementing
the new construction has run.

## Evidence and competing explanations

Read-only inspection of `output/uipc_manip/m3_observed_20261002/` found:

| Corrected validation | Completed result |
|---|---|
| pass/current-pose planner | Grasp criterion violated at step 13; zero action changes |
| pass/GICP | Grasp criterion violated at step 161; maximum armhole fraction about 0.551 |
| pass/observed-motion planner | Still running at inspection |
| lift/observed-motion planner | Still running at inspection |

At lift/observed decision 12, ten of twelve forecast candidates were infeasible;
the two feasible candidates followed observed arm motion. This is a candidate
coverage observation, not evidence that frictional trapping is the cause.
The current implementation holds one action for four decisions. It can choose
different actions on successive real decisions, but cannot score a deliberate
release-then-resume sequence within one forecast.

Important competing explanations are forecast error, insufficient candidate
coverage, action latency, initial alignment, normal geometric obstruction,
soft attachment tracking, and inconsistent replay. The environment declares
grasp loss when held-vertex tracking exceeds 2 cm. It does not simulate physical
finger slip. A contact-based method that merely exploits this attachment model
would have a weak claim to physical dressing transfer.

## Proposed mechanism

Some intact-grasp failures may require a temporary release or lateral motion
before forward progress can resume. A scalar outcome for a few pulling actions
does not identify which physical dependence a recovery should change.

For example, reducing friction at a wrist patch may let a nominal pull advance.
That observation alone is insufficient: the robot may be unable to unload the
patch without losing its grasp or removing the sleeve. The candidate learns
both the obstruction and whether the available robot motions can reduce it.
No assumption is made that less friction is always beneficial.

## Mathematical construction

### 1. Counterfactual continuation deficit

Let $x_t$ be full simulator state, $h_t$ deployable point-cloud/proprioceptive
history, $F_0$ normal physics, and $F_c$ physics with friction reduced only at
cloth/body patch $c$. Normal collision, elasticity, attachment and other
contacts remain intact. Human motion $m$ is identical across paired branches.
Use a fixed continuation policy $\bar\pi$, horizon $H$, and geometric cost $C$:

$$
D_c(x_t;m)=
\mathbb E[C(F_0^H(x_t,\bar\pi;m))]
-\mathbb E[C(F_c^H(x_t,\bar\pi;m))].
$$

The expectations include replay variability. This is a sensitivity to a
specified intervention, not unique causal blame. Use normalized sleeve progress,
stretch and attachment margin, and report failure separately. Do not rely on
the uncalibrated force proxy. For a small set of patches, collect
$d(x;m)=[D_1,\ldots,D_K]^T$. Its entries need not add up to total failure.

### 2. Physical action changes the intervention dependence

Let $u$ parameterize a bounded recovery prefix of length $L$, always executed
under $F_0$. Let $u_0$ be a nominal prefix of the **same length**:

$$
x_u=F_0^L(x_t,u;m_{t:t+L}),\qquad
\Phi_t(u)=d(x_u;m_{t+L:t+L+H}).
$$

All prefixes finish at the same absolute human-motion time. Compare their
continuations over the same remaining horizon. Comparing an immediate
continuation against a delayed one would confound the intervention with motion
timing. Estimate locally:

$$
\Phi_t(u_0+\delta u)-\Phi_t(u_0)\approx B_t\delta u.
$$

$B_t$ is an **action-to-intervention-dependence map**. Initial labels can use
finite physical probes; a model $B_\theta(h_t,c)$ later amortizes those probes.
Patch features use body-relative geometry and garment location, not garment IDs
or a shared mesh vertex numbering. Garment transfer is a hypothesis to test.

### 3. Project a desired change onto the tested action family

Choose $r_t$ to reduce reliable positive entries of $\Phi_t(u_0)$ while
preserving helpful contact effects. Solve a small constrained problem:

$$
\delta u^* = \arg\min_{\delta u\in\mathcal U_t}
\|W_t^{1/2}(B_t\delta u-r_t)\|^2
+\lambda\|\delta u\|^2.
$$

$W_t$ downweights noisy intervention estimates; $\mathcal U_t$ bounds action,
prefix duration, predicted attachment violation and excessive progress loss.
Predicted constraints are screening tools, not physical guarantees.
Reusing the normal branch correlates the entries of $d$; use covariance-aware
whitening where estimable, or explicitly report a diagonal approximation.
Snapshot restoration alone does not establish beneficial common random numbers.

Keep a separate minimum-fit diagnostic:

$$
e_t=\min_{\delta u\in\mathcal U_t}
\|W_t^{1/2}(B_t\delta u-r_t)\|.
$$

For an unconstrained exact linear model, this equals the norm of the component
outside the column space of $W_t^{1/2}B_t$. With bounded actions, noise and a
nonlinear contact transition, it is only a **local model residual**. A large
value does not prove that the physical task is impossible. Record unsuccessful
searches as unresolved under the tested budget, not ground-truth impossibility.
Calibrate against fresh physical realization errors and report residual divided
by requested effect magnitude (with a noise floor), so tiny targets do not look
spuriously controllable.

### 4. Accept only verified useful physical recovery

Run $u_0+\delta u^*$ and the continuation under $F_0$. Require improvement over
$u_0$ followed by the same continuation, intact attachment, valid sleeve
topology appropriate to the current insertion stage, and bounded stretch.
Allow bounded useful retreat; do not require full enclosure before insertion
or permit a shortcut that loses the achieved threading. Validate promising
branches with fresh repeats after candidate selection to reduce selection bias.

Reducing $D_c$ alone is insufficient: both relaxed and ordinary outcomes could
become equally bad. Also compare relaxed continuation quality after $u$ against
that after $u_0$, at the same time; reject a supposed release that merely makes
both futures fail. Ultimately, normal-physics task improvement decides label
acceptance. Keep failed proposals as response-model evidence, not expert actions.

### 5. Learning and deployment

A first learner predicts intervention effects and their response to a queried
prefix from visual history. A possible supervised objective is

$$
\mathcal L =
\sum_c w_c\,\ell(\hat D_c,D_c)
+\alpha\sum_{u,c}w_{uc}\,
\ell(\widehat{\Phi(u)-\Phi(u_0)}_c,\Phi(u)_c-\Phi(u_0)_c)
+\beta\,\mathcal L_{\rm verified\ action}.
$$

$\ell$ is a robust regression loss; weights reflect measured label reliability.
The final action term uses only normal-physics-validated improvements. A small
proposal model or the existing student adapter can consume these labels. Flow
matching is optional if verified recovery actions are genuinely multimodal.
The deployment interface uses observed history; it cannot receive contact IDs,
future GRAB frames or intervention results from the simulator.
The response model predicts a conditional response under a deployable motion
forecast. Exact-future GRAB interventions may diagnose an upper bound but are
not automatically realizable supervision from $h_t$. Include a causal-forecast
version before claiming deployable recovery, and test aliasing/calibration.

The proposed contribution is the intervention-dependence response supervision
and its usefulness for discovering transferable executable recoveries. The
regression, local least-squares projection and distillation individually are
standard tools. No new convergence theorem is asserted.

## Prior-art boundaries

| Primary source | Already established | Distinction the proposed method must test |
|---|---|---|
| [GenH2R, CVPR 2024](https://arxiv.org/html/2401.00929v2) | Future-aware privileged planning and visual-history policy learning | Generic anticipatory teacher/student is a baseline |
| [Tuning-Free Contact-Implicit Trajectory Optimization, 2020](https://arxiv.org/html/2006.06176v1) | Relaxed contacts; section II-F uses virtual-force information to improve feasible actions | Learn how ordinary actions change counterfactual continuation deficits; compare with relaxation/refinement |
| [Minimum Constraint Removal, IJRR 2014](https://journals.sagepub.com/doi/10.1177/0278364913507795) | Identify constraints whose removal permits a path, including failure explanation and rearrangement | Obstruction localization alone is insufficient |
| [Learning Long-Horizon Robot Manipulation Skills via Privileged Action, 2025](https://arxiv.org/abs/2502.15442) | Virtual forces and relaxed constraints removed through curriculum | No privileged executed training actions; test the response-map supervision against a curriculum |
| [ContactMimic, 2026](https://arxiv.org/html/2607.08742v1) | Paired contact-command data and learned contact control | Contact presence labels versus measured intervention dependence and its action response |
| [TACO, 2026](https://arxiv.org/abs/2607.02840) | Imagined contact corrections, inverse-dynamics labeling and filtered post-training | Generic failure-to-correction synthesis is insufficient; compare ordinary physical subgoals |
| [Garment Diffusion Models, RA-L 2025](https://spiral.imperial.ac.uk/entities/publication/a0176f43-5eb7-4530-93d8-35c0118514df) | Action-conditioned garment prediction and MPC | Compare a same-capacity ordinary outcome predictor without interventions |

The team audit rejected presenting four-term action/motion contrasts, generic
uncertain-future MPC, action sets, or generic contact relaxation as independent
algorithm novelty. The proposed construction remains subject to these close
precedents. Search coverage is not proof of originality.

## Next three experiments and stopping conditions

### A. Establish whether the proposed mechanism exists

Select 12 intact-grasp, failure-adjacent states from at least three garments,
including both static and moving-arm cases. Include some progressing controls;
do not select only examples visually resembling a jam. Hold actions and motion
fixed. Compare ordinary cloth/body friction with reduced cloth/body friction,
preserving cloth self-contact and attachment. Repeat each branch three times:
72 short continuation runs, plus restoration controls. Start with a global
cloth/body intervention; local attribution is unjustified if that has no effect.

Report paired progress/attachment changes, solver failures separately, and
effect sizes relative to replay variation. Stop if effects are negligible,
unreliable, dominated by attachment artifacts, or ordinary velocity following
already removes the relevant failures. Three repeats are a diagnostic, not a
publication-level sample size.

### B. Test executable effect projection before policy training

On survivors, probe two candidate contact patches and a three-dimensional
prefix parameterization (retreat, lateral shift, observed-motion follow).
Use a common 2-decision prefix and 8-decision continuation initially; these are
proposed diagnostic horizons, not known sufficient horizons. A central-difference
map uses nominal plus six physical prefixes. With two patches, each prefix needs
one normal and two intervened continuations: seven physical prefixes plus 21
continuations per snapshot per repeat, before final validation. This cost is
substantial and must be included; the map is not a free IPC label.

Compare equal total simulator time and matched recovery families:

1. Direct full-physics search over release-then-resume sequences.
2. Relaxed-state physical refinement / ordinary inverse dynamics.
3. A geometric/Jacobian contact-correction proposer inspired by Onol section
   II-F, given the same final physical validation and a documented cloth adaptation.
4. Intervention response map without its residual screen.
5. Complete proposed construction.

Primary diagnostic: additional validated recoveries per simulator second.
Also report full episode completion, false proposals and residual calibration.
If direct search wins or matches it, do not scale the new representation. A
win over constant-action H=4 alone does not support the algorithm claim.

### C. Test transfer and amortization

Only if B succeeds, train a small response/proposal model using verified labels
and failed physical probes. Hold out whole garments and GRAB subjects/sequences;
evaluate unseen garments, unseen motions, and their conjunction. Use three
training seeds and paired task cells; do not count seeds as distinct motions.
Match data, encoder capacity, updates and compute against ordinary consequence
prediction and current history-only DAgger. Include shuffled patch/effect labels
and equal-cost extra physical samples as controls.

The intended scientific result is useful contact recovery at lower total data
generation cost, with transfer beyond memorized garments. If the learned model
does not amortize its extra queries, the method fails the workstation constraint.

## Implementation limitations

`DressingEnvConfig.friction` exists, but simply changing it after initialization
is not a validated intervention. The CUDA contact table is built at initialization;
`ContactElement::apply_to` assigns a geometry-level metadata ID. A local body-patch
intervention is therefore not presently a demonstrated one-line Python change.
First verify a controlled cloth/body pair intervention in an isolated harness;
local patches may require supported per-vertex attributes or backend work.

Snapshot controls must preserve position, velocity, cloth history, attachment,
body-motion clock and solver configuration. Equal positions alone do not certify
equal continuations. Do not change normal collision or cloth constitutive
parameters to implement a friction diagnostic. Do not rebuild or swap libraries
under the active M3 jobs.

This proposal deliberately does not assign a one-day completion promise.
Physics-query throughput, snapshot support and intervention validity determine
the implementation budget. Existing FMVP trajectories supply starting states,
nominal motion and controls; successful full expert trajectories are not required
for the first mechanism test.
