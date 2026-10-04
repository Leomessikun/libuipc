# Research brief for GPT-6 Astra: post-training that survives sim-to-real (2026-10-05)

Requested by the owner. Written by the Claude session that ran the value-teacher, composite-teacher and
DAgger studies (`2026-10-03-value-anchored-lookahead.md`). This is a request for a **research study**, not for
more collection: the owner judged that the composite-teacher + DAgger line has no value for the goal and
stopped it. All Claude jobs are stopped; the GPU is free of Claude work.

## Goal (owner)

A **better post-training method** for the pretrained FMVP-style dressing policy whose gains **minimize the
sim-to-real gap**: improvements obtained in Genesis+IPC must carry over to the real robot (one external
RealSense D435i, FMVP observation `wang_static_arm`, translation actions) with as little real fine-tuning as
possible. Generalization targets stay unseen garments (ClothesNet) and moving arms (GRAB).

## What is established (do not re-derive)

- Sim success of r1 (IPC-lookahead DAgger) is far above fmvp_sim (130 vs 100 of 201 test units); flow 138.
  Nothing so far measures whether any of these gains survive a different simulator, physics parameters or the
  real robot.
- Local per-decision post-training saturates at r1: one- and four-step IPC lookahead labels, four value-ranked
  lookahead variants (learned outcome value V(s), AUC 0.88), hazard-advantage-weighted BC (33 vs 39 of 55,
  below r1), effect codes (0 verified repairs). Most failures are decided at sleeve entry, long before the
  grasp is lost; a shared blind spot (37 of r1's 71 test failures fail for all eleven FMVP-derived policies).
- A privileged composite teacher (scripted translation-only entry route from arm landmarks, r1 after 60 % of
  the forearm) reaches 64 vs r1 45 of 69 development units, but it is a sim-privileged script; its distillation
  (BC + DAgger with state-feedback expert labels) reached 43 of 69 before it was stopped. Treat it as a
  diagnosis (entry phase is the bottleneck), not as a method.
- Outcome noise: identical replicas agree on the outcome in 82 % of groups; moving-arm runs do not reproduce
  run to run. Any ranking signal must be checked against this noise floor first.

## Research questions

1. **Prior art map (primary sources, 2022-2026).** Post-training of pretrained visuomotor policies in
   simulation with real-robot evidence that the gains transfer: residual RL on BC policies (ResiP, Policy
   Decorator), RL fine-tuning of diffusion/flow policies (DPPO, ReinFlow, DSRL), trust-region/KL anchoring to
   the base policy, sim-and-real co-training, real-to-sim-to-real (RialTo, Real-is-Sim), delta-action
   alignment (ASAP), TRANSIC, DFP/drifting (already assessed in `2026-10-03-garment-outcome-posttraining.md`
   Section 14). For each: mechanism, which gap (dynamics / observation / action), real data needed, real
   results.
2. **Cloth and dressing sim-to-real.** How FMVP / Yufei Wang et al., Dressing in Motion (arXiv 2609.04759),
   Garment Diffusion Models and cloth-folding works transferred; which simulator parameters (stiffness,
   friction, thickness, contact model) and which observation effects (point-cloud noise, occlusion, voxel
   size) dominate the reported gap.
3. **Robust improvement across simulators.** EPOpt / CVaR over model ensembles, robust MDPs, ADR / DROPO /
   BayesSim / SimOpt, sim-to-real predictivity metrics, pessimism over dynamics. What is already covered by
   "accept a post-training update only if it holds across a physics ensemble", and what would be a new,
   testable angle for a pretrained deformable-manipulation policy.
4. **One in-repo measurement before any method.** Do the existing sim gains survive physics perturbation?
   Paired evaluation of fmvp_sim, r1 and flow on the same development units under perturbed physics
   (cloth-body friction, Young's modulus, bending stiffness, density/thickness) and perturbed observations
   (point-cloud noise, dropout, voxel shift). `collect_garment.py` already exposes `--friction`,
   `--cloth-density`, `--cloth-strain-rate`; `cloth_youngs`, `cloth_bending_stiffness`, `cloth_thickness`
   are `DressingEnv` config fields reachable through the same `material` override (a few lines). If r1's
   gain over fmvp_sim shrinks under perturbation, the current post-training exploits the simulator, and that
   measured shrinkage is the target a sim-to-real-aware post-training method must reduce.

## Deliverable

A short report in `agent_docs/performance/` (English) with: the prior-art table (with URLs), the measured
robustness of existing gains (question 4, if run), and **one or two concrete post-training method
candidates** that target the sim-to-real gap, each with its falsifiable prediction, the strongest prior-art
objection, the smallest decisive experiment and its cost. State clearly what is new and what is not.

## Constraints

- Total GPU use below 80 GiB; others hold about 64-67 GiB, so one sim worker at a time.
- Development bodies are test bodies 1-14; test bodies 15-41 are reserved for one final check. Paired units,
  exact McNemar; report aggregated numbers.
- Do not restart the M4 queue (`STOP_REQUESTED.json`); stage explicit paths when committing, never `add -A`.
- Relevant code: `scripts/wang_transfer/{collect_garment.py, eval_policies_batched.py, finetune_fmvp_bc.py,
  train_flow_policy.py, outcome_value.py, expert_relabel.py}`; policies under `output/uipc_manip/`
  (`fmvp_ipc_dagger_r1_20260924/model/fmvp_ipc_bc.pt`, `/home/ge47gax/Desktop/fmvp_sim.pt`), paired
  baselines in `output/uipc_manip/policy_eval_test_20260927/results.jsonl`.
