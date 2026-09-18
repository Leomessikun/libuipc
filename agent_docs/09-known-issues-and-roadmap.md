# 09 — Known Issues, Tech Debt, and Roadmap

## Motion-pretraining pilot — completed negative result, 2026-09-18

The [bounded PointZero-inspired comparison](performance/2026-09-18-motion-pretraining.md)
is implemented, tested and complete. Each variant receives 3,000 FQL updates and
eight full native dressing rollouts. Random / geometry / motion actor-encoder
initialization scores **2/8 / 1/8 / 0/8** under final coverage plus whole-grasp
validity; zero simulator errors. No motion rollout even reaches peak coverage .7.

Improved whole-cloth prediction does not establish policy improvement. Near the
held-out elbow, prediction is worse than zero motion, and opening-boundary track
coverage is sparse. Do not scale this recipe without a new, bounded hypothesis.
One seed and four repeated configurations cannot establish a general method ranking.
The early-turn flag is already true at reset for both withheld configurations;
its interpretation and the shoulder reward boundary still need validation.

No new training is running. Full online continuation, critic transfer, continued
auxiliary prediction, broader motion coverage and multi-seed confirmation remain
untested. Fourteen focused tests pass; deployment uses only the ordinary FQL actor.

## First FQL run — authorized and implemented, 2026-09-17

The owner requested training. [Protocol and implementation](performance/2026-09-17-fql-pretraining.md)
select known FQL after checking newer RQL and other Levine-group references.
Point-cloud learner, full-state checkpoints, data-boundary tests and native
evaluation are implemented; 19 focused tests pass. Existing expert sequences
were replayed with true successor observations and a declared shoulder-ray
correction. Preparation and the first offline run now finish in 662.56 s and
334.00 s. In repeated complete dressing episodes, FQL actor / flow prior score
4/8 versus 0/8 by final coverage and whole-episode grasp; all four FQL passes
retain coverage at the final 20 decisions. Both score 0/8 on the historical
early-turn paper filter. This does not isolate Q improvement from one-step
distillation, nor establish robustness or the best algorithm.

Full-state continuation to 30,000 total gradient updates and native evaluation
finished on 2026-09-18: actor 5/8, prior 1/8 by coverage plus whole-grasp validity.
Actor training-body results improve 2/4 to 4/4, withheld-body results fall 2/4
to 1/4; this small sample does not establish improved generalization. Every actor
episode triggers early_turn, and neither policy passes paper-filter AND valid
whole-episode grasp. No FQL training process remains.

Next gates: audit actual route/geometry and persistent withheld failures; add a
matched one-step control without actor Q improvement; then bounded same-learner
online continuation and comparisons with from-scratch FQL and SAC/prior-data SAC.
Full online integration remains open. See the
[updated research plan](performance/2026-09-17-fql-pretraining.md#research-plan-after-the-completed-pilot).
Do not describe resumed offline training as online adaptation or generic FQL
integration as algorithmic novelty.

## Shared FQL pretraining — proposed first implementation, 2026-09-17

The owner requested a concrete way to train dressing. The
[design](performance/2026-09-17-rl-pretraining-contract.md#concrete-first-training-design-shared-fql-pretraining-and-continuation)
selects a known FQL learner: behavior prior, critics and one-step actor, trained
from compatible existing data and continued under the same objectives with IPC
interaction. Keep the existing observation/action/controller contract. This
replaces the learning objective, not the controller, and is not SAC with an IPC
gradient. No algorithmic novelty or policy improvement is claimed.

Implementation order: validate a common corrected reward/success contract;
adapt existing episode/replay data with true successors and explicit reward
versions; add the FQL learner; connect the existing collector; compare complete
learning against SAC and prior-data RLPD at matched total workstation cost.
Do not mix unrelabelable old rewards with corrected rewards or manufacture
timeout successors. Training and native simulation remain stopped; no learner
implementation was made in this design review.

## RL pretraining contract — current design priority, 2026-09-17

The owner challenged the sleeve-transfer proposal and referred to Berkeley
CS 285. [Reassessment](performance/2026-09-17-rl-pretraining-contract.md): preserve
the goal of fast reward-based policy learning using existing experience. Define
what pretraining transfers and how subsequent RL consumes it; evaluate downstream
improvement rather than equating prediction/BC loss with useful initialization.
Component-only pretraining is valid; pretraining every network is not required.

Representation, BC, replay-IQL and online-SAC paths have different contracts.
The recorded IQL control did not use expert trajectories or continue online.
RLPD/FQL are reference methods, not demonstrated dressing solutions. Include the
shoulder-overshoot metric issue from the newer reward audit when specifying the
learning objective and evaluation. Align compatible existing transitions,
checkpoint semantics and total-cost/success evaluation before selecting another
algorithm modification. No new training, native rollout or implementation.

## Sleeve-transfer proposal — withdrawn as default next step, 2026-09-17

After the completed SAC audit, the owner again asked to abandon the previous idea
and explore a new direction. [Research assessment and bounded comparison](performance/2026-09-17-dressing-transfer-direction.md):
investigate transferable sleeve-motion goals with domain-specific execution.
Published dressing diffusion/MPC, relative placement and hierarchical/latent
transfer are close precedents; generic module combinations are not novel.

No method is implemented and no new policy result exists. First proposed screen:
matched action-sequence imitation versus sleeve-goal control on compatible native
data, complete dressing episodes, two seeds and withheld body configurations.
Only a useful native result warrants testing whether additional source geometry
reduces target learning needs. Source geometry availability, target reachability,
occluded-state estimation and real-robot adaptation are unresolved. Do not assume
that contact-independent goals exist or that a good predicted goal is executable.
Existing trajectory contracts need alignment; do not restart long SAC or IPC
correction runs. Training remains stopped.

## Ordinary SAC rollout diagnosis — completed, 2026-09-17

The owner abandoned the IPC algorithm direction and explicitly requested normal
SAC rollout analysis first. [Completed audit](performance/2026-09-17-normal-sac-rollout-audit.md):
SAC partially advances then stalls, or moves the sleeve opening away from the
upper arm. Zero controller rejections across eight saved and six new episodes;
some grasp violations, plus fully grasp-valid failures. Failure also occurs on
a sampled training pose. New geometry captures total 1,800 decisions / 240.69 s;
no learning or simulator errors, metrics independently recomputed from geometry.

Existing trajectories were not consumed by this baseline. Source SAC sees scarce
successful experience and only 408 complete episodes across 225 planned cells.
Historical ordinary SAC peak 7/25 geometric success is not stable across later
checkpoints/seeds or validated as whole-episode grasp success. Compare saved best
and final SAC under one fixed protocol before isolating critic/state/exploration
causes; do not select a replacement algorithm from BC evidence. Training remains
stopped. All older proposed IPC/recovery next steps below are historical and
superseded by this owner instruction.

## IPC policy-gradient research — abandoned by owner, 2026-09-17

The owner rejected the novelty of supervised SAC. The
[new research note](performance/2026-09-17-ipc-policy-gradient-research.md)
compares close prior art and specifies a candidate joint query/horizon/repetition
allocation problem for return-based policy-gradient correction. Basic gradient
and selection-weight identities are established methods, not novel contributions.
Seven standalone mathematical tests pass; no native correction collector or
learned allocator is implemented. First demonstrate a better correction direction
per wall time at learner states, including restoration variability and time-limit
bootstrap accounting, before training. Do not silently substitute teacher
continuations for current-policy returns. No training is running; algorithmic
novelty, dressing improvement and real-world transfer remain unproven.

## IPC recovery supervision — 2026-09-17

[SAC pretraining integration](performance/2026-09-17-recovery-sac-pretraining.md)
implements the owner's clarified scope: verified recovery actions enter the
ordinary SAC actor update during pretraining, while downstream plain SAC and
inference use the existing interface. Critic targets are unchanged. Seventy-four
focused tests pass. The fixed live comparison was stopped during its control
run after the owner's novelty objection; treatment and final evaluation never
started. Keep the option as a conventional baseline, not a novel RL result.

[Novelty audit and pilot](performance/2026-09-17-recovery-teacher.md): the broad
teacher/pretrain/correct/SAC recipe has close prior art. New bounded prototype
compares full-episode geometric recoveries at a BC-visited training state with
policy and scaled-policy controls, then requires fresh-world verification before
admitting labels. Seventeen focused tests plus two native terminal/reset checks
pass. Outward-route search and independent verification succeed on both copies,
producing 360 admitted rows. BC transfer comparison finishes at 2/8, 2/8, 3/8
successes for original/continued/recovery actors, with all failing the target
sleeve twice. Matching the teacher's command caps does not rescue that failure.
All BC/teacher jobs finished. The useful recovery teacher is demonstrated on one
case; its autonomous transfer, improved robustness and novelty remain unproven.

## Existing-expert policy pretraining — 2026-09-17

[Training and full-episode protocol](performance/2026-09-17-expert-policy-pretraining.md):
existing-data reconstruction and fast CUDA BC implemented; 17 focused tests pass.
Twelve replays pass geometric/grasp admission but fail historical early-turn
admission. Body-disjoint training uses 1,800 rows from body 14046; another 1,800
rows on 14047/14048 are withheld. Fixed 3,000-update BC completed in 86.8 s through
scheduled validation, without simulation during training. Full dressing evaluation
completed: BC/SAC 2/8 versus 0/8 coverage-and-grasp successes, .25968/.29953 mean
coverage. BC succeeds only on one training configuration; withheld-body success
is 0/4. Both pass the historical paper filter 0/8. Robustness and a new RL method
remain unproven. Increase compatible body/state coverage and verify learner-state
recoveries; inspect the training-case .99-to-zero coverage collapse before making
new improvement labels. All jobs finished. The larger Newton corpus requires
explicit observation/control transfer.
Do not treat the BC checkpoint's untrained critic as a pretrained SAC critic or
silently reuse the current development bodies as pristine research test states.

## Parallel dressing trajectory search — 2026-09-17

[Implementation and bounded protocol](performance/2026-09-17-parallel-dressing-trajopt.md):
CUDA CEM plus batched IPC forward solves implemented; 12 focused tests pass.
The optimizer evaluates each candidate on each restored slot and caps command
magnitudes against the reference. Native tests completed: matched 384-decision
evaluation is 1.439x faster with four copies, but only 1.079x including setup and
approach. In fresh validation CEM/reference/SAC coverage is .57704/.58330/.57259,
each 0/4 sustained successes; no robust policy benefit established. All jobs
finished. Resolve forward/recovery variability and objective/horizon validity
before increasing the learning budget.
CPU environment orchestration remains; a complete contact adjoint, policy
distillation and robust dressing improvements are not established by this work.

## Dressing recovery research — 2026-09-17

[New evidence and experiment plan](performance/2026-09-17-dressing-research-redesign.md):
fixed identical actions after snapshot recovery produce a .04916 final coverage
range over three 12-decision runs, despite identical restored positions, valid
grasp and zero rejection. Actor inference on cached input is identical. Compare
cold-prefix replay versus recovery and inspect full state/numerical convergence;
do not assume velocities are omitted or nondeterminism makes RL impossible.
Audit compatible existing expert data and success/admission semantics before new
pretraining: one 25-episode diagnostic set has 11 geometric final successes,
six also within 2 cm maximum tracking, but no policy observations and no entries
passing the early-turn filter. This is not an inventory of the full corpus.
Then test distinct recovery routes with matched movement and compute controls;
longer local gradient sequences were already tried. Only train a correction-guided
SAC if the recovery teacher beats its controls. Research/132-decision diagnostic
complete; no policy training or demonstrated new algorithm in this stage.

## SAC actor-action audit — 2026-09-17

[Bounded test and stop rules](performance/2026-09-17-dressing-actor-audit.md):
diagnostic actor guidance now includes tool-relative observation derivatives and
per-substep controller masks/finite rotations. Full cloth–arm friction history
and contact/observation branch changes remain outside the analytic chain. A
six-state executed finite-difference and action-improvement audit completed:
all six proposal types pass 0/6 gates; no actor training followed. Earlier body
14046's repeated SAC coverage differs by .06709, making complete restored-rollout
repeatability a prerequisite for that case. Geometry-based derivatives often
agree, but a useful, repeatable recovery and robust value guidance remain unproven.

## Dressing first-principles audit — 2026-09-17

[Current research and control implementation](performance/2026-09-17-dressing-first-principles.md):
fixed the missing replay-action anchor in the dressing IPC actor update, which
previously bypassed action-locality gating. Evaluation now records grasp-valid
success and controller-rejection traces. Full six-substep/controller derivatives,
stale critic labels, partial observability and useful elbow exploration remain
open. Existing replay supports bounded IQL/BC/replay-SAC controls; this is not a
new RL contribution or an established offline-to-online solution. The completed
two-round development comparison gives source/IQL/BC/replay-SAC coverage
.161/.135/.184/.223, with 0/4 successes each and substantial variability/grasp
failures. No robust advantage is established; all jobs finished. Only 31/2,400
labels in the historical IPC replay are within the endpoint actor's configured
action radius, making naive uniform replay effectively starve the corrected
physics term. Fresh same-action supervision and the complete executed/observed
derivative remain open. Keep further online experiments bounded and justified by dressing
failure evidence; do not restart the unrelated cloth-drag study.

## Dressing RL research update — 2026-09-16

The cloth-drag 40k study was stopped by the owner before completion; its interim
scores do not measure dressing. The positive 24k dressing pair is one seed with
an unstable SAC control. Open work is a reliable dressing comparison that isolates
physics proposal quality, actor fitting, and full-episode success at matched cost.
See the [redesign proposal](performance/2026-09-16-dressing-rl-redesign.md) for
implementation order, existing-data reuse, derivative limitations, and rejection
criteria. The [implemented local gate](performance/2026-09-16-dressing-verified-results.md)
accepted 0/8 IPC corrections. Preserving Adam and the existing replay now works
in controlled new branches; one 2,400-transition SAC continuation improved
two-cell development coverage .15164 -> .51295, with success still 0/2. The
matched IPC actor reached .36100, also 0/2, and used 592 s versus 391 s of training
time. Three fixed-seed endpoint rounds gave mean coverage .47347 versus .37301,
with 0/6 successes each on two repeated configurations. This short comparison has
sparse physics labels and still needs broader development cells and independent
training seeds before a general ranking. Native simulation dominates measured environment time; collision-candidate
search was 56% of the instrumented window. Larger homogeneous batches did not
improve throughput past eight slots, but two eight-slot MPS processes yielded
1.61x aggregate simulator throughput. A shared-policy asynchronous collector is
not implemented, and no novel-algorithm result is established. Do not
restart the stopped benchmark as a dependency of this work.

## Solver roadmap context

Status as of 2026-09-03. Completed performance work is
recorded in `handoff.md`; this file tracks what is **open** — analyze here first
before planning new work.

AL-IPC now has the fork-derived earliest-TOI active-set filter, decay-derived
pair lifetime, conditioning-aware penalty initialization, and a full-step inner
solve boundary. Its multi-state `K_min > 1` cumulative termination rule is now
implemented and validated on sample 88. Remaining paper-parity work includes
penalty-free moving boundaries and the specialized conflict-free analytic PSD
Hessian assembly. Do not claim complete parity with the reference simulator
until those paths and its large stress scenes are independently ported and
validated.

The uniform AL `diag_norm` penalty mode remains experimental. It caused
repeated non-descent line-search failures in mixed-resolution cloth/FEM sample
88 even after friction derivatives were corrected; `per_vertex` is therefore
the default. A future conditioning design should retain local mass/material
heterogeneity rather than broadcasting one global scalar.

## Cross-cutting source-audit findings

These previously verified gaps are closed on `refactor-main` as of 2026-08-25:

- The historical `baraff_witkin_shell.h` is now a compatibility include for the
  implemented `StrainLimitingBaraffWitkinShell`; its empty `.cpp` and other
  unreferenced zero-byte scaffolds were removed. A repository gate rejects new
  zero-byte files under `include/` and `src/`.
- `scripts/check_constitution_api.py` compares every exported constitution class
  with its pybind class name and verifies every binding initializer is registered.
  `RotatingMotor` and `LinearMotor` are both covered. Internal UID 27/28 remain
  intentionally internal and therefore outside the public-header contract.
- Every external GitHub Action is pinned to a reviewed full commit SHA. Both
  `johnwason/vcpkg-action` inputs and the Linux wheel container use the same
  immutable vcpkg commit as the generated registry baseline. The repository
  contracts workflow rejects floating action and revision refs.

## Current performance baseline

The current machine-specific reference is the
[2026-09-01 cross-domain baseline](performance/2026-09-01-cross-domain-baseline.md):
RTX 5090, CUDA 13.2, Windows Release, three fresh processes per scene, canonical
Timer-free throughput paths.

| Benchmark | Frames | Reference mean | Three-run range | Newton/frame | PCG/frame |
|---|---:|---:|---:|---:|---:|
| pure ABD wrecking balls | 120 | 129.5 ms | 126.4–131.3 ms | 3.95 | 107.3 |
| case2 FEM + cloth | 250 | 201.1 ms | 199.0–230.6 ms | 6.64 | 246.8 |
| MAS bunny | 100 | 60.2 ms | 60.1–62.0 ms | 4.67 | 358.8 |
| ABD wall + cloth | 100 | 125.6 ms | 121.8–165.0 ms | 5.13 | 200.1 |

All 12 throughput runs completed and converged without hitting iteration
limits. Collision-rich scenes have trajectory and WDDM/dynamic-memory
variability, so future work must compare three-run envelopes plus structured
iteration counts. The older Stiff-GIPC ratios below have not been refreshed
under this contract and must not be presented as current cross-project results.

### Superseded Stiff-GIPC comparison context

Measured on aligned scenes (same machine, clean runs; see handoff for the
full evidence chain):

- 6_wrecking_balls: libuipc 73.0 ms/frame vs Stiff 42.8-50.4 → **~1.5-1.7×**
- case2 (samples 88_stiff_gipc_benchmark): 301 ms/frame vs 142.8 → **~2.1×**

Root cause is NOT numerical parameters anymore (kappa, d_hat, eps_velocity,
semi-implicit exit are all aligned; Newton iteration counts match). The gap
is structural host/device overhead (nsys evidence, case2 stacking phase):

- ~7429 kernel launches + 522 memcpys + 1787 memsets + 563 stream syncs per
  frame (Stiff: ~900 launches) — **host-side API overhead ≈ half the frame**
- FusedPCG: 68.6 ms/frame = ~83 iters × 118 µs/iter, of which only ~36 µs
  is kernel compute — the rest is launch gaps (7 serially-dependent kernels
  per iteration) + a convergence D2H sync every 5 iterations
- BVH self-queries ~31.5 ms/frame: dense bunny surface → ~450k AABB
  candidates per detect, only ~3-8k survive distance filtering
- Contact assembly ≈ 26 ms; SNH G/H 2.12 ms/call ×7 (Stiff's equivalent:
  0.77 ms — 2.75×; `make_spd` 9×9 EVD is a suspect but removing it hurts
  convergence — see doc 08)
- Newton iterations per frame (found 2026-08-23 evening, case 89 parity
  run): Stiff averages **2.55 Newton/frame** while libuipc ran **7.04** —
  the old `newton/min_iter` doubled as a hard floor (≥6 with the benchmark
  configs), cancelling the semi-implicit early exit. Since then
  `min_iter` is a pure floor with default 0 and the beta-accumulation
  start moved to `newton/semi_implicit/K_min`. Semi-implicit termination is
  now enabled by default with `K_min=6`. Per-Newton
  solver time already beats Stiff on the same MAS bunny (≈39 ms wall vs
  48.8 ms GPU), so the frame-time gap on MAS scenes is dominated by the
  Newton count, not solver efficiency.

**Historical optimization ledger (completed or explicitly deferred):**
1. ~~Cooperative-groups persistent-kernel fusion or CUDA-graph capture of the
   PCG inner loop~~ **DONE (2026-08-23)**: FusedPCG now replays
   `check_interval`-sized iteration blocks as CUDA graphs
   (`linear_system/use_cuda_graph`, default on; details in doc 05).
   Remaining within-PCG cost is the ~83-iteration count itself.
2. ~~Fuse exact distance tests into BVH query predicates~~ **PARTIALLY DONE
   (2026-08-23 evening)**: the four DCD leaf predicates in
   `info_stackless_bvh_simplex_trajectory_filter.cu` now run the exact
   `distance::*_distance2` test when `alpha == 0` (DCD detect pass),
   keeping only `D2 < (d_hat+thickness)^2` pairs; the `alpha > 0`
   trajectory pass keeps the conservative `ccd_broadphase` (CCD superset
   requirement). Kills the 450k-candidate materialization + the
   overflow-retry double traversal on the DCD path; Detect DCD Candidates
   dropped 39 → 29 ms/frame on case 88. The trajectory/CCD pass
   (`Detect Trajectory Candidates`, ~38 ms/frame on 88) still uses the
   conservative swept test — tightening it needs a provably conservative
   exact swept-distance predicate, more delicate.
3. ~~SNH/FEM assembly kernel throughput~~ **DONE (2026-08-24)**: the SNH
   constitution was replaced wholesale by Stiff-GIPC's SNK1 (their energy
   `0.5μ(Ic-3) + 0.5λ(J-1-μ/λ)²`, their gradient, and their analytic
   twist/flip + 3x3-direct SPD projection — the generic 9x9 `make_spd`
   EVD is gone from this kernel; `make_spd` itself remains for the other
   constitutions). Case 88 median 297 -> 266 ms/frame; case 89 PCG
   213 -> 77 per solve (the analytic projection is also a much
   better-conditioned system). NOTE: the FEM energy changed numerically —
   trajectories shift slightly vs older baselines (89 resting centroid
   -0.775 -> -0.786) but stay physically equivalent. Same round:
   `cuda_tool::eigen::svd/pd` re-implemented on our own
   `algorithm/qr_svd.hpp` (GEIGEN port) — no more Eigen JacobiSVD on the
   host path, and the double overload no longer downcasts to float.
4. Same-address atomic storms (DONE 2026-08-23 evening): `Spmv_rbk_sym_
   spmv_dot_kernel` and `fused_dot_kernel` now do two-level (warp → block)
   reduction with one atomicAdd per block — ~9k/~4k same-address atomic
   doubles per call serialized ~20-30 us before. SpMV 114 → 93 us/call;
   case 88 median 312 → 297 ms/frame. NEXT within-PCG target: the
   symmetric-storage transpose scatter (~243k atomic vec3 per SpMV);
   a full-storage row-based SpMV would eliminate atomics entirely but
   doubles matrix traffic and needs a converter variant — deferred.
5. MAS assembly overhead (DONE 2026-08-25): the static mesh-partition
   hierarchy is cached after `init_matrix()` instead of being rebuilt for
   every Newton iteration; Hessian scatter now walks BCOO directly instead of
   filling an identity-index buffer; and the 48x48 Gauss-Jordan kernel drops
   the barrier between independent row updates. On RTX 5090, case 88 over the
   same 60 frames / 333 Newton iterations reduced `Assemble Preconditioner`
   1149.9 -> 834.4 ms (-27.4%, 3.45 -> 2.51 ms/Newton). The single-run wall
   mean moved 213.3 -> 208.1 ms/frame; contact-stage variance is larger than
   the remaining MAS contribution, so stage timing is the reliable metric.
6. Default BVH rebuild overhead (DONE 2026-08-25):
   `info_stackless_bvh` now uses CUB radix sort and scan with the existing
   persistent per-stream workspace. Identity generation and required state
   resets are fused into one kernel; redundant/dead fills are gone. Moving
   scene-AABB, leaf-LCA, and depth initialization before the multi-block
   consumers also removes cross-block reset races. Case 88, same 60-frame
   phase on RTX 5090: trajectory detection 7.98 -> 4.98-5.00 ms/Newton and
   aggregate DCD 7.35 -> 4.59-4.60 ms/detect (both about -37.5%); two wall
   runs measured 175.6-177.1 ms/frame versus 208.1 ms before.
7. DyTopo intermediate conversion (DONE 2026-08-25): a scene with one diagonal
   receiver owning the dynamic vertex prefix now forwards its raw assembled
   doublets/triplets directly to that receiver. The final global matrix
   converter performs the required sort/reduce already. Multiple receivers,
   off-diagonal ranges, and ABD/FEM coupling retain the classified path. On
   case 88, `Compute DyTopo Effect` fell from 6.11 to 4.76 ms/Newton (-22.0%);
   final `Convert To BCOO` stayed flat (1.85 -> 1.86 ms/Newton), and the clean
   wall mean/median improved 175.6/196.4 -> 165.5/184.2 ms/frame.
8. Historical case-88 frame budget at that revision/window (60 frames,
   165.5 ms mean representative):
   Build Linear System 43.4 ms + DyTopo 26.3 ms + trajectory detect 26.9 ms
   + aggregate DCD 24.5 ms + FusedPCG 24.8 ms + misc. The next evidence-led
   targets are raw contact/FEM subsystem assembly and the remaining
   line-search launch traffic, not another broad-phase or DyTopo sort rewrite.
9. Dynamic output initialization/growth (DONE 2026-08-30):
   `DeviceVector` now has explicit discard/preserve and amortized reserve
   policies. Fully regenerated collision, line-search, matrix-conversion,
   active-set, and triplet outputs no longer copy or value-initialize dead
   ranges when their logical size fluctuates. Existing `resize()` semantics
   are unchanged for state and sentinel buffers. On case 88, `Scan and
   Allocate` fell from 80.8 ms total over 60 frames to 38.0-65.5 ms, while
   the initial/trial `Compute Energy` scopes together fell from 667.1 ms to
   390.2-415.8 ms. End-to-end wall variance remains larger than this isolated
   gain. The next step is batching collision counter readbacks, not broadening
   discard semantics to buffers whose initialization contract is uncertain.
10. Collision count readbacks (DONE 2026-08-30): the default trajectory
    filter batches four broad-phase query counters into one D2H synchronization
    and batches the four PP/PE/PT/EE CUB selection counters into another. Queue
    overflow handling remains exact and reruns only affected queries. A fully
    active detect/filter cycle therefore uses two count synchronizations rather
    than eight. On case 88, trajectory detection changed from 5.07 to 5.01
    ms/Newton and aggregate DCD from 4.67 to 4.61 ms/detect; the structural
    synchronization reduction is larger than the noisy wall-time movement.
    The next host/device target is line-search energy aggregation.
11. Line-search energy aggregation (DONE 2026-08-30): ABD, FEM, and DyTopo
    reporters now publish totals to contiguous device slots. One final CUB
    reduction writes to a separate slot, followed by one contiguous D2H copy
    for reporter diagnostics and the aggregate. On case 88, initial/trial
    energy evaluations improved by 15.2%/17.3%, aggregate line search improved
    by 5.0% per Newton iteration, and wall mean/median moved from 162.7/182.3
    to 158.1/178.4 ms/frame. The next evidence-led target remains raw
    contact/FEM gradient-Hessian assembly.
12. Raw contact/FEM assembly (DONE 2026-08-30): SNH now projects its `9x9`
    material Hessian directly into ten `3x3` vertex blocks instead of forming
    dense `9x12` and `12x12` intermediates. Stack use fell 6440 -> 1320
    bytes/thread, the SNH kernel fell 1.795 -> 1.047 ms/call (-41.7%), and
    case-88 `Assemble Subsystems` fell 3.60 -> 2.97 ms/Newton (-17.6%). IPC
    contact keeps a single heterogeneous launch but compile-time specializes
    gradient-only versus Hessian work. A per-stencil split was measured and
    rejected: serial PT/EE kernels raised DyTopo assembly 4.52 -> 7.67
    ms/Newton despite smaller static stack frames. Final DyTopo assembly is
    4.47 ms/Newton, and two clean wall runs measured 156.0-157.0 ms mean /
    173.1-173.9 ms median.
13. Backend module boundary (DONE 2026-08-30): test and packaged builds now
    use the same shared-library artifact semantics. A required
    `uipc_query_module` handshake validates ABI version, libuipc major/minor,
    and backend identity before PMR synchronization or engine construction.
    This prevents stale/mixed backend DLLs from reaching the C++ virtual ABI.
14. CUDA build ownership (DONE 2026-08-30, corrected 2026-08-31 and
    2026-09-01): 198 compiled
    backend sources belong to seven primary domains and one optional
    legacy-collision component, followed by one final RDC device-link into the
    existing backend DLL. Matching CMake/XMake manifests reject unowned and
    multiply-owned sources. CMake uses internal OBJECT targets; XMake keeps the
    same logical partition but attaches sources to the final target because its
    device-link omits CUDA OBJECT dependencies.
    CMake's final target additionally owns one generated comment-only CUDA
    language anchor because Visual Studio otherwise omits the RDC device-link
    when all real CUDA sources arrive through OBJECT expressions. Functional
    source ownership remains unchanged.
15. Scene configuration ownership (DONE 2026-08-30): typed runtime defaults and
    machine-readable schema metadata now come from one declaration. The public
    normalized schema remains exactly equivalent, while future key additions can
    no longer silently update only one of the two former parallel lists.
16. SimSystem topology (DONE 2026-08-30): creator instantiation and every
    collection traversal now use one deterministic complete-type-name order;
    exact lookup uses `std::type_index`, compatible lookup skips invalid
    variants, and active strong-dependency cycles fail with an explicit path.
17. Legacy collision isolation (DONE 2026-08-30): the three alternate simplex
    trajectory filters have a dedicated optional component. Lean builds omit
    the sources and registrations, while the build-specific scene schema drops
    the unavailable selectors instead of accepting a configuration that can
    only fail later during backend system construction.
18. Test/performance entry points (DONE 2026-08-30): isolated sim cases support
    stable manifests and round-robin shards without replacing the aggregate
    pollution test; CTest GPU aggregates share a resource lock; the Stiff-GIPC
    case2 sample has a root-owned benchmark contract and revision-recording
    runner instead of another copied scene.
19. Decision/evidence retention (DONE 2026-08-30): accepted architecture now
    has numbered ADRs, performance work has an evidence policy/template and a
    case2 roll-up, and `handoff.md` is explicitly the chronological trail rather
    than the sole permanent home for rationale and measurements.
20. CI portability follow-up (DONE 2026-08-31): XMake CUDA sources now stay on
    the final shared target, which supplies the source-root include,
    backend-directory definitions, shared-library PIC behavior, and the one
    complete RDC device-link on both platforms. This replaces an intermediate
    OBJECT-target implementation that failed successively on missing includes,
    definitions, Linux PIC, and finally Windows device-link. The repository
    contract guards the final-target ownership requirement. In addition,
    repository-contract tests no longer confuse an intentionally unmaterialized
    samples submodule with an invalid benchmark declaration.
21. Thin-shell reference measure (DONE 2026-09-01): elastic and both plastic
    Discrete Shells paths retain the paper's complete `L0/h_bar = 3L0^2/A`
    metric and no longer multiply the adjacent area a second time. The stored
    thickness is consistently the one-sided collision radius `r`: formula-based
    bending uses full thickness `2r`, and Baraff-Witkin stretch uses `2r` while
    its separately calibrated shear coefficient remains thickness-independent.
22. QR-SVD float sign transfer (DONE 2026-09-01): the Wilkinson shift in
    libuipc, GPU_IPC, and Stiff-GIPC no longer calls the standard-library
    sign-copy function from host/device templates. It applies the sign with a
    branch in the original scalar type and defines `sign(0)=+1`; a libuipc CUDA
    regression instantiates and executes the float path on the GPU.
Every such change must re-pass the full sim suite (currently 95 cases / 14212
assertions).

## Deliberately deferred

- **CFL floor semantics** (`alpha = max(alpha, alpha_CFL)` in Stiff-GIPC):
  can push the step past the CCD hit point and needs crossing-based
  penetration detection (signed-distance + edge-face crossing, D=0 legal)
  as a backstop. libuipc's current filter asserts D>0 and would abort.
  Do that semantics change first, then the floor. (handoff: "Line-search
  pre-cap alignment")

## Dependency pins to unwind (standing debt)

| Pin | Where | Unwind when |
|---|---|---|
| `ports/tinygltf` overlay (SHA512 of regenerated v2.9.6 tarball) | `ports/`, `scripts/gen_vcpkg_json.py` | microsoft/vcpkg fixes the tinygltf port hash |
| `octree v2.5` | `src/geometry/xmake.lua` | xmake-repo layout stabilizes |
| `tinygltf <3` | `src/core/xmake.lua` | xmake-repo v3 include layout decided / our includes updated |

## Open issues

- **Published wheels through 0.0.27 need the CUDA 12 cuBLAS runtime; current
  source removes it**: package
  installation and `import uipc` succeed, but `Engine("cuda", ...)` fails on
  a CUDA 13.2-only machine because `uipc_backend_cuda.dll` directly imports
  `cublas64_12.dll`; the machine provides only `cublas64_13.dll`. This is not
  a missing bundled vcpkg DLL and not a driver-compatibility problem. Current
  workaround for 0.0.27: install CUDA 12.8 side-by-side and expose its `bin`
  directory, or build from current source. The source tree now replaces the
  remaining cuBLAS dot/norm calls with persistent raw-CUDA/CUB reductions and
  audits every wheel for dynamic Toolkit dependencies. Future wheels require a
  compatible NVIDIA driver rather than a local Toolkit: the base CUDA 12.x
  floor applies to packaged SASS, while PTX-only GPUs require the recorded CUDA
  12.8 JIT-driver floor. Remaining before
  calling the release-level issue closed: publish those wheels and add a
  GPU-capable CI job that constructs the CUDA engine; hosted binary inspection
  plus the no-GPU smoke test cannot prove actual driver/GPU execution.
- **CUDA-graph capture crash in the C++ suite binary (worked around)**: with
  Timer objects created inside the captured call chain, the single-process
  suite deterministically fail-fasted (0xC0000409) at the second engine's
  capture (never in isolation, never in python multi-engine repro). The
  capture path now creates no Timers. A second unresolved thread: al-ipc +
  graph capture crashed in the suite binary even without Timers (python
  al-ipc captures fine) — graph replay is gated to `contact/constitution ==
  "ipc"` until root-caused. Both symptoms point at some lurking global-state
  interaction with stream capture in the test binary; if someone revisits,
  start from `scripts/run_sim_case_isolated.py` + a binary-search over
  engine-count.
- **Performance work must start from the current four-scene baseline**: graph
  replay, DCD distance fusion, SNK1, discard-aware growth, batched readbacks,
  and direct FEM/contact assembly are already included. The synchronized
  diagnostic has no single universal hotspot: rigid is global-solve/assembly
  heavy, MAS bunny is linear-solve heavy, and case2/wall-cloth distribute cost
  across solve, line search, trajectory detection, DyTopo, and DCD. Profile the
  target scene before selecting another lever.

## External PRs under review

- **libuipc PR #461** (tcordeboeuf, EmbeddedCollisionMesh — barycentric
  coupling of a dense passive surface onto a coarse FEM tet mesh, SOFA-style).
  Direction is valuable, but as submitted it: (1) never writes
  `ecm_tet_geo_id` in `apply_to`, so the CUDA side silently no-ops —
  feature dead as written; (2) writes reporter gradient doublets
  conditionally → uninitialized slots get atomic-added into the linear
  system (corruption); (3) forward pass hooks outside the Newton loop →
  stale surface positions from iteration 1 on; (4) predates the muda→
  cuda_tool migration and the CBCOO gradient layout — needs a port. Not
  merged; a fix round was scoped and then declined for now. If revived:
  port to cuda_tool raw kernels, write `ecm_tet_geo_id`, fill all reported
  doublet slots, move the forward hook into the Newton loop, populate the
  frame-0 positions, add pybind + tests.
- **libuipc-samples PR #5** (hugooole, keyboard→imgui in case 4): reviewed,
  safe to merge; would only want `keyboard` dropped from requirements.txt.

## Security advisories

- pytest `< 9.0.3` tmpdir CVE → both metadata files require
  `pytest>=9.0.3`; `python/uv.lock` resolves 9.1.1. Dev-only dependency.
- usd-core `< 25.8` (critical) is marked fixed in the repo, but USD is a
  local/optional dep — upgrade any local install to `usd-core >= 25.8`.

## Samples submodule state (spiriMirror/libuipc-samples)

The root repository tracks this repository as the `libuipc-samples/` submodule.
It currently has 52 example directories; numbering is non-contiguous and two
directories use the `40_` prefix, so paths/names—not integer IDs—are the stable
reference.

- `87_robot_hand` — URDF robot hand (ABD links + soft transform
  constraints) + ABD cube on the ground, manual GUI posing (sliders +
  pose save/load). Ported from `references/Robotics-Libuipc` and rewritten
  in a different structure; assets renamed `leap_hand → robot_hand`
  (urdf `filename=` refs rewritten; link/joint names unchanged because the
  pose jsons key on them). The scripted auto-grasp was removed at the
  user's request (manual posing instead).
- `88_stiff_gipc_benchmark` — the Stiff-GIPC set_case2 benchmark with a GUI
  (default) and the original headless loop (`--headless [N]`); both modes
  write `traj.csv` + timing summary.
- `89_mas_bunny` — Stiff set_case7 parity (single MAS bunny, E=1e7);
  `NO_MAS=1` / `NO_GRAPH=1` env A/B switches.
- `90_abd_fem_cube_stack` / `91_pinned_cloth` / `92_twisting_bar` /
  `93_cube_wall_cloth` — the remaining Stiff-GIPC set_cases 1/4/5/6
  (2026-08-24). 92 uses animated SoftPositionConstraint ends (twist);
  93 has MAS on (case6 P_type=1) plus a 1920-body ABD wall. Assets added:
  `stiff_cube.msh` (Stiff's own 0.4-size cube — the samples' cube.msh is
  dimensionally different and overlaps at case1/6 spacings) and
  `high_mat.msh`. Both Stiff meshes were missing `\$MeshFormat` /
  `\$EndNodes` / `\$EndElements` tags and were completed for igl::readMSH.

## MAS preconditioner parity check (2026-08-23, case 89)

Cross-project comparison on the same scene (SNK bunny2, E=1e7, 100 frames;
libuipc samples `89_mas_bunny` vs Stiff-GIPC `set_case7`, both with MAS):

| run | total PCG | avg per solve | Newton/frame |
|---|---|---|---|
| Stiff MAS (P_type=1) | 60293 | 236 | 2.55 |
| libuipc MAS (mesh_partition) | 3500 | **5** | 6.0 |
| Stiff diag (P_type=0) | 376877 | 1513 | 2.49 |
| libuipc diag (NO_MAS=1) | 922185 | 1310 | 6.0 |

Verdicts:
- diag-vs-diag is same order (1310 vs 1513) — solvers/systems are comparable.
- ITERATION-COUNT CAVEAT (corrected 2026-08-23 evening): the 5/solve figure
  was an artifact of a graph-capture bug — the non-blocking capture stream
  let the un-plumbed MAS engine execute during capture instead of joining
  the graph, so replays ran with a frozen preconditioner (false r^Tz
  collapse). Fixed by using a blocking capture stream (un-plumbed callees
  now invalidate capture -> automatic fallback). After the fix, MAS averages
  ~41/solve vs diag ~86/solve on the E=1e4 bunny, with trajectories matching
  to 0.3mm at frame 100. The MAS preconditioner itself was never wrong.
- MAS + CUDA graph (closed 2026-08-23): the engine's `apply` path is now
  stream-plumbed (`MASPreconditionerEngine::apply(..., stream)` +
  `FEMMASPreconditioner::do_apply` forwards `info.stream()`), so MAS scenes
  join the captured graph instead of falling back to plain launches
  (graph_mode=1 confirmed active in MAS scenes). Validation at
  E=1e7/100 frames with contact: MAS+graph trajectory identical to the
  diag+graph reference at 1e-4 print precision; resting centroid -0.7753 vs
  the rigid-body geometric prediction -0.7731 (2mm compression); iteration
  counts real (frame-20 solves: MAS 5/20/155/435/495/410/505 vs diag
  2380/1625/880/1585/1650/1625/785 — MAS 3-4x fewer, no false-convergence
  signature). Free-fall no-contact: graph on/off bit-identical. One red
  herring cost a debug round: a "frozen" E=1e4 trajectory turned out to be
  a stale site-packages dll, not a solver bug (see doc 08 stale-dll trap).
  Measured benefit (89_mas_bunny, E=1e7, 100 frames, 704 solves/~198k PCG
  iters, identical iteration counts and f100 centroid on both paths):
  graph on 29.8s vs graph off 36.0s wall = ~17% end-to-end (~32 us/PCG
  iteration of launch-gap savings). For reference the same scene with the
  diagonal preconditioner (graph on) takes ~74s — MAS itself is the bigger
  win (~2.5x), the graph plumbing adds its share on top.
- COVERAGE RULE (measured on 88_stiff_gipc_benchmark, E=1e7, two 19k-vert
  bunnies + 4.2k-vert cloth, 60 frames, 2026-08-23): MAS only pays off when
  it covers nearly all of the stiff DoFs. Partial coverage is net NEGATIVE:  | config | total PCG | avg/solve | wall |
  |---|---|---|---|
  | all diagonal | 366875 | 853 | 60s |
  | MAS on upper bunny only (~45% of FEM verts) | 397730 | 923 | 73s |
  | MAS on both bunnies (~90%) | 79225 | 184 | 29s |
  f59 centroids identical in all three — physics unaffected; only the
  preconditioner spectrum changes. PCG converges at the rate of the
  worst-preconditioned block, so a diag-covered stiff block (the lower
  bunny in ground contact + the cloth) dominates and the MAS per-iteration
  overhead (cluster-inverse assembly + 5-launch apply) is pure cost.
  This table was measured with per-mesh tagging; it is the direct
  motivation for the all-or-nothing redesign below.
- ALL-OR-NOTHING REDESIGN (2026-08-23): MAS activation is now a scene
  config switch — `linear_system/fem_preconditioner = "mas"` (default
  `"diag"`). When on, `FEMMASPreconditioner::do_init` auto-partitions
  EVERY non-Empty FEM SimplicialComplex internally on a private clone
  (fixed cluster size = `BANKSIZE` 16, the only size the engine's shared-
  memory kernels accept); a pre-existing `mesh_part` attribute (custom C++
  partitioning) is still respected as-is, but manual tagging alone no
  longer activates MAS. The python `uipc.geometry.mesh_partition` export
  was removed (the exposed `part_max_size` was a footgun — any value > 16
  trips the BANKSIZE assert). Defensive fix included: switch on but
  nothing partitionable (e.g. all-Empty FEM) -> do_apply falls back to
  z=r instead of leaving z stale. Migrated: sim_case 53-61/81 + the
  stitch regression (config switch instead of mesh_partition calls;
  58_hybrid_mix now verifies custom + auto partitions coexist), samples
  88/89 (NO_MAS=1 env flips the switch to "diag").
- SCENE-ADAPTIVE KAPPA CORRIDOR (2026-08-24): Stiff-GIPC's
  `suggestKappa`/`upperBoundKappa` (`GIPC.cu:9850-9888`) is ported into
  `GlobalContactManager::init` with the dt^2 conversion (their barrier
  applies kappa raw, ours scales by dt^2 → computed values divided by
  dt^2). The corridor [suggest, 100×suggest] now rules all kappa clamping
  (default-model resolution, per-model clamps, the adaptive strategy's
  projection); `contact/adaptive/{min,max}_kappa` are fallback-only when
  the corridor is not computable. The evaluation scale is user-configurable
  via `contact/adaptive/kappa_eval_scale` (default 1e-16 — Stiff's value;
  the /dt^2 conversion keeps it valid for any dt). Measured corridors:
  case 89 → [4.4e7, 4.4e9], case 88 → [4.7e6, 4.7e8]; the established
  κ=1e8 sits inside both, so aligned scenes are behaviorally unchanged.
- Structural reason libuipc's port is stronger: FEMMASPreconditioner
  rebuilds the cluster inverses from the current full diagonal Hessian
  (kinetic + material) every Newton iteration
  (`fem_mas_preconditioner.cu` `set_preconditioner(A.values(), ...)`).
- Not ported by design: Stiff's collision-aware clustering
  (`_buildCollisionConnection`, deliberately skipped per owner decision)
  and the size-specialized Schwarz kernels (`_schwarzLocalXSym3/6/9`,
  `__inverse6_P96x96`) — the latter is a throughput optimization lead.
- Stiff-GIPC local benchmark edits (uncommitted, local only): `set_case7`
  in `gl_main.cu` (single bunny, E=1e7, P_type=1), a 100-frame cap writing
  `timeCost.txt` (totalNT/totalCgCount), and a `GIPC_PTYPE` env var to flip
  the preconditioner type.


## Dressing force learning and elbow passage (2026-09-12; status updated September 13)

The [historical force-training research](performance/2026-09-12-force-training-research.md)
archives literature and raw elbow evidence against `4bfe88c2`. Its proposed per-decision force
experiments are superseded by the failed reliability gate and closure in `024c5716`.
Retain the readout and diagnostics; do not treat these reports as authorization to enable force
actor/critic/reward inputs. Force-conditioned and diffusion methods remain established baselines,
not a demonstrated novel contribution. The newer dense critic needs a fresh training comparison.

## Sequence pretraining proposal (2026-09-13)

See the [implementation specification](performance/2026-09-13-recurrent-pretraining-proposal.md).
First remeasure the corrected dense critic baseline. Then add optional episode identities and
sequence sampling without changing legacy flat replay. Separate episode resets from bootstrap,
keep independent train/eval states, and preserve dense action gradients and target-memory updates.
H4/H8 and GRU are controls before RLT-inspired attention; no architecture gain is established.

## RLT review and repaired experiment contract (2026-09-13)

The [independent review](performance/2026-09-13-pretraining-infrastructure-review.md) supersedes
claims that the original H8 all-position run isolates architecture or that the old probe measures
full SAC cost. New RLT runs default to endpoint learning; prefix remains a named different
objective and old checkpoints retain it. Required next work: versioned trajectory corpus and
target normalization/offline training, training-only valid guide prefixes and elbow roll-in,
GRU/parallel-Transformer controls, then measured episode-memory/cache designs. A streaming RLTState
is not an equivalent optimization of rolling H8; actor features cannot replace dense Q(o,a).

Status, later on 2026-09-13: the offline trainer exists
([record](performance/2026-09-13-offline-pretraining.md)); the corpus does not — the first run with
`--sequence-replay --record-privileged` is the first usable data.

## IPC Adjoint Q-Learning proposal (2026-09-14)

[ADR 0008](adr/0008-ipc-adjoint-q-learning.md) specifies an opt-in critic
derivative loss, complete-decision implicit sensitivities, paired soft targets,
replay refresh, and a `cloth_drag`-first evaluation sequence. MAGE and
First-order Sobolev RL are direct precedents; the candidate research extension
is calibrated contact/direction reliability. Current retained projected Newton
matrices and last-frame physics actor signals do not implement the proposed
Bellman derivative. Next work is the target/loss contract and complete-decision
benchmark adapter, followed by frozen-critic and matched SAC experiments.
The [benchmark prototype](performance/2026-09-14-iaql-benchmark.md) has
paired target refresh, mixed-derivative critic regression, native full-state
cloth collection, a bounded mechanics sidecar over a full TD replay and a CPU
refit phase. Measured: the gate passes at friction 0 (4/4), the derivative
loss raises the held-out slope cosine 0.38 → 0.84 while shuffled labels do
not; on 100 frictionless states the raw Hessian re-assembled at the accepted
state (`set_export_mode('converged_raw')`) cuts the median tangent error from
4.3 % to 0.20 % and removes its growth with drag, while re-assembling with
projection changes nothing, so the SPD projection was the whole error;
friction 0.6 failed the gate (1/4) for want of the lagged friction terms;
with the raw Hessian and the lagged coupling the same re-assembly now
exports (`export_prev_coupling`, half-plane only) the frictional tangent is
0.14 % median and the gate passes on every mechanically resolved state. The
benchmark became learnable once the GPU was free and the world held 64
lockstep cloths with the tangent on the GPU: at 20k transitions the actor's
direct use of the exact label (estimator replacement, ρ 0.5, continuation
trust) beats SAC in two of three seeds and loses one (not significant, with
the term's coverage collapsing after 5k transitions); the critic slope loss
alone adds nothing. The five-seed study of this configuration and of the
fresh-batch protocol (stopped by the owner at 35k of 40k transitions) fails the
pre-registered rule: the label wins 1 of 4 seeds against fresh-batch SAC
(−0.11 ± 0.35) and 2 of 4 against replay SAC (+0.09 ± 1.33), trails SAC at 10k,
and SAC alone reaches 59.5 of 64 successes. The IPC-label SAC updates are
closed; the mechanics, the multi-slot environment and the GPU tangent stay.
Open on the mechanics side only: simplex (cloth-body) friction
coupling, a stick-slip-switching state, a learnable version of the
benchmark (vanilla SAC must show a curve before any comparison), calibrated
directional contact trust, and visual transfer.

Update 2026-09-15: the [fresh actor implementation](performance/2026-09-15-fresh-ipc-actor.md)
separates actor mechanics from TD replay, fixes action coordinates, and adds
matched controls. Native smoke passes; multi-seed learning validation remains.
Dressing launchers now default to dense/residual with legacy resume preservation.
Next pretraining experiment: observed responses versus IPC-derived symmetric
counterfactuals versus explicit Jacobian supervision, on episode-held-out data.

The [counterfactual response pilot](performance/2026-09-15-counterfactual-response.md)
is implemented and evaluated on real IPC cloth_drag data, including friction
shift. Normalized differences improve learned response slopes, but frozen
encoder and value probes have not passed the gate for dressing RL. Remaining:
real visual/history data with tracked response correspondence, varied contact
regimes and garment/human holdouts. Constant-tangent and command-history baselines
must accompany future representation claims. The old replay actor also loses
seed 2 despite winning seeds 0/1; fresh-protocol learning curves are still needed.

2026-09-16: `pretrain_wang joint` removes the regional-teacher dependency for
one-policy multi-region training. [Status/plan](performance/2026-09-16-single-policy-plan.md).
Teacher-free generalization and visual dressing IPC integration are unvalidated.
Sparse fresh correction must precede replay actor steps; target-action regression
alone does not bound actual policy movement. Preserve replay on resumes and
separate update frequency from gradient mixture strength in future comparisons.

2026-09-16 actor-interface follow-up: [native results](performance/2026-09-16-actor-interface-results.md)
show normalized target fitting is first-step equivalent to the linear surrogate,
not a remedy for actor Jacobian/optimizer effects. Inherited Adam moves even for
a zero target gradient; an actual-action bound is separate from target radius.
Fresh optimizer state plus a bound improves all six probed batch means, but
ongoing optimizer evolution, matched hybrid learning and visual dressing remain
unvalidated. No long training was relaunched and no production SAC default changed.

2026-09-16 dressing trajectory generation: the [route pilot](performance/2026-09-16-dressing-route-pilot.md)
prepares a bounded comparison of the original expert, height tuning, and an
outward elbow route on training poses. CPU checks pass; native execution is pending
GPU capacity. Strategy synthesis, recovery, fresh action replay, sequence-corpus
conversion, and policy distillation remain open. No physical improvement is claimed.
