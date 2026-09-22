# Rules — Owner-Mandated Working Agreements

Rules the project owner (the user) has explicitly laid down. They override
habit and generic best practice. When one applies to your task, follow it;
when a rule is marked task-scoped, do not generalize it.

Standing rule for this file: **whenever the owner sets or changes a rule,
record it here in the same commit.**

## Change discipline

1. **Commit to the current working branch only.** Never open a PR to (or
   push to) the default branch (`main`) unless the owner explicitly says
   so. (Set 2026-08-23; current working branch at the time:
   `refactor-main`.)
2. **Commit in stages, push without asking.** During multi-step work,
   commit each completed stage separately (clear rollback points) and push
   directly — no per-commit confirmation. (Set 2026-08-23.)
3. **Minimal diffs**: make the smallest change that achieves the goal;
   reuse the project's existing algorithm/utility code instead of
   introducing parallel implementations; no opportunistic refactors,
   renames, or cleanups outside the task's scope.
4. **Keep agent_docs current in the same change.** Any update that changes
   structure, workflow, conventions, build, or behavior must update the
   matching doc (and `handoff.md` / `09-known-issues-and-roadmap.md` when
   it closes or opens work) **in the same commit**.

## Language

4. **English only in repo artifacts**: documentation, code comments, commit
   messages. (The conversation language stays whatever the owner uses.)

## Documentation scope

5. **Demo showcases belong on the project homepage, not in the technical
   documentation.** Keep video/demo catalogues in the root README or project
   website; keep `docs/` focused on tutorials, reference, specifications,
   build guidance, and development documentation. The documentation must
   expose a global link back to the project repository. (Set 2026-08-24.)
   Scene-configuration reference material must state every registered key's
   default, unit, effective-value rule, and valid domain/selector where one is
   defined. Representative simulation tutorials must cover rigid bodies, FEM,
   cloth, rigid-soft coupling, and contact in both C++ and Python, verified
   against `libuipc-samples`, public APIs/bindings, and the consuming backend
   implementation. Educational examples are documentation; showcase/demo
   catalogues remain homepage-only. (Expanded 2026-08-24.)

## Build system

6. **clang-format-18 before every C++ commit/PR**: run
   `python scripts/format_changed.py` (formats the C++ files changed vs the
   base ref, CI's exact selection) or `--check` for a dry run. CI rejects
   non-conforming diffs; the pinned local formatter is
   `output/venv_clangfmt/Scripts/clang-format.exe` (18.1.8, same major as
   CI's clang-format-18).
7. **CMake and XMake stay in sync**: every build-affecting change (sources,
   flags, dependencies, pins) must be applied to both build systems, even
   if only one can be verified locally.
8. **No ccache.** It was integrated once and the owner had it reverted; do
   not re-introduce it (or similar compiler-cache layers) without an
   explicit request.

- **Use only the embedded C++ METIS target.** Mesh partitioning must use
  `src/geometry/metis/` through `uipc_metis`; do not restore separate
  `external/METIS` or `external/GKlib` source trees or targets. Keep the CMake
  and XMake source boundary equivalent. (Set 2026-09-03.)

## CUDA backend

9. **Raw kernels only**: business GPU code is written as named
   `__global__` functions with `<<<>>>` launches — no lambda kernels, no
   ParallelFor-style wrappers.
10. **`cuda_tool` is the only device utility layer and must not contain
   muda** (no vendored copy, no submodule, no compat header). Keep
   cuda_tool minimal: no primitive without an in-tree user.
11. **Eigen stays** as the host/device small-matrix dependency. (A
   replacement experiment — geigen under `src/math/` — was tried and
   deliberately reverted; do not resurrect it without an explicit request.)

12. **Use CUB, not Thrust, for new or modified CUDA bulk algorithms.** CUB
   temporary storage must be persistent and reused (normally through
   `cuda_tool`'s per-stream workspace); never allocate/free CUB scratch on
   every call. (Set 2026-08-25.)

13. **Make CUDA buffer growth semantics explicit.** Keep `resize()` for
   value-initialized state. Use `resize_discard()` only for output that the
   following kernel/copy/CUB call completely regenerates, and use
   `resize_preserve()` only when the old logical range must survive but the new
   range may remain uninitialized. Growth capacity must be based on the latest
   requirement with deliberate headroom; do not copy or initialize dead output
   ranges merely to emulate `std::vector`. (Set 2026-08-30.)

14. **Profile heterogeneous kernel fusion/splitting end to end.** Static
    register or stack reductions do not prove a speedup. In particular, keep
    IPC simplex PT/EE/PE/PP Hessian assembly in one launch unless both the
    kernel profile and enclosing DyTopo timer prove a split is faster: rare
    PT/EE threads are expensive and overlap the dominant PE population in the
    fused launch. Compile-time specialization of a uniform mode such as
    `gradient_only` is preferred when it removes dead work without serializing
    stencil families. (Set 2026-08-30.)

15. **Use pipeline-specific parallel-EE handling.** AL-IPC never classifies
    parallel EE pairs: both normal and frictional contact use the shared
    negative disabled threshold, making `need_mollify()` false and routing all
    AL pairs through their ordinary EE paths. Standard IPC uses the positive
    `1e-3` coefficient. Its normal-contact path evaluates the complete
    mollified energy, gradient, and Hessian for detected parallel pairs; its
    friction path skips detected parallel pairs. Do not make `need_mollify()`
    globally constant. (Set 2026-09-03; refined 2026-09-03.)

## Task-scoped (recorded for context, not general policy)

- **Selected task: dress both arms and the torso of a seated person.**
  The owner explicitly abandoned the single-human-arm research direction and
  selected dressing a person sitting in a chair with both human arms raised
  forward. The garment must be put on the body; two sleeve insertions alone
  do not define completion. Do not substitute one sleeve manipulated by two
  robots, a standing recipient, or a robot-commanded human arm trajectory.
  Research concrete implementation and deployment, starting from the owner's
  PA-BiCoop paper (https://arxiv.org/pdf/2606.28192) and broader primary sources.
  The owner explicitly permits a non-RL approach; the older new-RL-algorithm
  requirement is no longer mandatory for this task. Garment type and initial
  presentation are not yet specified; distinguish proposed assumptions from
  owner decisions. Continue the research-only and no-teacher/student
  constraints below. (Supersedes the single-arm priority, 2026-09-22.)

- **Research only; abandon the teacher/student route.**
  The owner explicitly instructed: no more tests, abandon the Yufei-inspired
  teacher/student idea, and develop a new way to solve the task. Continue
  literature review, reasoning and inspection of existing evidence without
  asking again. Do not run tests, simulation, training, or restart Stage 0 under
  this research request. Do not reintroduce the rejected route through teacher
  goal imitation, distillation or DAgger. Preserve existing data/code; SAC and IPC are not design
  requirements. This supersedes earlier experiment authorizations for this
  research task. (Set 2026-09-22.)

- **Do not anchor the new algorithm to SAC or IPC.** The owner explicitly
  rejected preserving the current pure-SAC pretraining structure and making
  IPC-specific modifications the research contribution. Develop the learning
  mechanism independently of that architecture and simulator. Existing data,
  completed results and benchmarks remain useful evidence; this instruction
  does not request deleting code/data or terminating unrelated processes.
  A renamed combination of established components is not a novelty claim.
  (Set 2026-09-20.)

- **Research a replacement RL algorithm, without preserving the current learner.**
  The owner clarified that the goal is a new dressing RL algorithm, rather than
  the existing pretraining/RL structure, and asked whether first-order/gradient
  methods are the wrong approach. The owner explicitly requires independent
  assessment rather than following the proposed no-gradient explanation. Treat
  it as a competing hypothesis, not an established result or selected solution.
  Reuse data and simulation where useful; the earlier
  task-specific requirements to preserve SAC/pretraining interfaces do not
  constrain this replacement-algorithm research. Distinguish a proposed operator
  from validated novelty or a measured policy improvement. (Set 2026-09-20.)

- **Run the bounded PointZero-inspired pretraining experiment.** The owner
  approved implementing and testing cloth-motion representation pretraining
  using existing IPC trajectories, with matched RL controls. This authorizes
  necessary track preparation, encoder pretraining and bounded policy training/
  evaluation. Keep the RL interface and reward fixed; distinguish this small
  adaptation from a PointZero reproduction or a novel RL algorithm.
  (Set 2026-09-18.)

- **Start the dressing pretraining experiment.** The owner authorized training
  and selection of the best suitable offline-pretraining method from Sergey
  Levine's work. This authorizes implementation, necessary data preparation,
  bounded training and dressing evaluation; prior instructions to keep new
  training stopped no longer block this experiment. Do not restart abandoned
  IPC-gradient corrections or claim a universal best method from benchmark
  rankings alone. (Set 2026-09-17.)

- **The dressing goal is fast, easy training on one workstation.** The owner
  stated that the reference dressing pipeline (per-region reinforcement-learning
  teachers, then distillation) is hard to scale and train and needs a cluster;
  the new direction must train a policy fast and easily without one. Judge
  candidate methods by total wall time on the single-GPU workstation, data
  preparation and evaluation included, not by update speed alone.
  (Set 2026-09-17.)

- **Research a new direction after the SAC audit.** Following the completed
  ordinary SAC rollout analysis, the owner again instructed the agent to give
  up the previous idea and deeply research alternatives for dressing. Use the
  measured SAC findings as evidence, keep abandoned IPC correction experiments
  stopped, and distinguish a proposed contribution from an implemented or
  validated algorithm. The earlier diagnosis-first ordering has been satisfied;
  it must not prevent the requested research assessment. (Set 2026-09-17.)

- **Diagnose ordinary SAC rollouts before a replacement research proposal.** The
  owner abandoned the IPC algorithm direction, requested a new direction, then
  clarified twice that actual normal SAC policy results must be analysed first.
  BC or IPC-modified policy failures cannot stand in for SAC failure evidence.
  Keep the abandoned experiments stopped and distinguish measured rollout
  failures from hypotheses about their learning causes. This supersedes the
  following IPC-integration task instruction. (Set 2026-09-17.)

- IPC recovery guidance belongs inside RL pretraining; preserve the existing
  downstream policy/training interface rather than treating standalone imitation
  as the requested RL integration. The owner subsequently clarified that adding
  an established imitation loss to SAC does not meet the novel-algorithm goal;
  research and distinguish the actual algorithmic contribution before claiming
  novelty. (Clarified 2026-09-17.)

- During the Stiff-GIPC performance-alignment work: "when a design choice
  is uncertain, follow Stiff-GIPC's algorithm design" — that applied to
  that effort (and its evidence lives in `handoff.md`), not to unrelated
  features.
- During CI watch duty: "you confirm all operations yourself" — scoped to
  that autonomous monitoring task; normal confirmation rules apply
  elsewhere.

- IPC/dressing pretraining experiments use **dense action-conditioned Q plus
  a normalized residual trunk** as the chosen baseline. Keep dense Q's dressing
  paper provenance separate from residual/normalization design; preserve saved
  architectures on resume and retain latent/plain as explicit ablations.
  (Set 2026-09-15.)
