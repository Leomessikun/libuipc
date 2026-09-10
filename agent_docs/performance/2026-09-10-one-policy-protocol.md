# 2026-09-10 — Training one dressing policy for many garments and many bodies

- Status: Accepted and implemented in the trainer (cell planning, held-out bodies, per-cell
  evaluation, generalisation-first checkpoint score, resume refusal); live cells build for
  tshirt_26 and tshirt_4 on generated bodies, the other two garments are open
- Scope: the `python/uipc_manip/` training and evaluation protocol; no libuipc solver change
- Sources: Wang RSS 2023 in `/home/ge47gax/kun/dressing/curl/`, the Newton port's
  `dressing/docs/PIPELINE.md` and `dressing/training/`, and this port's own measurements in
  `2026-09-09-dressing-correctness.md` and `2026-09-08-uipc-manip-pretraining.md`


Paths are absolute. Reference = /home/ge47gax/kun/newton-fmvp/... and
/home/ge47gax/kun/dressing/curl/... . Port = /home/ge47gax/kun/libuipc/python/uipc_manip/.

## 1. What the reference does

### 1.1 Wang RSS 2023 (the source protocol)
- Distribution: 5 garments x 50 poses. `launch_train_curl.py:270` lists
  `['hospital_gown','tshirt_26','tshirt_68','tshirt_4','tshirt_392']`;
  `:429-433` sets `train_poses = range(0,45)`; `:436` sets `eval_poses = range(45,50)`.
  225 training configurations, 25 held-out. The split is on the POSE axis only;
  every garment appears on both sides.
- One environment, resampled per episode. `train_multi_garments.py:414-433`:
  on `done`, draw `garment_id`, `env.set_garment(...)`, draw
  `pose_id = train_poses[randint]`, `config_idx = pose_id*num_of_garments+garment_id`,
  `env.reset(config_id=config_idx)`.
- Curriculum: `:369-371` `curriculum_step = min(num_of_garments, step//curriculum_update_freq + 1)`;
  `:416-421` the next episode's garment is drawn only from the first
  `curriculum_step` entries of the garment list. Optional weighted sampling
  `:362-367`. Evaluation is also truncated to `curriculum_step` garments (`:87-89`).
- Replay: one buffer per garment by default (`:336-338`), and each update draws a
  random garment's buffer (`:455-467`).
- Evaluation round: `:109-128` iterates garments x eval poses, deterministic
  (`run_eval_loop(sample_stochastically=False)` at `:230`), one episode per
  (garment, pose) pair, i.e. 25 episodes. `:350-351` `num_eval_episodes = len(eval_poses)`.
  Optionally re-scores a random subset of training poses too (`:93-99`).
- Best checkpoint: `:393-395` keeps `best_test` on `avg_upperarm_ratio_unseen`,
  the mean upper-arm dressed ratio over the UNSEEN poses; `:389-391` keeps a
  separate `best_train`. Unseen is the primary artifact.
- Time limit is not terminal: `:479` `done_bool = 0 if episode_step+1 == env.horizon`.
- Cadence: `eval_freq = 10000` env steps (`launch_train_curl.py:440`),
  `num_train_steps = 5,000,000` (`:220`).

### 1.2 Newton FMVP port (the batched adaptation)
- Cells to slots: bound for the life of the world.
  `dressing_env.py:1956-2013` `_assign_static_config_ids` pins the held-out grid
  to the leading slots (`pinned_eval_human_ids`), then stratifies the remaining
  cells BY GARMENT and spreads bodies inside each garment, because block-uniform
  assignment over the flat accepted-config list gave 12/12/2/4/10 slots.
  `train_pointcloud_sac.py:1405-1412` reads them back as `garment|human=N` labels.
- No per-episode resampling. PIPELINE.md:74-83 and `config.py:269-290`:
  "Newton binds each batched slot to one garment for the life of the environment,
  so the schedule gates which slots may write to replay instead of resampling a
  garment per episode."
- Curriculum gating: `train_pointcloud_sac.py:2043-2045` builds `_slot_garment_rank`
  from the curriculum order; `:2209-2220` computes `_active_garments` and
  `valid_training_rows = training_slot_mask & (_slot_garment_rank < _active_garments) & ~sim_error_np`;
  `:2222-2231` only those rows call `online_buffer.add`. Order is measured
  reachability, easiest first (`config.py:252-266`: gown .742, t26 .670, t4 .461,
  t68 .375, t392 .211) - deliberately NOT Wang's launcher order.
- Held-out choice: `config.py:48-78` `select_complete_heldout_human` returns the
  GREATEST body id that carries every requested garment, so the reserved row is
  deterministic and garment composition is not confounded with the pose split.
  `train_pointcloud_sac.py:1425-1433` applies it automatically under
  `--fmvp-pretrain-defaults`; `:1455-1465` refuses an incomplete held-out row;
  `:1490-1498` splits slots; `:2005-2018` recomputes the mask after a rotation;
  `:1535-1549` converts `--total-transitions` using TRAINING cells only.
- Coverage invariants: `:1509-1516` formal training must cover every cached cell;
  `:1562-1570` evaluation episodes are raised to at least one per cell.
- Evaluation round: `:752-816` gives every slot an explicit quota
  (`target_counts`), labels it `garment|human=N`, and refuses to count a second
  episode from a fast slot while a slow one is unmeasured. `:864-902` records
  per-episode outcomes and per-cell success. `:926-949` reports
  `fmvp_success_rate`, `strict_success_rate`, `worst_cell_strict_success_rate`,
  ... `:951-979` `add_subset` emits `heldout_*` and `training_*` copies of every
  rate. One deterministic episode per cell: 35 envs, 35 episodes.
- Best checkpoint: `:2370-2378` `policy_checkpoint_score(result)`.
  `eval_metrics.py:16-52` prefixes every key with `heldout_` when
  `heldout_episode_count > 0`, ranks `fmvp_success_rate` (final ratio >= 0.7)
  first, then strict/topo/progress/threading/ratio/return, and only then falls
  back to all-cache keys "without allowing training-cell success to outrank
  generalization".
- Resume pinning: `:1040-1051` refuses a `--heldout-human` that differs from the
  checkpoint; `:1767-1783` refuses a cell list that differs from the checkpoint.
- Cadence actually used (`runs/fmvp_sac_multiscene/2026-08-18_00-53-11_fmvp_stage1_wang_noforce_heldout_h7_seed1/params/train_args.json`):
  `num_envs 35, heldout_human [7], total_transitions 5,000,000, eval_freq 2000,
  num_eval_episodes 35, checkpoint_interval 2000, updates_per_step 30, batch 64`.
  Regional teacher (`runs/fmvp_regional_teachers/2026-08-19_17-51-44_region_h6_curriculum_seed1/params/train_args.json`):
  `regional_teacher 6, garment_curriculum_interval 4000, total_transitions 2,000,000,
  eval_freq 4000, updates_per_step 35`.
  PIPELINE.md:259-264: evaluation + snapshots are held near 15% of wall clock.
- Rotation: `:2382-2394` every `--config-rotation-interval` steps, re-draw the
  training slots' cells (held-out slots frozen), refresh the mask, reset. It is a
  re-draw of which cached config a slot runs, NOT a world rebuild.

## 2. What this port does today, and the gaps

Measured state (branch cloth-cable-manip-rl):
- Cell library actually loaded: `dressing_env.py:194-197`
  `garments = [g for g in cfg.garments if (g, int(cfg.human)) in self.cache.cells]`
  then `self.cells = [cache.load(garments[i % len(garments)], cfg.human) for i in range(num_envs)]`.
  One body, garments round-robin. `train_sac.py:43` `--human` is a single int.
- Current run `output/uipc_manip/sac_dressing_h150x6_softcloth_seed1/config.json`:
  16 envs, garments `[tshirt_26, tshirt_392]`, human 0, horizon 150 x repeat 6,
  eval_freq 600, num_eval_episodes 16, curriculum interval 600.
  `eval_log.csv`: best round step 1800 -> success 0.50, tshirt_26 1.00,
  tshirt_392 0.00, early_turn_rate 1.00 in every round (paper_filter_rate 0.00).
- Baked cache inventory (measured):
  23 cells over bodies 0-7; per garment gown 8, tshirt_392 8, tshirt_26 4,
  tshirt_68 3. Bodies carrying all four of
  {gown, t26, t392, t68}: only 0, 2, 6.
- `dressing_live.LiveCellFactory` can build any (garment, body) but is not
  referenced anywhere outside `tests/test_dressing_live.py`.

| # | Reference behaviour | Port today | Gap |
|---|---|---|---|
| G1 | Cell axis is (garment, body) | one body for the whole world (`dressing_env.py:194-197`, `train_sac.py:43`) | no body axis at all |
| G2 | Per-slot collider is that slot's body | every slot gets `cell0`'s eroded arm (`dressing_env.py:211-213`, `246-247`, `271`) though it is NAMED `arm_human_{cell.human}_{env_idx}` (`:278`) | multi-body is silently wrong until fixed; this is the blocking bug |
| G3 | Deterministic held-out row, complete in garments (`config.py:48-78`) | none | evaluation scores only trained cells |
| G4 | Replay masked by held-out slot (`train_pointcloud_sac.py:2220`) | mask is curriculum-only (`train_sac.py:567-568`) | no way to exclude a slot from replay |
| G5 | Budget converted over training cells (`:1535-1544`) | `total_transitions` already counts only admitted rows (`train_sac.py:546`, `571-572`) | NOT a gap - semantics already match |
| G6 | One update per valid transition | `--updates-per-step 0` -> `gradient_update_budget` (`train_sac.py:579-585`) | NOT a gap |
| G7 | Per-cell eval quotas + `garment\|human=N` labels (`:803-816`) | per-slot quota exists (`train_sac.py:265-267`) but records carry only `garment` (`:305-306`) | cannot tell an unseen body from a hard garment |
| G8 | `heldout_*` / `training_*` / `worst_cell_*` subsets (`:942`, `:951-979`) | `success_rate_{garment}` only (`train_sac.py:336-343`) | summary cannot express generalization |
| G9 | Score ranks held-out first (`eval_metrics.py:16-52`) | `checkpoint_score` = (success, -distance, max_ratio, return) (`train_sac.py:385-392`); `-mean_final_distance` is redundant, `distance = 1 - upperarm_ratio` (`dressing_env.py:471`) | a memoriser can win |
| G10 | Coverage invariant: cells <= slots, eval episodes >= cells (`:1509-1516`, `:1562-1570`) | none | a run can silently train on 2 cells |
| G11 | Checkpoint pins held-out ids and cell list, resume refuses a mismatch (`:1040-1051`, `:1767-1783`) | `metadata["training_args"]` carries curriculum/updates/replay/init only (`train_sac.py:514-516`) | a resume can train on a previously held-out body |
| G12 | Curriculum order = measured reachability (`config.py:252-266`) | `WANG_GARMENT_ORDER` = Wang's launcher order (`curriculum.py:14`) | not a bug; document which one and why |
| G13 | Cells resampled/rotated | `reset()` is `_world.recover(_snapshot_frame)` (`dressing_env.py:406-426`, snapshot dumped once at `:343-345`) | per-episode resampling is unavailable; see 3.1 |
| G14 | FMVP trajectory filter usable | `early_turn_rate = 1.00` in all four eval rounds incl. the 0.50-success one, so `paper_filter_rate` is a constant 0 (`train_sac.py:300`, `dressing_reward.py:218-245`) | Stage I-B filtering is dead; needs a validation before it gates anything |
| G15 | - | `_BODY_TREES` holds one entry and `clear()`s on every miss (`dressing_reward.py:62-67`); `cache.load` returns a fresh `human_points` per cell (`dressing_assets.py:146`) | cache thrashes: measured cKDTree build 1.45 ms x 16 = 23 ms/decision, which is exactly the 24.7 ms the perf record attributes to "all sixteen rewards". Harmless (0.8% of a 3.06 s decision at 32) but must be keyed per body before 32 distinct bodies |
| G16 | - | `DEFAULT_GARMENTS` is 3 names in `dressing_env.py:36` and 4 in `dressing_assets.py:27` | launcher default disagreement |

Two facts that are already right and must be preserved:
- The port's `success` IS FMVP's own criterion. `dressing_env.py:468` reads
  `upperarm_ratio >= success_upperarm_ratio` at the step reported, episodes end
  only at the shared time limit (`:459`, `:488`), so the flag at `done` is the
  ratio the trajectory ENDS with. That is `fmvp_success_rate`, not a max.
- Time-limit bootstrapping is Wang-equivalent: `dressing_env.py:489-491` stashes
  `terminal_obs` before the auto-reset and `train_sac.py:570-571` stores it with
  `done=False`.

## 3. The plan, as edits to named functions

### 3.0 Where this port must differ from the reference, and why
1. **A cell is bound to a slot for the life of the WORLD, not merely of the
   environment.** `GenesisIPCDressingEnv._prepare_start` (dressing_env.py:328-346)
   settles once and `_world.dump()`s a single snapshot; `reset()`
   (`:406-426`) is `_world.recover(self._snapshot_frame)`. There is one snapshot
   per world, so a slot cannot be handed a different garment or body at reset.
   Wang's per-episode `env.reset(config_id=...)` (train_multi_garments.py:433)
   has no analogue, and Newton's `rotate_static_configs` has none either: Newton
   re-points a slot at another cached config without rebuilding, this port would
   have to rebuild the Genesis scene and re-settle.
   **Do instead:** give every cell its own permanent slot, and make the cell
   library exactly as large as the vector width. Measured, a rebuild is not
   absurd (`env.json build_seconds = 3.58` at 16 slots, plus 30 settle steps at
   ~0.32 s = ~13 s total), but rebuilding a Genesis/libuipc world in-process is
   unvalidated - the drape bake already had to run one interpreter per garment
   because libuipc's sanity checker carries state across worlds in one process
   (agent_docs/performance/2026-09-09-dressing-correctness.md:191-195). Treat
   rotation-by-rebuild as a separate, tested feature, not a prerequisite.
2. **The held-out unit is a BODY, not a pose id.** `select_complete_heldout_human`
   picks the greatest body carrying every garment. With `LiveCellFactory` every
   body carries every garment, so the rule degenerates to "the highest N seeds";
   keep the completeness check anyway so the baked-cache path stays honest
   (bodies 0, 2, 6 are the only complete rows there).
   Never hold out a garment: neither reference does, and a garment-held-out score
   answers a different question.
3. **One deterministic episode per cell per round.** Physics and the deterministic
   actor are the same every round; only observation augmentation varies
   (`train_sac.py:268` reseeds the per-slot rngs). Extra episodes per cell buy
   augmentation variance, not distribution coverage. Newton plays 35/35, Wang 25.
4. **No arm erosion for live cells.** `arm_erosion_m = 0.006` (dressing_env.py:59)
   exists because the baked pre-worn states interpenetrate. `LiveCellFactory`
   spawns the garment `clearance_m = 0.10` outside the fingertip
   (dressing_live.py:44-48), so erosion must be applied only to cache-sourced cells.

### 3.1 `python/uipc_manip/dressing_env.py`
**`DressingConfig`** - replace the single-body fields with a cell list.
- Add `cells: tuple[tuple[str, int], ...] = ()` - the (garment, body) bound to
  each slot, in slot order. Keep `human`/`garments` as the legacy single-body
  shorthand that `train_sac.make_env` expands into `cells`.
- Add `cell_source: str = "cache"` (`"cache"` or `"live"`) and
  `live: LiveCellConfig = field(default_factory=LiveCellConfig)`.
- Make `arm_erosion_m` apply only when `cell_source == "cache"`.
- `to_dict()` must emit `cells` so it lands in `run_dir/env.json` and the
  checkpoint metadata.

**`GenesisIPCDressingEnv.__init__` (line 180-203)** - build cells from the list.
- Replace line 197 with: if `cell_source == "live"`, `factory = LiveCellFactory(cfg.live)`
  and `self.cells = [factory.build(g, b) for g, b in cfg.cells]`; otherwise
  `self.cells = [self.cache.load(g, b) for g, b in cfg.cells]`.
- `len(cfg.cells)` must equal `num_envs`; raise otherwise.
- Keep a `self.cell_labels = [f"{c.garment}|human={c.human}" for c in self.cells]`.

**`GenesisIPCDressingEnv._build_scene` (line 208-326)** - per-slot arm. THE
BLOCKING FIX.
- Delete the module-level `arm_eroded`/`self.arm_vertices`/`self.arm_faces`
  derived from `cell0` (lines 212-213, 246-247).
- Add `self.arm_meshes: list[tuple[np.ndarray, np.ndarray]]`, one entry per cell:
  `erode_arm_mesh(cell.arm_points, cell.arm_faces, erosion, cell.finger, cell.shoulder)`
  with `erosion = cfg.arm_erosion_m if cache-sourced else 0.0`.
- Inside `add_objects_with_garments` (line 270-271) use
  `arm = ipc_trimesh(*self.arm_meshes[env_idx])` so the collider matches the name
  already written at line 278.
- Keep `self.arm_vertices/faces` as aliases of slot 0 only for the viewer
  (`_draw`, line 583) and trajectory export (`train_sac.py:312-313`); the export
  must switch to `env.arm_meshes[slot]`.
- The observation path already handles ragged bodies:
  `BatchedDressingObservationBuilder._pad` (dressing_obs.py:267-285) pads per
  environment, so differing arm vertex counts are fine.
- Keep the post-build `is_valid()` check (line 321-325); with live cells it should
  never fire, and if it does the cell is the report.

**`GenesisIPCDressingEnv.describe` (line 541-564)** - already returns `garment`
and `human`; add `"cell": f"{garment}|human={human}"` and `"cell_source"`.

**`GenesisIPCDressingEnv.step` (line 463-487)** - add `"human": cell.human` and
`"cell": self.cell_labels[i]` to the info dict, so `evaluate` never has to join
against `descriptions` by index.

**`dressing_reward.nearest_body_distance` (line 52-69)** - drop the
`_BODY_TREES.clear()` and key the dict by `id(human_points)` with a bounded
size (or hang the tree off the `DressingCell`). Measured: 1.45 ms per build,
23 ms/decision wasted at 16 slots today, ~46 ms at 32 distinct bodies.

### 3.2 `python/uipc_manip/dressing_live.py`
Only additions; the placement math is already correct.
- `LiveCellFactory.plan(garments, bodies)` -> `list[tuple[str,int]]`, the
  Cartesian product in a fixed order, so the launcher and the checkpoint agree
  on cell identity.
- `LiveCellFactory.preflight(cells) -> dict` wrapping the existing
  `clearances()` (line 316-321): for each cell report `cloth_to_arm_m` and
  `opening_to_finger_m`, and reject any cell below `2 * d_hat`. This is the
  substitute for Newton's bake accept/reject list; run it before a training run
  so an unusable body seed never occupies a slot for 10 hours.
- Cache `_bodies` to disk keyed by seed and `BodyConfig` hash, so a 32-slot world
  does not resample SMPL-X 32 times per process and an eval-only replay gets the
  identical bodies.

### 3.3 `python/uipc_manip/train_sac.py`
**`build_parser` (line 40-95)** - new options:
- `--cell-source {cache,live}` (default `live`).
- `--bodies N` or `--body-seeds 0,1,...` - the body axis. `--human` stays as the
  single-body shorthand (a regional teacher, PIPELINE.md:207-218) and becomes
  mutually exclusive with `--bodies`.
- `--heldout-bodies N` (default 2 for `live`, 1 for `cache`) or
  `--heldout-body-seeds a,b`.
- `--cells "g|human=N,..."` - an explicit slot plan; required for an
  `--eval-only` replay on unseen bodies.
- `--allow-partial-cell-coverage` - the escape hatch for smoke tests, mirroring
  PIPELINE.md:198-205. Without it the launcher fails fast when
  `len(cells) != num_envs`, when any garment is absent, or when
  `num_eval_episodes < len(cells)`.
- `--regional-body N` - alias of `--human`, reserves no held-out row (a region
  has one body), matching `--regional-teacher`.

**New `plan_cells(args) -> (slot_cells, heldout_slots, heldout_bodies)`** -
the prototype in
`/tmp/claude-4102472/.../scratchpad/cellplan_prototype.py` is the intended
implementation, tested by `test_cellplan.py` there:
- `select_heldout_bodies(cells, garments, count)` = greatest N bodies carrying
  every garment (port of `config.select_complete_heldout_human`).
- `plan_slots(cells, num_envs, heldout_bodies)` pins held-out cells to the
  leading slots (Newton `pinned_eval_human_ids`) and stratifies the rest by
  garment before spreading bodies (Newton `by_garment`).
- Print one `[uipc-manip distribution]` line: slots, unique cells, training
  cells, held-out cells, garments, bodies, held-out bodies - the analogue of
  `train_pointcloud_sac.py:1519-1533`.

**`make_env` (line 213-237)** - pass `cells=slot_cells`, `cell_source`, `live`
into `DressingConfig`.

**`evaluate` (line 258-345)** - three changes:
- Record `record["slot"] = i` and `record["cell"] = info["cell"]` (from 3.1).
- Replace the per-garment block (lines 336-343) with the prototype's
  `summarize(records, slot_cells, heldout_slots)`: per-cell
  `success_rate_{garment}|human={b}`, per-garment rollups, and
  `""`/`heldout_`/`training_` blocks each carrying `episode_count`, `cell_count`,
  `success_rate`, `worst_cell_success_rate`, `zero_cell_count`,
  `paper_filter_rate`, `mean_return`, `mean_final_upperarm_ratio`,
  `mean_final_forearm_ratio`.
- Enforce `episodes >= len(slot_cells)` when coverage is required, and keep
  `episodes_per_slot = 1` as the default (3.0 item 3).
The eval round stays deterministic and stays on the same env: held-out slots are
stepped during training and evaluated here, exactly as PIPELINE.md:182-191.
Evaluation is NOT truncated to the active curriculum stage. Wang truncates it
(`train_multi_garments.py:87-89` sets `num_of_garments_eval = curriculum_step`);
Newton evaluates all 35 cells every round. Follow Newton: a denominator that
changes between rounds makes best-checkpoint scores incomparable, which is the
whole point of `policy_checkpoint_score`.

**`checkpoint_score` (line 385-392)** - replace with the prototype's, i.e. the
`heldout_`-prefixed mirror of `eval_metrics.policy_checkpoint_score`:
`(heldout success_rate, heldout worst_cell_success_rate,
heldout mean_final_upperarm_ratio, heldout mean_final_forearm_ratio,
heldout mean_return, all-cell success_rate, all-cell mean_final_upperarm_ratio)`,
falling back to the all-cell keys when no held-out subset was evaluated.
Drop `-mean_final_distance`: `distance = 1 - upperarm_ratio` (dressing_env.py:471).
Do NOT put `paper_filter_rate` in the score. The reference uses the early-turn
test only in the Stage I-B rollout filter (`eval_metrics.fmvp_teacher_rollout_success:242-263`),
never in `policy_checkpoint_score`; it stays in the summary.
`worst_cell` sits second on purpose - the port's actual failure mode is one
garment at exactly 0.00 while the mean reads 0.50.

**`main` (line 425-626)**:
- After `plan_cells`, build `training_slot_mask` (numpy bool, held-out False).
- Line 567-568 becomes
  `if not (training_slot_mask[i] and slot_rank[i] < stage): continue`
  - the port of `train_pointcloud_sac.py:2220`. `added` then counts only training
  rows, so `--total-transitions` is already a training-cell budget (G5).
- Split `recent_success` into training/held-out lists for the CSV.
- `metadata` (line 507-517) gains `"cells": [...]`, `"heldout_cells": [...]`,
  `"heldout_bodies": [...]`, `"cell_source"`, `"live"` config dict, and the
  garment curriculum order, so `restore_resume_args` (line 108) can refuse a
  resume whose cell plan or held-out set differs - the port of
  `train_pointcloud_sac.py:1040-1051` and `:1767-1783`. `--eval-only` may differ
  but must print the difference (Newton `:1773-1779`).
- `eval_freq` / `checkpoint_interval` should be set from the eval-cost rule in 4.3,
  not left at 600/300.

### 3.4 What the evaluation summary must report (per round)
Required keys, in the eval CSV and the printed line:
- `episodes`, `cell_count`
- `success_rate`, `worst_cell_success_rate`, `zero_cell_count`
- `heldout_episode_count`, `heldout_cell_count`, `heldout_success_rate`,
  `heldout_worst_cell_success_rate`, `heldout_zero_cell_count`,
  `heldout_mean_final_upperarm_ratio`, `heldout_mean_final_forearm_ratio`,
  `heldout_mean_return`, `heldout_paper_filter_rate`
- the same block prefixed `training_`
- `success_rate_{garment}|human={b}` and `mean_final_upperarm_ratio_{garment}|human={b}`
  for every slot
- `success_rate_{garment}` rollups (kept, they answer "which garment is hard")
- `early_turn_rate`, `paper_filter_rate`, `sim_errors`, `max_tracking_error`
  (existing; `max_tracking_error` is the physics honesty check - a "success"
  above `grasp_tracking_tolerance_m = 0.02` is a dropped garment).

## 4. The experiment that demonstrates "one policy for them all"

Measured costs used throughout (agent_docs/performance/2026-09-09-dressing-correctness.md:519-534,
one decision = 6 simulation steps, contended GPU): 2.36 s/decision at 16 copies,
3.06 s at 32; 32 copies give 62.8 simulation steps/s against 40.7 at 16.
`build_seconds = 3.58` at 16 slots plus 30 settle steps.

### 4.1 The cell library
32 cells = 4 garments x 8 body seeds, one permanent slot each, `--num-envs 32`.
- Garments: `tshirt_26, tshirt_4, tshirt_68, tshirt_392`.
  Exclude `hospital_gown` from the training world: it is 10,436 vertices against
  3,529-6,837 for the shirts, the world iterates for its worst copy
  (perf record :386-392: 8 heterogeneous copies need 4.65 Newton iterations
  against 3.35 for 8 identical), and Newton measured it unusable at a fixed
  solver budget (PIPELINE.md:220-227). Score it separately in 4.4.
- Bodies: `--cell-source live --body-seeds 0..7`. The baked cache cannot supply
  this grid at all: `tshirt_4` is absent from every cached cell (measured
  inventory: gown 8, tshirt_392 8, tshirt_26 4, tshirt_68 3, tshirt_4 0), and
  even for the cache's own four garments only bodies 0, 2 and 6 are complete
  rows. `--cell-source live` is a requirement of this experiment, not an option.
- Split: hold out body seeds 6 and 7 -> 8 held-out cells, 24 training cells.
  One body (4 held-out cells) matches Newton's resolution of 5, but 4 episodes
  resolve success only to 0.25 and cannot separate 0.50 from 0.75; Wang's own
  held-out set is 25 configurations. The cost of the second body is 24 vs 28
  training slots, i.e. 7.84 vs 9.15 replay transitions/s.

### 4.2 Pre-flight, before any training (about 1 h)
1. `LiveCellFactory.preflight` over all 32 cells: `cloth_to_arm_m` and
   `opening_to_finger_m`. Reject and replace any body seed whose clearance is
   below `2*d_hat = 2 mm`. CPU + one short GPU bake per garment; the bakes are
   content-hash cached.
2. One scripted-expert episode per cell (`--policy heuristic --eval-only`,
   150 decisions, 32 copies = 7.6 min): record `final_upperarm_ratio` per cell
   and the measured seconds per decision on the real 32-cell world.
   Verify the expert on two live cells FIRST: `LiveCellFactory.build` sets
   `pull_waypoints = [picker_pos, right_shoulder]` (dressing_live.py:310-312), a
   straight chord, where the bake's cells carry a multi-waypoint schedule, and
   `scripted_actions()` threading a sleeve from 10 cm in front of the fingertip
   on that chord is unverified. If the expert does not thread on live cells, keep
   Wang's launcher order (G12 stays open) rather than deriving an order from a
   broken expert.
   This is the port's analogue of Newton's reachability sweep and it produces
   two things the run needs: the measured easiest-first curriculum order (so the
   port stops inheriting Wang's launcher order by assumption, G12), and the
   ceiling each cell can reach at all. A cell the expert leaves at 0.00 is an
   asset defect, not a policy failure.
3. Confirm `early_turn` on those expert episodes (G14). A CPU probe already
   narrows it: the detector is NOT trivially true. At the spawn position it
   returns False for all four cached garments on body 0, and sampling 51 points
   along the ideal straight pull spawn -> shoulder flags 0 of 51 on body 0 - but
   14 of 51 on body 6, whose arm is bent differently. So the observed
   `early_turn_rate = 1.00` comes from a 150-decision exploring policy latching
   the flag once (`train_sac.py:286` is a running OR), which is exactly the
   situation FMVP never applies the test to: Appendix A.1 filters frozen-teacher
   ROLLOUTS, not mid-training evaluation episodes. Two consequences: keep
   `paper_filter_rate` out of the checkpoint score, and before Stage I-B ever
   uses it, check the body-dependence above - the port resolves the sign in the
   arm's bend plane (dressing_reward.py:242-243) where FMVP uses world XZ cross
   products (`eval_metrics.fmvp_early_turn_mask:338-352`).

### 4.3 The training run
```
--task dressing --cell-source live --num-envs 32
--garments tshirt_26 tshirt_4 tshirt_68 tshirt_392 --body-seeds 0,1,2,3,4,5,6,7
--heldout-body-seeds 6,7
--horizon 150 --action-repeat 6            # 900 simulation steps/episode, discount 0.99, reward scale 1.0
--total-transitions 300000
--garment-curriculum-interval 900          # 4 garments all admitted by step 2,700 = 22% of the run
--garment-curriculum-order <measured order from 4.2>
--eval-freq 1200 --num-eval-episodes 32 --checkpoint-interval 1200
--batch-size 64 --updates-per-step 0 --seed 1
```
Cost, computed from 3.06 s/decision and 24 training slots, INCLUDING the
curriculum ramp (the loop at `train_sac.py:546` runs until the transition budget
is met, and gated slots do not contribute, so the ramp costs extra steps: 6/step
until 900, 12/step until 1,800, 18/step until 2,700, 24/step after - 32,400
transitions in the first 2,700 steps instead of 64,800):
- 300,000 transitions = 13,850 vector steps (12,500 without the ramp) = 11.8 h
  of stepping.
- One evaluation round = 150 decisions x 3.06 s = 7.6 min; at `--eval-freq 1200`
  that is 11.5 rounds = 1.5 h.
- Total about 13.2 h, evaluation 11% of wall clock (Newton targets ~15%,
  PIPELINE.md:259-264).
- A 600,000-transition run is 26,350 vector steps, 22.4 h + 2.8 h = 25.2 h.
Note the semantic difference from the reference: Newton converts
`--total-transitions` into a fixed VECTOR-STEP count up front
(`train_pointcloud_sac.py:1535-1544`), so under a curriculum it collects fewer
transitions than requested; this port's loop meets the transition budget exactly
and pays with extra steps. The port's is the more honest budget; state it.
Both figures assume 3.06 s/decision, which was measured on 2 garments and 1 body.
The world iterates for its worst copy and 8 heterogeneous copies needed 4.65
Newton iterations against 3.35 for 8 identical ones (perf record :386-392), so a
32-cell world of 4 garments and 8 distinct bodies will be slower. Recompute the
budget from the seconds-per-decision the pre-flight expert sweep (4.2 step 2)
measures on the real 32-cell world.
Reference points for the budget: this port first dressed an arm at 24,008
transitions on 2 cells; Newton's first single-cell upper-arm contact came at
350,000; the formal Newton run requested 5,000,000 over 30 cells and the
regional teacher 2,000,000 over 35 slots. 300k over 24 cells is 12,500
transitions per cell against Newton's 167,000 - so 300k is the *screening*
budget that answers "does the held-out score move off zero", and 600k-1M is what
a claim of a converged multi-cell teacher needs. Say which one is being run.
Seeds: one seed for the screening run; the claim needs 3 seeds (Wang runs
100/200/300, launch_train_curl.py:428), which is 36 h at 300k or a second GPU.

### 4.4 What the evaluation must show
Primary, from the best checkpoint by the 3.3 score:
1. `heldout_success_rate > 0` with `heldout_zero_cell_count = 0` - every one of
   the 8 unseen-body cells ends at least one trajectory with upper-arm ratio
   >= 0.7. The current failure mode is precisely a zero cell hiding inside a
   0.50 mean (tshirt_26 1.00, tshirt_392 0.00), so `worst_cell` and
   `zero_cell_count` are the load-bearing numbers, not the mean.
2. `heldout_success_rate` within one episode of `training_success_rate`
   (resolution 1/8 = 0.125 on the held-out side). A large drop is memorisation;
   equality at zero is not a result.
3. `success_rate_{garment}` non-zero for all four garments on the held-out
   bodies - "many garments" is a per-garment claim.
4. `max_tracking_error <= 0.02 m` on every counted episode; a success with a
   dropped cuff is not a success.
Secondary, and the strongest available generalization evidence, from a separate
`--eval-only` process (7.6 min per round, no training):
5. Zero-shot on 32 FRESH cells: `--cells` naming 4 garments x body seeds 8-15,
   none of which appeared in training or in the held-out grid. Report the same
   per-cell table. This is the closest analogue to FMVP rolling teachers over
   the full 27-region distribution.
6. `hospital_gown` scored separately on the same bodies, as an unseen GARMENT.
   Neither reference trains that way, so this is exploratory, not a claim.
7. 5 rounds on the held-out grid with different augmentation seeds (40 episodes,
   38 min) for a per-cell rate with some resolution. Be explicit that this adds
   robustness-to-observation-noise evidence, not distribution evidence: the
   physics and the deterministic actor repeat exactly.

### 4.5 Honest limits to state with the result
- 8 body seeds are a proxy distribution, not 8 participants; FMVP's Stage I is
  27 regions x 5 poses x 5 garments = 675 combinations with 27 regional teachers
  and a distillation step, and this is one joint teacher over 24 cells
  (PIPELINE.md:160-180, 207-218).
- With 8 held-out episodes per round, "not zero on an unseen body" is the claim
  available; a confidence interval is not.
- Newton measured a joint teacher over all 8 bodies splitting cleanly - four
  bodies at 0.19-0.27 mean upper-arm ratio and four at 0.00-0.04 across 21
  evaluations (PIPELINE.md:210-218). If this run reproduces that split, the
  correct next step is regional teachers plus distillation, not a longer joint
  run; `--regional-body` exists for exactly that.

## 5. Switch to Wang's regional teachers and distillation (2026-09-10, evening)

The joint run of section 4 is stopped before its budget. It is the design Wang
reports as the weak baseline: a single policy trained directly across the pose
range reaches an upper-arm ratio of 0.34 against 0.68 for regional teachers
distilled with the Earth Mover's loss (Wang RSS 2023, Table II), and Newton's
joint teacher over eight bodies split into four that learned and four at zero.
Mapped onto Wang's 27 arm-pose regions (`dressing_body.POSE_REGION_EDGES`), the
six training bodies are single poses in five regions (10, 12, 13, 20, 22, with
region 10 twice), and held-out bodies 6 and 7 sit in regions 6 and 9, which have
no training pose, so the run could not test generalisation even if it learned.
The privileged-critic variant was stopped at step 580 without a checkpoint. The
point-critic baseline is kept to its step-1000 checkpoint, evaluation and replay
snapshot, a resumable no-distillation control for a later Table-II comparison.

Body ids now carry a region. An id of `1000 (r + 1) + k` samples its arm pose
uniformly inside region `r`, whose three angle intervals are Wang's Appendix B.3
table. Ids below 1000 sample the whole range with the same draws as before, so
bodies 0 to 7 are unchanged. `env.json` records each slot's `pose_region`, which a
student uses to pick the teacher for a slot.

The plan, scaled to one GPU:

1. Pilot region 13, the middle interval of all three angles. Screen 24 candidate
   bodies with the scripted expert on both usable garments and keep bodies where
   both reach the forearm: 12 training and 2 held-out, 28 cells, one slot each.
   This screen is the one deliberate deviation from Wang. It removes placement
   defects such as tshirt_68 on bodies 1, 4 and 7, not hard poses, and the
   per-cell expert ceilings are recorded so the teacher's scores can be read
   against them.
2. Teacher 1 on those cells with Wang's point critic, horizon 300 with six steps
   per decision, 600,000 transitions, evaluation every 1000 steps. Read the
   per-cell curve at 300,000 and stop early if it is flat. No RL policy has yet
   dressed a live cell in this port, so a narrow-region teacher is the cleanest
   test that the task is learnable here.
3. Further regions one after another, since two runs on the GPU do no more work
   than one after the other. At about 8 to 9 transitions per second a 600,000
   teacher takes about 20 h, so 27 regions are out of reach. Three to five regions
   is the realistic scope, and choosing them is the user's call.
4. The student: Wang's SAC with the teacher loss of `SAC_AWAC.py:1025-1036`. That
   loss is the sum over the batch and the action dimensions of the squared mean
   difference plus the squared difference of the square-rooted standard
   deviations, at the tool point, with the teacher fed the student's own
   observation. The paper states a weight of 0.01 and the launcher in the
   reference checkout 0.002; the student launcher is not in the checkout, so
   the weight stays a flag. Wang keeps one entropy temperature per region and
   draws each update's batch from one region's replay. FMVP's
   behaviour-cloning route (`collect_rollouts.py`, `distill.py`) is the fallback.

A known gap remains: Wang draws a new pose every episode, while a slot here keeps
its cell for the life of the world. A student therefore sees 28 divided by the
number of regions poses per region, and rebuilding the world on a rotation is
left for later.

