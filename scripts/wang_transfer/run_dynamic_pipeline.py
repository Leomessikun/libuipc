"""Bounded corrected-teacher -> shared-data students -> DAgger -> held-out run.

The continuation gate is explicit and failures are retained. Uses at most two
own simulation processes and starts only below 70 GiB total used GPU memory.
CPU feature extraction/training do not allocate CUDA. Never stops other jobs.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
WT = ROOT.parent / "libuipc-anticipatory-dressing"
GENESIS = ROOT.parent / "genesis-world/.venv/bin/python"
CURL = Path("/home/ge47gax/miniconda3/envs/curl/bin/python")
GRAB = ROOT.parent / "GRAB/data/GRAB/grab"


def save(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, default=str) + "\n")
    tmp.replace(path)


def alive(pid):
    try:
        # A finished adopted process can remain a zombie until its parent reaps.
        return Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1][0] != "Z"
    except FileNotFoundError:
        return False


def canonical_command(command):
    return [str((ROOT / item).resolve()) if "/" in item and not Path(item).is_absolute()
            and (ROOT / item).exists() else item for item in command]


class Pipeline:
    def __init__(self, root, max_jobs, *, teacher_condition="observed", validation_repeats=1,
                 evaluation_repeats=1, reference_root=None, wait_for_root=None, preflight_heldout=False,
                 independent_resets=False):
        self.root, self.max_jobs = root, max_jobs
        self.teacher_condition = teacher_condition
        self.validation_repeats, self.evaluation_repeats = validation_repeats, evaluation_repeats
        self.reference_root, self.wait_for_root = reference_root, wait_for_root
        self.preflight_heldout = preflight_heldout
        self.independent_resets = independent_resets
        config = dict(teacher_condition=teacher_condition, validation_repeats=validation_repeats,
                      evaluation_repeats=evaluation_repeats, reference_root=str(reference_root) if reference_root else None,
                      wait_for_root=str(wait_for_root) if wait_for_root else None, preflight_heldout=preflight_heldout,
                      max_jobs=max_jobs, independent_resets=independent_resets)
        config_path = root / "pipeline_config.json"
        if config_path.exists() and json.loads(config_path.read_text()) != config:
            raise ValueError("Pipeline configuration changed; use a new output root")
        save(config_path, config)
        self.processes = {}
        self.env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        self.status = dict(stage="starting", started_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                           teacher_condition=teacher_condition, jobs={})

    def write_status(self, **values):
        self.status.update(values)
        save(self.root / "pipeline_status.json", self.status)

    def gpu_ready(self):
        result = subprocess.run([str(GENESIS), "-c", "import torch; f,t=torch.cuda.mem_get_info(); print((t-f)/2**30)"],
                                capture_output=True, text=True, timeout=30)
        if result.returncode:
            self.write_status(gpu_probe_error=result.stderr[-500:])
            return False
        used = float(result.stdout.strip().splitlines()[-1])
        self.write_status(last_gpu_used_GiB=used)
        return used < 70.

    def run_jobs(self, jobs, *, gpu=True):
        """Resume only exact own manifests; never silently overwrite a failed run."""
        jobs = [dict(job, command=[str(x) for x in job["command"]]) for job in jobs]
        save(self.root / f"{self.status['stage']}_jobs.json", jobs)
        remaining = list(jobs)
        while remaining:
            active = 0
            pending = []
            for job in remaining:
                name, marker = job["name"], Path(job["complete"])
                launch_path = self.root / f"{name}_launch.json"
                launch = json.loads(launch_path.read_text()) if launch_path.exists() else None
                if launch and canonical_command(launch["command"]) != canonical_command(job["command"]):
                    raise RuntimeError(f"Launch arguments changed for {name}; use a new output root")
                process = self.processes.get(name)
                if process is not None:
                    process.poll()
                if launch and alive(launch["pid"]):
                    active += 1
                    pending.append(job)
                    self.status["jobs"][name] = dict(status="running", pid=launch["pid"])
                elif marker.exists():
                    if gpu:
                        metrics = json.loads(marker.read_text())
                        if any(m.get("failure", {}).get("kind") == "invalid_physics" for m in metrics if m.get("failure")):
                            raise RuntimeError(f"Invalid physics in {name}; diagnose before continuing")
                    self.status["jobs"][name] = dict(status="complete", artifact=str(marker))
                elif launch:
                    raise RuntimeError(f"Job {name} exited without completion; see {name}.log")
                else:
                    pending.append(job)
                    self.status["jobs"][name] = dict(status="pending")
            remaining = pending
            for job in remaining:
                name = job["name"]
                if self.status["jobs"][name]["status"] != "pending":
                    continue
                if active >= (self.max_jobs if gpu else 1):
                    break
                if gpu and not self.gpu_ready():
                    break
                env = self.env if gpu else dict(self.env, CUDA_VISIBLE_DEVICES="")
                with (self.root / f"{name}.log").open("x") as log:
                    process = subprocess.Popen(job["command"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                               env=env, start_new_session=True)
                self.processes[name] = process
                save(self.root / f"{name}_launch.json", dict(pid=process.pid, command=job["command"]))
                self.status["jobs"][name] = dict(status="running", pid=process.pid)
                print(f"[launch] {name} pid={process.pid}", flush=True)
                active += 1
                # Give the first simulation time to establish its CUDA context
                # before measuring memory for the next one.
                if gpu:
                    time.sleep(15)
            self.write_status()
            if remaining:
                time.sleep(15)

    def sim(self, name, motion, condition, *, student=None, beta=1., onset=1., seed=0, steps=None):
        out = self.root / name
        cmd = [GENESIS, "-u", HERE / "motion_lookahead_probe.py", "--motion", motion,
               "--out", out, "--conditions", condition, "--candidate-set", "observed_common",
               "--planning-window", "0", "5", "--save-trajectory"]
        if onset != 1.:
            cmd += ["--onset", str(onset)]
        if seed:
            cmd += ["--seed", str(seed)]
        if beta != 1.:
            cmd += ["--teacher-execution-probability", str(beta)]
        if student:
            cmd += ["--student-checkpoint", student]
        if steps is not None:
            cmd += ["--steps", str(steps)]
        return dict(name=name, command=cmd, complete=out / "metrics.json")

    def wait_for_previous_queue(self):
        if self.wait_for_root is None:
            return
        if self.wait_for_root.resolve() == self.root.resolve():
            raise ValueError("A queue cannot wait for itself")
        status_path = self.wait_for_root / "pipeline_status.json"
        launch_path = self.wait_for_root / "pipeline_launch.json"
        if not status_path.exists() or not launch_path.exists():
            raise FileNotFoundError("Dependency must have status and launch manifests")
        self.write_status(stage="waiting_for_reference", dependency=str(self.wait_for_root))
        while True:
            previous = json.loads(status_path.read_text())
            pid = json.loads(launch_path.read_text())["pid"]
            jobs_alive = any(alive(job["pid"]) for job in previous.get("jobs", {}).values()
                             if job.get("status") == "running" and job.get("pid"))
            if not alive(pid) and not jobs_alive:
                if previous["stage"] not in ("complete", "teacher_gate_failed", "error"):
                    raise RuntimeError(f"Dependency exited without terminal status: {previous['stage']}")
                return
            time.sleep(15)

    def validation_job(self, clip, condition, repeat, motion):
        suffix = f"_rep{repeat}" if self.validation_repeats > 1 else ""
        job = self.sim(f"validate_{clip}_{condition}{suffix}", motion, condition, seed=repeat)
        if self.reference_root is None or repeat != 0 or condition == self.teacher_condition:
            return job
        folder = self.reference_root / f"validate_{clip}_{condition}"
        metadata = json.loads((folder / "run.json").read_text())
        args = metadata["arguments"]
        expected = dict(conditions=[condition], seed=0, horizon=4, steps=450, candidate_set="observed_common",
                        planning_window=[0., 5.], onset=1., success=.7, hold=20, obs_mode="wang_live_arm",
                        teacher_execution_probability=1., student_checkpoint=None, motion_speed=1.,
                        yaw=267., tracking_tolerance=.002, drive_strength=1e6,
                        newton_velocity_tolerance=.01, interval=1, hang_key="k300", save_trajectory=True)
        if any(args.get(k) != value for k, value in expected.items()):
            raise ValueError(f"Reference protocol differs in {folder}")
        if hashlib.sha256(Path(motion).read_bytes()).hexdigest() != metadata["motion_sha256"]:
            raise ValueError(f"Reference motion differs in {folder}")
        if hashlib.sha256(Path(args["checkpoint"]).read_bytes()).hexdigest() != metadata["checkpoint_sha256"]:
            raise ValueError(f"Reference checkpoint changed in {folder}")
        for f in ("metrics.json", f"{condition}_initial.npz"):
            if not (folder / f).exists():
                raise FileNotFoundError(folder / f)
        launch = json.loads((self.reference_root / f"validate_{clip}_{condition}_launch.json").read_text())
        job.update(command=launch["command"], complete=folder / "metrics.json",
                   reference_root=str(self.reference_root),
                   reference_metrics_sha256=hashlib.sha256((folder / "metrics.json").read_bytes()).hexdigest())
        return job

    def prepare_motions(self, specs):
        jobs = []
        for name, body, subject, clip in specs:
            path = self.root / "motions" / f"{name}.npz"
            if path.exists():
                continue
            jobs.append(dict(name=f"prepare_{name}", complete=path,
                             command=[GENESIS, WT / "scripts/wang_transfer/prepare_grab_motion.py",
                                      GRAB / subject / f"{clip}.npz", "--out", path, "--body", body,
                                      "--start", "1", "--duration", "3", "--amplitude", ".35"]))
        self.run_jobs(jobs, gpu=False)
    def fit(self, round_name, episodes):
        cache = self.root / f"{round_name}_features"
        self.write_status(stage=f"{round_name}_features")
        self.run_jobs([dict(name=f"{round_name}_features", complete=cache / "manifest.json",
                            command=[CURL, "-u", HERE / "train_dynamic_student.py", "features",
                                     "--episodes", *episodes, "--out", cache,
                                     "--teacher-condition", self.teacher_condition])], gpu=False)
        self.write_status(stage=f"{round_name}_train")
        jobs = []
        for seed in range(3):
            for mode in ("history", "current"):
                name = f"{round_name}_{mode}_seed{seed}"
                out = self.root / name
                cmd = [CURL, "-u", HERE / "train_dynamic_student.py", "train", "--features", cache,
                       "--validation-bodies", "14052", "--out", out, "--seed", str(seed)]
                if mode == "current":
                    cmd += ["--current-only"]
                jobs.append(dict(name=name, command=cmd, complete=out / "manifest.json"))
        self.run_jobs(jobs, gpu=False)
        configs = [json.loads(Path(job["complete"]).read_text()) for job in jobs]
        for key in ("feature_manifest_sha256", "base_sha256", "updates", "batch", "parameters", "validation_bodies"):
            if len({json.dumps(config[key], sort_keys=True) for config in configs}) != 1:
                raise RuntimeError(f"Student controls differ in {key}")

    def run(self):
        self.wait_for_previous_queue()
        self.write_status(stage="validation")
        validation = []
        for clip in ("pass", "lift"):
            motion = WT / f"output/anticipatory_dressing/motions/s1_mug_{clip}_body14046.npz"
            for repeat in range(self.validation_repeats):
                for condition in (self.teacher_condition, "current", "gicp"):
                    validation.append(self.validation_job(clip, condition, repeat, motion))
        self.run_jobs(validation)
        table = {}
        initial_comparisons = {}
        for clip in ("pass", "lift"):
            for repeat in range(self.validation_repeats):
                cell = clip if self.validation_repeats == 1 else f"{clip}_rep{repeat}"
                table[cell] = {}
                folders = []
                for condition in (self.teacher_condition, "current", "gicp"):
                    suffix = f"_rep{repeat}" if self.validation_repeats > 1 else ""
                    job = next(j for j in validation if j["name"] == f"validate_{clip}_{condition}{suffix}")
                    folder = Path(job["complete"]).parent
                    folders.append(folder)
                    metric = json.loads((folder / "metrics.json").read_text())[0]
                    table[cell][condition] = bool(metric["final_success"])
                # Check actual settled starts across independent native processes.
                import numpy as np
                initial = [np.load(folder / f"{condition}_initial.npz")
                           for folder, condition in zip(folders, (self.teacher_condition, "current", "gicp"))]
                differences = [{k: float(np.linalg.norm(other[k] - initial[0][k], axis=-1).max())
                                for k in initial[0].files} for other in initial[1:]]
                errors = [v for diff in differences for v in diff.values()]
                for part in initial:
                    part.close()
                initial_comparisons[cell] = dict(reference_condition=self.teacher_condition,
                    compared_conditions=["current", "gicp"], max_displacement_m=differences,
                    exact_within_10um=bool(np.isfinite(errors).all() and max(errors) <= 1e-5),
                    comparison="independent cloth settles; matched configuration" if self.independent_resets else "strict initial-array match")
                save(self.root / "initial_comparisons.json", initial_comparisons)
                protected_errors = [v for diff in differences for k, v in diff.items() if k != "positions"]
                if (not np.isfinite(errors).all() or max(protected_errors, default=0.) > 1e-5
                        or (not self.independent_resets and max(errors) > 1e-5)):
                    raise RuntimeError(f"Initial states differ in {cell}: {errors}")
        teacher_wins = sum(v[self.teacher_condition] for v in table.values())
        gate = (all(any(v[self.teacher_condition] for key, v in table.items() if key.split('_rep')[0] == clip)
                    for clip in ("pass", "lift")) and
                all(teacher_wins > sum(v[c] for v in table.values()) for c in ("current", "gicp")))
        save(self.root / "teacher_gate.json", dict(passed=gate, results=table, teacher=self.teacher_condition,
             rule="At least one teacher success per clip and more total successes than each comparator; feasibility gate only"))
        if not gate:
            self.write_status(stage="teacher_gate_failed", results=table,
                              next="Diagnose selected teacher/candidate effect before more collection; no student trained")
            return

        # Fixed before collection: training/development s1 only, three fit bodies
        # plus one validation body. Completely separate subjects/sequences at test.
        specs = [(f"body{body}_{clip}", body, "s1", f"mug_{'pass_1' if clip == 'pass' else 'lift'}")
                 for body in (14047, 14048, 14050, 14052) for clip in ("pass", "lift")]
        self.write_status(stage="prepare_training_motions")
        self.prepare_motions(specs)
        self.write_status(stage="teacher_initialization")
        init_jobs = [self.sim(f"init_{name}", self.root / "motions" / f"{name}.npz", self.teacher_condition,
                              onset=.8 if i % 2 == 0 else 1.2, seed=1)
                     for i, (name, _, _, _) in enumerate(specs)]
        self.run_jobs(init_jobs)
        episodes = [self.root / job["name"] for job in init_jobs]
        self.fit("round0", episodes)

        self.write_status(stage="dagger_student_rollin")
        dagger = []
        for mode in ("history", "current"):
            for body, clip in ((14047, "pass"), (14048, "lift")):
                name = f"dagger_{mode}_body{body}_{clip}"
                dagger.append(self.sim(name, self.root / "motions" / f"body{body}_{clip}.npz", self.teacher_condition,
                                       student=self.root / f"round0_{mode}_seed0/student.pt", beta=0., onset=1.4, seed=2))
        self.run_jobs(dagger)
        # Both models train on exactly the same union, including failed roll-ins.
        self.fit("round1", episodes + [self.root / job["name"] for job in dagger])

        test_specs = [(f"test_body{body}_{name}", body, subject, clip)
                      for body in (14054, 14055)
                      for name, subject, clip in (("pass", "s2", "mug_pass_1"), ("phone", "s3", "phone_call_1"))]
        self.write_status(stage="prepare_test_motions")
        self.prepare_motions(test_specs)
        if self.preflight_heldout:
            self.write_status(stage="heldout_motion_preflight")
            checks = [self.sim(f"preflight_{name}", self.root / "motions" / f"{name}.npz", "hold", steps=60)
                      for name, _, _, _ in test_specs]
            self.run_jobs(checks)
            results = {job["name"]: json.loads(Path(job["complete"]).read_text())[0] for job in checks}
            passed = all(m.get("failure") is None and m.get("time_s", 0.) >= 6. - 1e-6
                         and m.get("body_tracking_max_m", 1.) <= .002 for m in results.values())
            save(self.root / "heldout_motion_preflight.json", dict(passed=passed, results=results,
                 rule="Six-second passive hold covers complete motion with valid attachment and <=2 mm body tracking; no success-rate selection"))
            if not passed:
                self.write_status(stage="heldout_preflight_failed", results=results,
                                  next="Inspect prescribed motion/initial grasp before evaluating these fixed held-out cells")
                return
        self.write_status(stage="heldout_evaluation")
        tests = []
        for name, _, _, _ in test_specs:
            for repeat in range(self.evaluation_repeats):
                label = name if self.evaluation_repeats == 1 else f"{name}_rep{repeat}"
                motion = self.root / "motions" / f"{name}.npz"
                for baseline in ("none", "gicp"):
                    tests.append(self.sim(f"{label}_{baseline}", motion, baseline, seed=3 + repeat))
                tests.append(self.sim(f"{label}_r1_fixed_encoder", motion, "none", seed=3 + repeat,
                                      student=self.root / "round0_history_seed0/initial_student.pt"))
                for seed in range(3):
                    for mode in ("history", "current"):
                        tests.append(self.sim(f"{label}_{mode}_seed{seed}", motion, "none", seed=3 + repeat,
                                              student=self.root / f"round1_{mode}_seed{seed}/student.pt"))
        self.run_jobs(tests)
        summary = {job["name"]: json.loads(Path(job["complete"]).read_text())[0] for job in tests}
        save(self.root / "heldout_results.json", summary)
        self.write_status(stage="complete", results_file=str(self.root / "heldout_results.json"),
                          interpretation="Four held-out task cells, three model seeds; no claim of statistical generalization from four tasks")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--max-jobs", type=int, choices=(1, 2), default=2)
    parser.add_argument("--teacher-condition", choices=("observed", "causal"), default="observed")
    parser.add_argument("--validation-repeats", type=int, choices=(1, 2), default=1)
    parser.add_argument("--evaluation-repeats", type=int, choices=(1, 2), default=1)
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--wait-for-root", type=Path)
    parser.add_argument("--preflight-heldout", action="store_true")
    parser.add_argument("--independent-resets", action="store_true",
                        help="Report independent cloth settling differences; still require matched human/tool starts")
    args = parser.parse_args()
    args.root = args.root.resolve(); args.root.mkdir(parents=True, exist_ok=True)
    (args.root / "motions").mkdir(exist_ok=True)
    with (args.root / "pipeline.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pipeline = Pipeline(args.root, args.max_jobs, teacher_condition=args.teacher_condition,
                            validation_repeats=args.validation_repeats, evaluation_repeats=args.evaluation_repeats,
                            reference_root=args.reference_root.resolve() if args.reference_root else None,
                            wait_for_root=args.wait_for_root.resolve() if args.wait_for_root else None,
                            preflight_heldout=args.preflight_heldout, independent_resets=args.independent_resets)
        try:
            pipeline.run()
        except Exception as exc:
            pipeline.write_status(stage="error", error=repr(exc))
            raise


if __name__ == "__main__":
    main()
