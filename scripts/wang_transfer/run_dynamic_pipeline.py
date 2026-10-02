"""Bounded corrected-teacher -> shared-data students -> DAgger -> held-out run.

The continuation gate is explicit and failures are retained. Uses at most two
own simulation processes and starts only below 70 GiB total used GPU memory.
CPU feature extraction/training do not allocate CUDA. Never stops other jobs.
"""
from __future__ import annotations

import argparse
import fcntl
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
    def __init__(self, root, max_jobs):
        self.root, self.max_jobs = root, max_jobs
        self.processes = {}
        self.env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        self.status = dict(stage="starting", started_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), jobs={})

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

    def sim(self, name, motion, condition, *, student=None, beta=1., onset=1., seed=0):
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
        return dict(name=name, command=cmd, complete=out / "metrics.json")

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
                                     "--episodes", *episodes, "--out", cache])], gpu=False)
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
        self.write_status(stage="validation")
        validation = []
        for clip in ("pass", "lift"):
            motion = WT / f"output/anticipatory_dressing/motions/s1_mug_{clip}_body14046.npz"
            for condition in ("observed", "current", "gicp"):
                validation.append(self.sim(f"validate_{clip}_{condition}", motion, condition))
        self.run_jobs(validation)
        table = {}
        for clip in ("pass", "lift"):
            table[clip] = {}
            for condition in ("observed", "current", "gicp"):
                folder = self.root / f"validate_{clip}_{condition}"
                metric = json.loads((folder / "metrics.json").read_text())[0]
                table[clip][condition] = bool(metric["final_success"])
            # Check actual settled starts across independent native processes.
            import numpy as np
            initial = [np.load(self.root / f"validate_{clip}_{c}/{c}_initial.npz")
                       for c in ("observed", "current", "gicp")]
            errors = [float(np.linalg.norm(other[k] - initial[0][k], axis=-1).max())
                      for other in initial[1:] for k in initial[0].files]
            for part in initial:
                part.close()
            if not np.isfinite(errors).all() or max(errors) > 1e-5:
                raise RuntimeError(f"Initial states differ in {clip}: {errors}")
        gate = (all(v["observed"] for v in table.values()) and
                all(any(v["observed"] and not v[c] for v in table.values()) for c in ("current", "gicp")))
        save(self.root / "teacher_gate.json", dict(passed=gate, results=table,
             rule="Observed succeeds on both clips, with at least one paired win over each comparator; feasibility gate only"))
        if not gate:
            self.write_status(stage="teacher_gate_failed", results=table,
                              next="Diagnose observed forecast/common-candidate effect before more collection; no student trained")
            return

        # Fixed before collection: training/development s1 only, three fit bodies
        # plus one validation body. Completely separate subjects/sequences at test.
        specs = [(f"body{body}_{clip}", body, "s1", f"mug_{'pass_1' if clip == 'pass' else 'lift'}")
                 for body in (14047, 14048, 14050, 14052) for clip in ("pass", "lift")]
        self.write_status(stage="prepare_training_motions")
        self.prepare_motions(specs)
        self.write_status(stage="teacher_initialization")
        init_jobs = [self.sim(f"init_{name}", self.root / "motions" / f"{name}.npz", "observed",
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
                dagger.append(self.sim(name, self.root / "motions" / f"body{body}_{clip}.npz", "observed",
                                       student=self.root / f"round0_{mode}_seed0/student.pt", beta=0., onset=1.4, seed=2))
        self.run_jobs(dagger)
        # Both models train on exactly the same union, including failed roll-ins.
        self.fit("round1", episodes + [self.root / job["name"] for job in dagger])

        test_specs = [(f"test_body{body}_{name}", body, subject, clip)
                      for body in (14054, 14055)
                      for name, subject, clip in (("pass", "s2", "mug_pass_1"), ("phone", "s3", "phone_call_1"))]
        self.write_status(stage="prepare_test_motions")
        self.prepare_motions(test_specs)
        self.write_status(stage="heldout_evaluation")
        tests = []
        for name, _, _, _ in test_specs:
            motion = self.root / "motions" / f"{name}.npz"
            for baseline in ("none", "gicp"):
                tests.append(self.sim(f"{name}_{baseline}", motion, baseline, seed=3))
            tests.append(self.sim(f"{name}_r1_fixed_encoder", motion, "none", seed=3,
                                  student=self.root / "round0_history_seed0/initial_student.pt"))
            for seed in range(3):
                for mode in ("history", "current"):
                    tests.append(self.sim(f"{name}_{mode}_seed{seed}", motion, "none", seed=3,
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
    args = parser.parse_args()
    args.root = args.root.resolve(); args.root.mkdir(parents=True, exist_ok=True)
    (args.root / "motions").mkdir(exist_ok=True)
    with (args.root / "pipeline.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pipeline = Pipeline(args.root, args.max_jobs)
        try:
            pipeline.run()
        except Exception as exc:
            pipeline.write_status(stage="error", error=repr(exc))
            raise


if __name__ == "__main__":
    main()
