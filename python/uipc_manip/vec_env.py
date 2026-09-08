"""Process-parallel wrapper around :class:`GenesisIPCManipEnv`.

Native IPC geometries reach Genesis through coupler internals that support one
environment per scene, so parallelism comes from separate processes, each with
its own Genesis initialisation and its own IPC workspace. Workers use the
``spawn`` start method because CUDA state cannot be forked.

Episodes auto-reset inside the worker. The observation returned for a finished
step is the first observation of the next episode, and the observation that
ended the episode travels in ``info["terminal_obs"]`` so the trainer can store
a time-limit transition whose bootstrap target is the true final state.
"""

from __future__ import annotations

import multiprocessing as mp
import sys
import traceback
from pathlib import Path

import numpy as np

_PACKAGE_PARENT = str(Path(__file__).resolve().parents[1])


def _worker(conn, cfg_dict: dict) -> None:
    if _PACKAGE_PARENT not in sys.path:
        sys.path.insert(0, _PACKAGE_PARENT)
    env = None
    try:
        from uipc_manip.genesis_env import EnvConfig, GenesisIPCManipEnv

        env = GenesisIPCManipEnv(EnvConfig(**cfg_dict))
        conn.send(("ready", {"obs_dim": env.spec.dim, "action_dim": env.action_dim, "describe": env.describe()}))
        while True:
            cmd, arg = conn.recv()
            if cmd == "reset":
                conn.send(("ok", env.reset(arg)))
            elif cmd == "step":
                try:
                    obs, reward, done, info = env.step(arg)
                except Exception as exc:  # simulation failure: report and start a fresh episode
                    info = {"sim_error": True, "error": repr(exc), "success": False, "distance": float("nan")}
                    obs = env.reset()
                    conn.send(("ok", (obs, 0.0, True, info)))
                    continue
                if done:
                    info["terminal_obs"] = obs
                    obs = env.reset()
                conn.send(("ok", (obs, reward, done, info)))
            elif cmd == "state":
                conn.send(("ok", env.state()))
            elif cmd == "describe":
                conn.send(("ok", env.describe()))
            elif cmd == "close":
                break
    except Exception:
        conn.send(("error", traceback.format_exc()))
    finally:
        if env is not None:
            env.close()
        conn.close()


class SubprocVecEnv:
    """Vector of environments living in separate processes."""

    def __init__(self, cfg_dicts: list[dict]) -> None:
        ctx = mp.get_context("spawn")
        self._conns = []
        self._procs = []
        for cfg in cfg_dicts:
            parent, child = ctx.Pipe()
            proc = ctx.Process(target=_worker, args=(child, dict(cfg)), daemon=True)
            proc.start()
            child.close()
            self._conns.append(parent)
            self._procs.append(proc)
        self.descriptions = []
        for conn in self._conns:
            status, payload = conn.recv()
            if status != "ready":
                self.close()
                raise RuntimeError(f"Environment worker failed to start:\n{payload}")
            self.descriptions.append(payload["describe"])
        self.obs_dim = int(self.descriptions[0]["obs_dim"])
        self.action_dim = int(self.descriptions[0]["action_dim"])
        self.num_envs = len(self._conns)

    def _request(self, cmd: str, args: list) -> list:
        for conn, arg in zip(self._conns, args, strict=True):
            conn.send((cmd, arg))
        results = []
        for conn in self._conns:
            status, payload = conn.recv()
            if status != "ok":
                self.close()
                raise RuntimeError(f"Environment worker failed:\n{payload}")
            results.append(payload)
        return results

    def reset(self, seeds: list[int | None] | None = None) -> np.ndarray:
        seeds = [None] * self.num_envs if seeds is None else list(seeds)
        return np.stack(self._request("reset", seeds)).astype(np.float32)

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        actions = np.asarray(actions, dtype=np.float32).reshape(self.num_envs, self.action_dim)
        results = self._request("step", [a.copy() for a in actions])
        obs = np.stack([r[0] for r in results]).astype(np.float32)
        rewards = np.asarray([r[1] for r in results], dtype=np.float32)
        dones = np.asarray([r[2] for r in results], dtype=bool)
        infos = [r[3] for r in results]
        return obs, rewards, dones, infos

    def states(self) -> list[dict]:
        return self._request("state", [None] * self.num_envs)

    def close(self) -> None:
        for conn in self._conns:
            try:
                conn.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for proc in self._procs:
            proc.join(timeout=10)
            if proc.is_alive():
                proc.terminate()
        self._conns = []
        self._procs = []


class DummyVecEnv:
    """In-process vector environment (normally a single environment) for debugging and viewers."""

    def __init__(self, cfg_dicts: list[dict]) -> None:
        from .genesis_env import EnvConfig, GenesisIPCManipEnv

        self.envs = [GenesisIPCManipEnv(EnvConfig(**cfg)) for cfg in cfg_dicts]
        self.descriptions = [env.describe() for env in self.envs]
        self.obs_dim = self.envs[0].spec.dim
        self.action_dim = self.envs[0].action_dim
        self.num_envs = len(self.envs)

    def reset(self, seeds: list[int | None] | None = None) -> np.ndarray:
        seeds = [None] * self.num_envs if seeds is None else list(seeds)
        return np.stack([env.reset(seed) for env, seed in zip(self.envs, seeds, strict=True)]).astype(np.float32)

    def step(self, actions: np.ndarray):
        actions = np.asarray(actions, dtype=np.float32).reshape(self.num_envs, self.action_dim)
        obs_list, rewards, dones, infos = [], [], [], []
        for env, action in zip(self.envs, actions, strict=True):
            obs, reward, done, info = env.step(action)
            if done:
                info["terminal_obs"] = obs
                obs = env.reset()
            obs_list.append(obs)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        return (
            np.stack(obs_list).astype(np.float32),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(dones, dtype=bool),
            infos,
        )

    def states(self) -> list[dict]:
        return [env.state() for env in self.envs]

    def close(self) -> None:
        for env in self.envs:
            env.close()


def make_vec_env(cfg_dicts: list[dict], *, subprocess: bool = True):
    if subprocess:
        return SubprocVecEnv(cfg_dicts)
    return DummyVecEnv(cfg_dicts)
