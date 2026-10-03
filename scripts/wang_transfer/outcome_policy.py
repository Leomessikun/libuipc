"""CPU proposal inference bridge with explicitly restorable causal history."""
from __future__ import annotations

from collections import deque
from contextlib import redirect_stdout
import copy
import io
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np

from dynamic_student import encode, sha256
from motion_observation import history_inputs
from outcome_goals import base_action, goal_input, proposal_net, residual_action
from train_dynamic_student import PACKAGE


class OutcomeClient:
    def __init__(self, models, effect_models=None):
        self.reply_horizon = (json.loads((Path(effect_models) / 'interface.json').read_text())['horizon']
                              if effect_models is not None else 8)
        command = ['/home/ge47gax/miniconda3/envs/curl/bin/python', '-u', str(Path(__file__).resolve()), str(models)]
        if effect_models is not None:
            command.append(str(effect_models))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', PYTHONPATH=f'/home/ge47gax/kun:{PACKAGE}')
        self.proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, env=env)
        startup = bytearray()
        while True:
            line = self.proc.stderr.readline(); startup.extend(line)
            if line.strip() == b'[outcome] ready':
                break
            if not line:
                raise RuntimeError('Outcome bridge failed: ' + startup.decode(errors='replace'))

    def request(self, op, **arrays):
        buffer = io.BytesIO(); np.savez(buffer, op=np.asarray(op), **arrays)
        payload = buffer.getvalue()
        self.proc.stdin.write(struct.pack('<Q', len(payload)) + payload); self.proc.stdin.flush()
        data = bytearray()
        floats = self.reply_horizon * 6 + 50
        while len(data) < floats * 4:
            chunk = self.proc.stdout.read(floats * 4 - len(data))
            if not chunk:
                raise RuntimeError('Outcome bridge closed: ' + self.proc.stderr.read().decode(errors='replace'))
            data.extend(chunk)
        result = np.frombuffer(data, np.float32).copy()
        if not np.isfinite(result).all():
            raise FloatingPointError('Invalid bridge reply')
        count = self.reply_horizon * 6
        return result[:count].reshape(self.reply_horizon, 6), result[count:]

    def close(self):
        if self.proc.poll() is None:
            self.proc.stdin.close()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.terminate(); self.proc.wait(timeout=10)


class OutcomePolicy:
    def __init__(self, models, effect_models=None):
        sys.path.insert(0, str(PACKAGE))
        import torch
        from uipc_manip.wang_bridge import ReferencePolicy
        torch.set_num_threads(1)
        self.models = {}
        for kind in ('geometry', 'generic'):
            d = torch.load(str(models / f'{kind}.pt'), map_location='cpu', weights_only=False)
            config = d['config']
            if sha256(config['base_checkpoint']) != config['base_sha256']:
                raise ValueError('Base policy changed')
            if sha256(Path(__file__).with_name('outcome_goals.py')) != config['geometry_sha256']:
                raise ValueError('Goal/controller code changed after training')
            if config['chunk'] not in (1, 8):
                raise ValueError('This pilot bridge supports one- or eight-decision chunks')
            model = proposal_net(config['input_dim'], config['chunk']); model.load_state_dict(d['state_dict']); model.eval()
            self.models[kind] = (model, d['mean'], d['std'])
        self.config = config
        self.base = ReferencePolicy(config['base_checkpoint'], device='cpu', yaw_deg=config['yaw'])
        self.features = deque(maxlen=config['frames'])
        self.tools = deque(maxlen=config['frames'])
        self.executed = deque(maxlen=config['frames'] - 1)
        self.action, self.feature, self.tool = np.zeros(6), np.zeros(50), np.zeros(3)
        self.effect_models = None
        if effect_models is not None:
            from effect_codes import EffectModels
            self.effect_models = EffectModels(effect_models)
            if (self.effect_models.config['base_sha256'] != config['base_sha256']
                    or self.effect_models.config['frames'] != config['frames']):
                raise ValueError('Effect/base observation contracts differ')

    def request(self, request):
        import torch
        op = str(request['op'])
        if op == 'observe':
            if 'previous_action' in request:
                self.executed.append(request['previous_action'].copy())
            self.feature, logits = encode(self.base, request['pos'], request['flags'])
            self.action = base_action(logits, self.base.rotation)
            self.tool = request['tool']
            self.features.append(self.feature.copy())
            self.tools.append(self.tool @ self.base.rotation.T)
        elif op == 'save':
            self.saved = copy.deepcopy((self.features, self.tools, self.action, self.feature, self.tool, self.executed))
        elif op == 'restore':
            self.features, self.tools, self.action, self.feature, self.tool, self.executed = copy.deepcopy(self.saved)
        elif op == 'effect_sample':
            from effect_codes import ACTION_AXES
            if self.effect_models is None:
                raise ValueError('No effect-code models were supplied')
            hist = history_inputs(np.asarray(self.features), np.asarray(self.tools), self.config['frames'])[-1]
            past = np.zeros((self.config['frames'] - 1) * 4, np.float32)
            if len(self.executed):
                a = np.asarray(self.executed)[:, ACTION_AXES].reshape(-1)
                past[-len(a):] = a
            context = np.r_[hist, past, self.action[ACTION_AXES]].astype(np.float32)
            action, code, probabilities = self.effect_models.sample(str(request['variant']), context,
                                                                    int(request['rank']), int(request['seed']))
            output = np.zeros((len(action), 6), np.float32); output[:, ACTION_AXES] = action
            # The auxiliary response is metadata for this operation, not an observation feature.
            metadata = np.zeros(50, np.float32); metadata[0] = code
            metadata[1:1 + len(probabilities)] = probabilities
            return output, metadata
        elif op in ('geometry', 'generic'):
            hist = history_inputs(np.asarray(self.features), np.asarray(self.tools), self.config['frames'])[-1]
            goal = (goal_input(request['goals'], request['landmarks'], self.tool, self.base.rotation)
                    if op == 'geometry' else (request['goals'] - self.feature).reshape(-1))
            x = np.r_[hist, goal, float(request['remaining']) / self.config['horizon']].astype(np.float32)
            model, mean, std = self.models[op]
            with torch.no_grad():
                action, _ = residual_action(model, torch.from_numpy((x - mean) / std)[None],
                                             torch.from_numpy(self.action)[None])
            result = action[0].numpy()
            return np.repeat(result, 8, axis=0) if len(result) == 1 else result, self.feature
        elif op == 'reset':
            self.features.clear(); self.tools.clear(); self.executed.clear()
        else:
            raise ValueError(f'Unknown operation {op}')
        return np.tile(self.action, (8, 1)), self.feature


def main():
    output = sys.stdout.buffer
    with redirect_stdout(sys.stderr):
        policy = OutcomePolicy(Path(sys.argv[1]), Path(sys.argv[2]) if len(sys.argv) > 2 else None)
    print('[outcome] ready', file=sys.stderr, flush=True)
    while True:
        header = sys.stdin.buffer.read(8)
        if len(header) != 8:
            return
        size = struct.unpack('<Q', header)[0]
        raw = bytearray()
        while len(raw) < size:
            chunk = sys.stdin.buffer.read(size - len(raw))
            if not chunk:
                return
            raw.extend(chunk)
        with np.load(io.BytesIO(raw), allow_pickle=False) as request, redirect_stdout(sys.stderr):
            action, feature = policy.request(request)
        horizon = policy.effect_models.config['horizon'] if policy.effect_models is not None else 8
        if len(action) != horizon:
            action = np.tile(action, (int(np.ceil(horizon / len(action))), 1))[:horizon]
        output.write(np.r_[action.reshape(-1), feature].astype(np.float32).tobytes()); output.flush()


if __name__ == '__main__':
    main()
