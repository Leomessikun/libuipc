"""Summarize saved response experiments, including a train-mean tangent baseline."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for root in args.runs:
        results = json.loads((root/'results.json').read_text())
        protocol = json.loads((root/'protocol.json').read_text())
        config = protocol['config']
        with np.load(Path(config['data'])/'data.npz') as data:
            train = np.isin(data['episode'], protocol['train_episodes'])
            mean_d = data['tangent'][train].mean(0)
        test_root = Path(config.get('ood_data') or config['data'])
        with np.load(test_root/'data.npz') as data:
            test = np.ones(len(data['episode']), dtype=bool) if config.get('ood_data') else np.isin(data['episode'], protocol['test_episodes'])
            v = data['probe_direction'][test]
            actual = np.einsum('bnca,ba->bnc', data['tangent'][test], v).reshape(len(v), -1)
            pred = np.einsum('nca,ba->bnc', mean_d, v).reshape(len(v), -1)
        cosine = np.sum(actual*pred, axis=1) / (np.linalg.norm(actual, axis=1)*np.linalg.norm(pred, axis=1)+1e-15)
        groups = {}
        for variant in dict.fromkeys(x['variant'] for x in results):
            rows = [x for x in results if x['variant'] == variant]
            metrics = {}
            for key in rows[0]:
                if key not in {'seed', 'variant'} and isinstance(rows[0][key], (int, float)):
                    values = np.array([x[key] for x in rows])
                    metrics[key] = dict(mean=float(values.mean()), std=float(values.std()), n=len(values))
            groups[variant] = metrics
        summary[root.name] = dict(variants=groups, constant_jvp_cosine=float(cosine.mean()),
                                  constant_jvp_relative_rmse=float(np.linalg.norm(actual-pred)/np.linalg.norm(actual)))
    (args.out/'summary.json').write_text(json.dumps(summary, indent=2))
    lines = ['# Response experiment summary', '', 'Mean ± across-seed population standard deviation. These are offline fixed-mesh probes, not RL success rates.', '']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(summary), 2, figsize=(13, max(4, 3.5*len(summary))), squeeze=False)
    for index, (name, record) in enumerate(summary.items()):
        lines += [f'## {name}', '', f"Training-mean tangent cosine: {record['constant_jvp_cosine']:.4f}; relative RMSE: {record['constant_jvp_relative_rmse']:.4f}.", '',
                  '| Variant | Response RMSE (mm) | JVP cosine | Relative JVP RMSE | Dense Q MSE | Response Q MSE |',
                  '|---|---:|---:|---:|---:|---:|']
        groups = record['variants']
        for variant, metrics in groups.items():
            values = []
            for key in ('response_rmse_m', 'jvp_cosine', 'jvp_relative_rmse', 'dense_q_mse', 'response_q_mse'):
                m = metrics[key]; scale = 1000 if key == 'response_rmse_m' else 1
                values.append(f"{m['mean']*scale:.4f} ± {m['std']*scale:.4f}")
            lines.append('| '+variant+' | '+' | '.join(values)+' |')
        lines.append('')
        for ax, key, scale, label in zip(axes[index], ('response_rmse_m', 'jvp_cosine'), (1000, 1), ('Response RMSE (mm), lower is better', 'JVP cosine, higher is better')):
            names = list(groups)
            ax.bar(range(len(names)), [groups[n][key]['mean']*scale for n in names],
                   yerr=[groups[n][key]['std']*scale for n in names], capsize=3, color='#407ba7')
            if key == 'jvp_cosine':
                ax.axhline(record['constant_jvp_cosine'], color='#bd5035', linestyle='--', label='Training-mean tangent')
                ax.legend(fontsize=8)
            ax.set_xticks(range(len(names)), [n.replace('_', '\n') for n in names], fontsize=7)
            ax.set_title(name, fontsize=9); ax.set_ylabel(label, fontsize=9)
            ax.grid(axis='y', alpha=.2)
    fig.tight_layout()
    fig.savefig(args.out/'comparison.png', dpi=160)
    fig.savefig(args.out/'comparison.pdf')
    (args.out/'summary.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
