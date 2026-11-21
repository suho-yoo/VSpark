"""Collect draft training data: target rollouts with calibrated action noise, labeled with clean target outputs.

Each eligible chunk is saved with the target's tokens, final hidden states, tap features,
token embeddings and positions. Collection stops at the per-task chunk count of the paper.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vspark.rollouts import rollouts  # noqa: E402
from vspark.targets import load_target, model_config  # noqa: E402

FIELDS = ('tokens', 'prefix_length', 'hidden', 'embedding', 'positions', 'features')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=['pi0_fast', 'actioncodec', 'starvla'], required=True)
    p.add_argument('--suite', required=True)
    p.add_argument('--task-ids', type=int, nargs='+', default=list(range(10)))
    p.add_argument('--states', required=True, help='Output of prepare_demo_states.py')
    p.add_argument('--noise', default='paper', help="'paper', 'none', or a noise profile from calibrate_noise.py")
    p.add_argument('--actioncodec-repo')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    config = model_config(args.model)
    covariance = (None if args.noise == 'none' else
                  np.asarray(config['noise_covariance'] if args.noise == 'paper' else json.loads(Path(args.noise).read_text())['covariance']))
    target = load_target(args.model, config['draft']['tap_indices'], args.actioncodec_repo)
    support = set(config['support_token_ids'])
    for task_id in args.task_ids:
        quota = config['target_chunks'][args.suite][str(task_id)]
        folder = args.output / args.suite / f'task{task_id:02d}'
        folder.mkdir(parents=True, exist_ok=True)
        records = []
        for episode, step, trace, _ in rollouts(target, args.suite, task_id, args.states,
                                                config['noise_seed'] + 1000 * task_id, covariance, support):
            if trace is None:
                continue
            path = folder / f'episode{episode:03d}_step{step:04d}.pt'
            torch.save({k: trace[k] for k in FIELDS}, path)
            records.append(str(path.relative_to(args.output)))
            if len(records) == quota:
                break
        if len(records) < quota:
            raise RuntimeError(f'Only {len(records)} of {quota} chunks for {args.suite} task {task_id}')
        (folder / 'records.json').write_text(json.dumps(records))
        print(json.dumps(dict(suite=args.suite, task_id=task_id, chunks=len(records))), flush=True)


if __name__ == '__main__':
    main()
