"""Evaluate VSpark (or the autoregressive target with --ar) on LIBERO, LIBERO-Plus or LIBERO-Pro.

Writes one JSON line per episode and a summary with success rate, acceptance length
and decoding time per action.
"""
import argparse
import collections
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vspark import draft as draft_lib  # noqa: E402
from vspark import libero  # noqa: E402
from vspark.decode import generate  # noqa: E402
from vspark.targets import load_target, model_config  # noqa: E402


def timing(episodes, action_steps):
    """Decoding time per executed action and acceptance length.

    Episodes with invalid action outputs and chunks that hit the token limit are excluded.
    """
    seconds = steps = cycles = committed = 0
    for e in episodes:
        if e['error']:
            continue
        for i, capped in enumerate(e['chunk_capped']):
            if capped:
                continue
            seconds += e['chunk_seconds'][i]
            steps += min(action_steps, e['steps'] - i * action_steps)
            if e['chunk_cycles']:
                cycles += e['chunk_cycles'][i]
                committed += e['chunk_committed'][i]
    result = dict(ms_per_action=1000 * seconds / steps)
    if cycles:
        result['acceptance_length'] = committed / cycles
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=['pi0_fast', 'actioncodec', 'starvla'], required=True)
    p.add_argument('--benchmark', choices=['libero', 'plus', 'pro'], default='libero')
    p.add_argument('--suite', required=True, help='e.g. libero_goal; for LIBERO-Pro, libero_goal_lan')
    p.add_argument('--checkpoint', help='VSpark draft checkpoint')
    p.add_argument('--ar', action='store_true', help='Evaluate the target with plain autoregressive decoding')
    p.add_argument('--trials', type=int, default=None, help='Episodes per task (default: 50; 1 for LIBERO-Plus)')
    p.add_argument('--tree-nodes', type=int, default=60)
    p.add_argument('--top-k', type=int, default=8)
    p.add_argument('--actioncodec-repo')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    if args.ar == bool(args.checkpoint):
        raise SystemExit('Pass exactly one of --checkpoint or --ar')
    trials = args.trials or (1 if args.benchmark == 'plus' else 50)

    from libero.libero import benchmark
    np.random.seed(libero.SEED)
    config = model_config(args.model)
    target = load_target(args.model, config['draft']['tap_indices'], args.actioncodec_repo)
    draft = None if args.ar else draft_lib.load(args.checkpoint)
    suite = benchmark.get_benchmark_dict()[args.suite]()
    max_steps = libero.horizon(args.suite)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    episodes = []
    with (out / 'episodes.jsonl').open('w') as log:
        for task_id in range(suite.n_tasks):
            task = suite.get_task(task_id)
            env = libero.make_env(task)
            env.seed(libero.SEED)
            states = libero.initial_states(args.benchmark, suite, task_id)
            try:
                for trial in range(trials):
                    env.reset()
                    env.set_init_state(states[trial])
                    obs = libero.settle(env)
                    actions = collections.deque()
                    record = dict(task_id=task_id, trial=trial, success=False, error=None,
                                  chunk_seconds=[], chunk_cycles=[], chunk_committed=[], chunk_capped=[])
                    for step in range(max_steps):
                        if not actions:
                            batch = target.preprocess(libero.observation(obs, task))
                            torch.cuda.synchronize()
                            start = time.perf_counter()
                            if draft is None:
                                result = target.generate(batch)
                                tokens = result['tokens'].tolist()
                            else:
                                result = generate(target, draft, batch, nodes=args.tree_nodes, top_k=args.top_k)
                                tokens = result['tokens']
                                record['chunk_cycles'].append(len(result['cycles']))
                                record['chunk_committed'].append(sum(c['committed'] for c in result['cycles']))
                            record['chunk_capped'].append(not result['terminated'])
                            try:
                                chunk = target.decode(tokens)
                            except (ValueError, AssertionError) as exc:
                                record['error'] = str(exc)
                            torch.cuda.synchronize()
                            record['chunk_seconds'].append(time.perf_counter() - start)
                            if record['error']:
                                break
                            actions.extend(chunk)
                        obs, _, done, _ = env.step(np.clip(actions.popleft(), -1, 1).tolist())
                        if done:
                            record['success'] = True
                            break
                    record['steps'] = step if record['error'] else step + 1
                    episodes.append(record)
                    log.write(json.dumps(record) + '\n')
                    log.flush()
                    print(json.dumps({k: v for k, v in record.items() if k != 'chunk_seconds'}), flush=True)
            finally:
                env.close()

    summary = dict(model=args.model, benchmark=args.benchmark, suite=args.suite, decoding='ar' if draft is None else 'vspark',
                   episodes=len(episodes), success_rate=sum(e['success'] for e in episodes) / len(episodes),
                   **timing(episodes, target.action_steps))
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
