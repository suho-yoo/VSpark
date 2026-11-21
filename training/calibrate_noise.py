"""Calibrate the rollout noise covariance from the draft's own action errors (Eq. 4).

The paper uses 16 clean target chunks per suite (64 in total). In each chunk, up to four
evenly spaced blocks of seven target tokens are replaced by the draft's greedy proposal;
the decoded action difference is one error sample. The covariance
is the uncentered second moment of these errors. Blocks with any |error| > 10 are dropped.
The covariances used in the paper ship in configs/<model>.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vspark import draft as draft_lib  # noqa: E402
from vspark.rollouts import rollouts  # noqa: E402
from vspark.targets import load_target, model_config  # noqa: E402
from vspark.tree import build  # noqa: E402


@torch.inference_mode()
def block_errors(target, draft, trace, reference):
    """Errors of four draft blocks substituted at evenly spaced positions of the chunk."""
    tokens = trace['tokens'].tolist()
    start = trace['prefix_length'] - 1
    h, p = trace['features'].cuda()[None], trace['positions'].cuda()[None]
    e = trace['embedding'].cuda()[None]
    blocks = []
    for first in np.unique(np.linspace(2, max(2, len(tokens) - 9), 4, dtype=int)):
        if first + 8 >= len(tokens):
            continue
        at = start + int(first)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            context = draft.context(h[:, :at], p[:, 1:at + 1])
            block, _ = draft.propose(h[:, at:at + 1], e[:, at + 1:at + 2], p[:, at + 1:at + 2], context, 7)
            tree = build(draft, block, tokens[first], target.end_ids, top_k=1, budget=8)
        if len(tree.tokens) != 8 or any(t in target.end_ids for t in tree.tokens[1:]):
            continue
        try:
            decoded = np.asarray(target.decode(tokens[:first + 1] + tree.tokens[1:] + tokens[first + 8:]), dtype=np.float64)
        except (ValueError, AssertionError, IndexError):
            continue
        if decoded.shape == reference.shape and np.isfinite(decoded).all():
            blocks.append(decoded[:, :6] - reference[:, :6])
    return blocks


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=['pi0_fast', 'actioncodec', 'starvla'], required=True)
    p.add_argument('--checkpoint', required=True, help='A draft trained on clean data')
    p.add_argument('--states', required=True)
    p.add_argument('--suites', nargs='+', default=['libero_goal', 'libero_spatial', 'libero_object', 'libero_10'])
    p.add_argument('--chunks-per-suite', type=int, default=16, help='Clean target chunks per suite, spread evenly over its tasks')
    p.add_argument('--actioncodec-repo')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    config = model_config(args.model)
    target = load_target(args.model, config['draft']['tap_indices'], args.actioncodec_repo)
    draft = draft_lib.load(args.checkpoint)
    support = set(config['support_token_ids'])
    blocks = []
    for suite in args.suites:
        for task_id in range(10):
            quota, seen = args.chunks_per_suite // 10 + (task_id < args.chunks_per_suite % 10), 0
            if quota == 0:
                continue
            for _, _, trace, actions in rollouts(target, suite, task_id, args.states, config['noise_seed'] + 1000 * task_id, None, support):
                if trace is None:
                    continue
                blocks += block_errors(target, draft, trace, actions)
                seen += 1
                if seen == quota:
                    break
    kept = [b for b in blocks if np.abs(b).max() <= 10]
    if not kept:
        raise SystemExit('No valid error blocks: use a clean draft trained to convergence or more chunks per suite')
    errors = np.concatenate(kept)
    second = errors.T @ errors / len(errors)
    eigenvalues, vectors = np.linalg.eigh(second)
    covariance = (vectors * np.maximum(eigenvalues, 1e-10)) @ vectors.T
    Path(args.output).write_text(json.dumps(dict(covariance=covariance.tolist(), observations=len(errors),
                                                 blocks=len(blocks), kept_blocks=len(kept)), indent=2))
    print(json.dumps(dict(observations=len(errors), blocks=len(blocks), kept_blocks=len(kept))))


if __name__ == '__main__':
    main()
