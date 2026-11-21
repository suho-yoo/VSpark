"""Train the VSpark draft on collected rollouts (CE + feature alignment, Eq. 5).

AdamW (0.9, 0.95), peak LR 1e-4, log warmup over the first 5% then linear decay, batch 16, gradient clipping 0.5,
10 epochs. The target LM head (action rows) is frozen and shared with the draft.
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vspark import draft as draft_lib  # noqa: E402
from vspark.loss import loss_batch  # noqa: E402
from vspark.targets import load_target, model_config  # noqa: E402

WARMUP_FRACTION = 0.05


class WarmupDecay:
    """Logarithmic warmup to the peak learning rate, then linear decay to zero."""

    def __init__(self, optimizer, peak, warmup, total):
        self.optimizer, self.peak, self.warmup, self.total, self.step_count = optimizer, peak, warmup, total, -1
        self.step()

    def step(self):
        self.step_count += 1
        s = self.step_count
        fraction = math.log(s + 1) / math.log(self.warmup) if s < self.warmup else max(0., (self.total - s) / (self.total - self.warmup))
        for group in self.optimizer.param_groups:
            group['lr'] = self.peak * fraction


def target_assets(name, config, actioncodec_repo):
    """Action-vocabulary rows of the target LM head and the mask-token input embedding."""
    target = load_target(name, config['draft']['tap_indices'], actioncodec_repo, device='cpu')
    ids = torch.tensor(config['support_token_ids'])
    head = target.head.weight[ids].detach().to(torch.bfloat16)
    mask = (target.lm.embed_tokens.weight[config['mask_token_id']].detach().to(torch.bfloat16) * config['mask_embedding_scale']).to(torch.bfloat16)
    del target
    return head, ids, mask


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=['pi0_fast', 'actioncodec', 'starvla'], required=True)
    p.add_argument('--data', type=Path, required=True, help='Output of collect_rollouts.py')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--microbatch', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--actioncodec-repo')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    config = model_config(args.model)
    head, ids, mask = target_assets(args.model, config, args.actioncodec_repo)
    model = draft_lib.initialize(draft_lib.Draft(draft_lib.DraftConfig(**config['draft']), head, ids, mask)).cuda().train()
    optimizer = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=args.lr, betas=(.9, .95), eps=1e-8,
                                  weight_decay=0., foreach=False)
    records = sorted(str(args.data / r) for f in args.data.glob('*/task*/records.json') for r in json.loads(f.read_text()))
    updates = len(records) // args.batch_size
    total = updates * args.epochs
    scheduler = WarmupDecay(optimizer, args.lr, max(2, round(total * WARMUP_FRACTION)), total)
    for epoch in range(1, args.epochs + 1):
        order = list(records)
        random.Random(1000 + epoch).shuffle(order)
        start, running = time.time(), 0.
        for update in range(updates):
            samples = [{k: v.cuda() if torch.is_tensor(v) else v for k, v in torch.load(path, weights_only=False).items()}
                       for path in order[update * args.batch_size:(update + 1) * args.batch_size]]
            optimizer.zero_grad(set_to_none=True)
            loss_sum, weight = 0., 0.
            for i in range(0, len(samples), args.microbatch):
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    value, stats = loss_batch(model, samples[i:i + args.microbatch])
                value.backward()
                loss_sum, weight = loss_sum + float(value.detach()), weight + float(stats[2])
            for q in model.parameters():
                if q.grad is not None:
                    q.grad.div_(weight)
            torch.nn.utils.clip_grad_norm_(model.parameters(), .5, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            running += loss_sum / weight
            if (update + 1) % 100 == 0:
                print(json.dumps(dict(epoch=epoch, update=update + 1, of=updates, loss=running / (update + 1),
                                      lr=optimizer.param_groups[0]['lr'], minutes=(time.time() - start) / 60)), flush=True)
        draft_lib.save(args.output / f'epoch_{epoch:03d}.pt', model, epoch=epoch)
    print(json.dumps(dict(checkpoint=str(args.output / f'epoch_{args.epochs:03d}.pt'))))


if __name__ == '__main__':
    main()
