"""Training objective: sampled-anchor block loss with CE and feature alignment (Eq. 5)."""
import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.checkpoint import checkpoint


def sample_anchor_positions(seq_len, loss_mask, num_anchors, device):
    """Uniformly sample up to num_anchors valid anchors per row (from DeepSpec, MIT)."""
    n = max(seq_len - 1, 0)
    valid = (loss_mask[:, :n] > .5) & (loss_mask[:, 1:n + 1] > .5)
    counts = valid.sum(1)
    b = loss_mask.shape[0]
    if n == 0:
        return torch.zeros(b, num_anchors, dtype=torch.long, device=device), torch.zeros(b, num_anchors, dtype=torch.bool, device=device)
    indices = torch.arange(n, device=device)[None].expand(b, -1)
    masked = torch.where(valid, indices, torch.full_like(indices, seq_len + 1))
    noise = torch.where(valid, torch.rand(b, n, device=device), torch.full((b, n), 2., device=device))
    gathered = torch.gather(masked, 1, noise.sort(dim=1)[1])
    if n < num_anchors:
        gathered = torch.cat((gathered, torch.full((b, num_anchors - n), seq_len + 1, dtype=gathered.dtype, device=device)), 1)
    anchors = gathered[:, :num_anchors].sort(dim=1).values
    keep = torch.arange(num_anchors, device=device)[None] < counts[:, None].clamp(max=num_anchors)
    return torch.where(keep, anchors, torch.zeros_like(anchors)), keep


def loss_batch(model, samples, anchor_batch=128):
    """Weighted sum of block losses and its weight, over all sampled anchors of the samples.

    Each sample holds one target chunk: final hidden states `hidden`, tap features `features`,
    token embeddings `embedding`, positions, generated `tokens` and the prompt `prefix_length`.
    """
    c = model.config
    device = samples[0]['hidden'].device
    h = pad_sequence([s['hidden'] for s in samples], batch_first=True)
    e = pad_sequence([s['embedding'] for s in samples], batch_first=True)
    f = pad_sequence([s['features'] for s in samples], batch_first=True)
    p = pad_sequence([s['positions'][1:] for s in samples], batch_first=True)
    lengths = torch.tensor([len(s['hidden']) - 1 for s in samples], device=device)
    starts = torch.tensor([s['prefix_length'] - 1 for s in samples], device=device)
    context = model.context(f[:, :-1], p)
    seq_len = h.shape[1] + 1
    indices = torch.arange(seq_len, device=device)[None]
    action_mask = (indices >= starts[:, None] + 1) & (indices <= lengths[:, None] + 1)
    anchors, keep = sample_anchor_positions(seq_len, action_mask, c.num_anchors, device)
    roots = [anchors[i, keep[i]] - 1 for i in range(len(samples))]
    rows = torch.cat([torch.full_like(r, i) for i, r in enumerate(roots)])
    roots = torch.cat(roots)
    filler = int(model.token_ids[0])
    tokens = pad_sequence([s['tokens'] for s in samples], batch_first=True, padding_value=filler)
    token_lengths = torch.tensor([len(s['tokens']) for s in samples], device=device)
    k = c.proposal_tokens
    offset = torch.arange(k, device=device)
    decay = torch.exp(-offset.float() / c.loss_decay_gamma)

    def group(row, at, *kv):
        mask = at[:, None] + offset < lengths[row, None]
        idx = (at[:, None] + offset).minimum((lengths[row] - 1)[:, None])
        size = int(at.max())
        kv = tuple(t[row, :, :size] for t in kv)
        key_valid = torch.cat((torch.arange(size, device=device)[None] < at[:, None], torch.ones_like(mask)), 1)
        out, _ = model.propose(f[row, at][:, None], e[row, at + 1][:, None], p[row, at][:, None], kv, k, key_valid)
        ti = (at - starts[row])[:, None] + offset
        last = token_lengths[row, None] - 1
        inside = ti + 1 < token_lengths[row, None]
        truth = torch.where(inside, tokens[row[:, None], (ti + 1).clamp_min(0).minimum(last)], filler)
        previous = torch.where(inside, tokens[row[:, None], ti.clamp_min(0).minimum(last)], filler)
        logits = model.markov(model.logits(out), previous).float()
        ce = F.cross_entropy(logits.transpose(1, 2), model.classes(truth), reduction='none')
        feature = F.smooth_l1_loss(out.float(), h[row[:, None], idx + 1].float(), reduction='none').mean(-1)
        weights = mask.float() * decay
        value = ((c.ce_weight * ce + c.feature_weight * feature) * weights).sum()
        return value, torch.stack(((ce.detach() * weights).sum(), (feature.detach() * weights).sum(), weights.sum()))

    total, stats = model.head.sum() * 0., torch.zeros(3, device=device)
    for lo in range(0, len(roots), anchor_batch):
        args = (rows[lo:lo + anchor_batch], roots[lo:lo + anchor_batch], *context)
        value, st = checkpoint(group, *args, use_reentrant=False) if torch.is_grad_enabled() else group(*args)
        total, stats = total + value, stats + st
    return total, stats
