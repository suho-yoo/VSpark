"""VSpark draft: a one-layer parallel drafter with a Markov transition head.

The draft attends to the target's multi-layer hidden features through a
teacher-only KV context and proposes a block of tokens in one forward pass.
"""
from dataclasses import asdict, dataclass, field

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class DraftConfig:
    width: int = 2048
    intermediate: int = 16384
    heads: int = 8
    kv_heads: int = 1
    eps: float = 1e-6
    rope_theta: float = 10000.0
    tap_indices: list = field(default_factory=lambda: [2, 5, 9, 13, 16])
    proposal_tokens: int = 7
    markov_rank: int = 256
    input_vocab: int = 257152
    end_token_id: int = 235371
    ce_weight: float = 0.1
    feature_weight: float = 0.9
    loss_decay_gamma: float = 4.0
    num_anchors: int = 512

    def to_dict(self):
        return asdict(self)


def rms(x, weight, eps):
    return (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps) * weight).to(x.dtype)


def rope(x, positions, theta):
    d = x.shape[-1]
    angles = positions.float()[..., None] * (theta ** (-torch.arange(0, d, 2, device=x.device).float() / d))
    angles = torch.cat((angles, angles), -1)[:, None]
    rotated = torch.cat((-x[..., d // 2:], x[..., :d // 2]), -1)
    return x * angles.cos().to(x.dtype) + rotated * angles.sin().to(x.dtype)


class Block(nn.Module):
    """Decoder block whose input is the concatenation of token embedding and target feature."""

    def __init__(self, c):
        super().__init__()
        self.c = c
        w, d = c.width, c.width // c.heads
        self.q = nn.Linear(2 * w, w, bias=False)
        self.k = nn.Linear(2 * w, c.kv_heads * d, bias=False)
        self.v = nn.Linear(2 * w, c.kv_heads * d, bias=False)
        self.o = nn.Linear(w, w, bias=False)
        self.attn_norm = nn.Parameter(torch.ones(w))
        self.hidden_norm = nn.Parameter(torch.ones(w))
        self.norm = nn.Parameter(torch.ones(w))
        self.gate = nn.Linear(w, c.intermediate, bias=False)
        self.up = nn.Linear(w, c.intermediate, bias=False)
        self.down = nn.Linear(c.intermediate, w, bias=False)

    def kv(self, x, positions):
        c = self.c
        b, s, _ = x.shape
        d = c.width // c.heads
        k = self.k(x).view(b, s, c.kv_heads, d).transpose(1, 2)
        v = self.v(x).view(b, s, c.kv_heads, d).transpose(1, 2)
        return rope(k, positions, c.rope_theta), v

    def forward(self, x, positions, cache, mask):
        c = self.c
        b, s, _ = x.shape
        d = c.width // c.heads
        e, h = x.chunk(2, -1)
        h = rms(h, self.hidden_norm, c.eps)
        z = torch.cat((rms(e, self.attn_norm, c.eps), h), -1)
        q = rope(self.q(z).view(b, s, c.heads, d).transpose(1, 2), positions, c.rope_theta)
        k, v = self.kv(z, positions)
        k, v = torch.cat((cache[0], k), 2), torch.cat((cache[1], v), 2)
        repeats = c.heads // c.kv_heads
        if repeats != 1:
            k, v = k.repeat_interleave(repeats, 1), v.repeat_interleave(repeats, 1)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = h + self.o(y.transpose(1, 2).reshape(b, s, c.width))
        z = rms(x, self.norm, c.eps)
        return x + self.down(F.silu(self.gate(z)) * self.up(z))


class Draft(nn.Module):
    def __init__(self, config, head_weight, token_ids, mask_embedding):
        """head_weight: target LM-head rows of the action vocabulary; token_ids: their ids."""
        super().__init__()
        c = self.config = config
        w = c.width
        self.register_buffer('teacher_rows', head_weight.to(torch.bfloat16).clone())
        self.register_buffer('token_ids', token_ids.long().clone())
        self.head = nn.Parameter(head_weight.to(torch.bfloat16).clone(), requires_grad=False)
        self.mask_embedding = nn.Parameter(mask_embedding.float().clone(), requires_grad=False)
        self.fusion = nn.Linear(len(c.tap_indices) * w, w, bias=False)
        self.layers = nn.ModuleList([Block(c)])
        self.context_norm = nn.Parameter(torch.ones(w))
        self.final_norm = nn.Parameter(torch.ones(w))
        self.future_hidden = nn.Parameter(torch.randn(len(c.tap_indices) * w))
        self.markov_embedding = nn.Embedding(c.input_vocab, c.markov_rank)
        self.markov_projection = nn.Linear(c.markov_rank, len(token_ids), bias=False)
        nn.init.normal_(self.markov_embedding.weight, std=.02)

    def fuse(self, h):
        return self.fusion(h) if h.shape[-1] != self.config.width else h

    def context(self, h, positions, cache=None):
        """Append target features to the teacher-only draft KV cache (RoPE at position - 1)."""
        layer = self.layers[0]
        h = self.fuse(h)
        normalized = rms(h, layer.hidden_norm, self.config.eps)
        k, v = layer.kv(torch.cat((torch.zeros_like(normalized), normalized), -1), positions - 1)
        if cache is not None:
            k, v = torch.cat((cache[0], k), 2), torch.cat((cache[1], v), 2)
        return k, v

    def propose(self, h, embedding, positions, cache, k, key_valid=None):
        """Root feature/embedding plus k-1 mask slots, one bidirectional block pass."""
        b = h.shape[0]
        past = 0 if cache is None else cache[0].shape[2]
        context = self.context(h, positions, cache)
        inputs = torch.cat((h, self.future_hidden.to(h.dtype)[None, None].expand(b, k - 1, -1)), 1)
        slots = torch.cat((embedding, self.mask_embedding.to(embedding.dtype)[None, None].expand(b, k - 1, -1)), 1)
        slot_positions = positions[:, :1] + torch.arange(k, device=h.device)[None]
        if key_valid is None:
            valid = torch.ones(b, past + 1 + k, device=h.device, dtype=torch.bool)
        else:
            root = torch.ones(b, 1, device=h.device, dtype=torch.bool)
            valid = torch.cat((key_valid[:, :past], root, key_valid[:, past:]), 1)
        mask = valid[:, None, None, :].expand(-1, 1, k, -1)
        x = self.layers[0](torch.cat((slots, self.fuse(inputs)), -1), slot_positions, context, mask)
        return rms(x, self.final_norm, self.config.eps), context

    def logits(self, hidden):
        return F.linear(hidden, self.head)

    def markov(self, scores, previous_ids):
        """Add the rank-r transition bias of the preceding token (zero outside the vocabulary)."""
        n = self.config.input_vocab
        valid = (previous_ids >= 0) & (previous_ids < n)
        embedding = self.markov_embedding(previous_ids.clamp(0, n - 1)) * valid[..., None]
        return scores + self.markov_projection(embedding)

    def classes(self, ids):
        match = ids[..., None] == self.token_ids
        if not bool(match.any(-1).all()):
            raise ValueError('Token outside the draft vocabulary')
        return match.long().argmax(-1)


def initialize(model):
    """Training initialization: default PyTorch Linear init, N(0, 0.02) Markov embedding."""
    for module in model.modules():
        if isinstance(module, nn.Linear):
            module.reset_parameters()
    nn.init.normal_(model.markov_embedding.weight, std=.02)
    return model


def save(path, model, **extra):
    torch.save(dict(config=model.config.to_dict(), model=model.state_dict(), **extra), path)


def load(path, device='cuda'):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    config = DraftConfig(**saved['config'])
    state = saved['model']
    model = Draft(config, state['head'], state['token_ids'], state['mask_embedding'])
    model.load_state_dict(state)
    return model.to(device).eval()
