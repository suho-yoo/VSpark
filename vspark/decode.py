"""VSpark decoding: draft tree, one exact verification pass per cycle, continuation reuse."""
import torch

from .tree import build, continuations, insert


@torch.inference_mode()
def generate(target, draft, batch, depth=7, nodes=60, top_k=8, reuse=True):
    """Greedy speculative decoding; the returned tokens equal the target's own greedy output."""
    h, cache, emb, valid, pos = target.prefill(batch)
    compact = valid[0]
    autocast = torch.autocast('cuda', dtype=torch.bfloat16, enabled=h.is_cuda)
    with autocast:
        context = draft.context(target.last_features[:, compact][:, :-1], pos[:, compact][:, 1:])
    previous = target.last_features[:, -1:]
    pending = int(target.scores(h[:, -1:]).argmax(-1))
    output, cycles, chains = [pending], [], []

    def done():
        return output[-1] in target.end_ids or len(output) >= target.max_tokens

    while not done():
        base = cache.get_seq_length()
        emb = target.embed(torch.tensor([[pending]], device=h.device)).to(h.dtype)
        with autocast:
            block, _ = draft.propose(previous, emb, valid.sum(1, keepdim=True), context, min(depth, target.max_tokens - len(output)))
            tree = build(draft, block, pending, target.end_ids, top_k, nodes, chains if reuse else None)
        if reuse and chains:
            tree = insert(tree, [c[:target.max_tokens - len(output)] for c in chains], nodes)
        ids, mask, depths = tree.tensors(h.device)
        verified, cache, positions = target.consume_tree(ids, cache, valid, mask, depths)
        features = target.last_features
        predictions = target.scores(verified, output[:-1], tree).argmax(-1)[0].tolist()
        path, current = [0], 0
        while len(output) < target.max_tokens:
            child = next((i for i in tree.children(current) if tree.tokens[i] == predictions[current]), None)
            if child is None:
                break
            output.append(tree.tokens[child])
            path.append(child)
            current = child
            if output[-1] in target.end_ids:
                break
        chains = continuations(tree, predictions, current, predictions[current], target.end_ids) if reuse else []
        cache = target.select_cache_path(cache, base, path)
        valid = torch.cat((valid, torch.ones((1, len(path)), device=h.device, dtype=torch.bool)), 1)
        cycles.append(dict(proposed=len(tree.tokens) - 1, accepted=len(path) - 1, committed=len(path)))
        if not done():
            pending = predictions[current]
            output.append(pending)
        if not done():
            chosen = torch.tensor(path, device=h.device)
            selected = features.index_select(1, chosen)
            with autocast:
                context = draft.context(torch.cat((previous, selected[:, :-1]), 1), positions.index_select(1, chosen), context)
            previous = selected[:, -1:]
    return dict(tokens=output, cycles=cycles, terminated=output[-1] in target.end_ids)
