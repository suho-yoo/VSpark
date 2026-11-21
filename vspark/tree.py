"""Draft trees: beam expansion over the Markov head, budget pruning and exact early stopping."""
from dataclasses import dataclass
import math

import torch


@dataclass
class Tree:
    tokens: list
    parents: list
    depths: list
    scores: list

    def tensors(self, device):
        """Token ids, ancestor mask (each node attends to itself and its ancestors) and depths."""
        n, rows = len(self.tokens), []
        for i, p in enumerate(self.parents):
            row = rows[p].copy() if p >= 0 else [False] * n
            row[i] = True
            rows.append(row)
        return (torch.tensor([self.tokens], device=device), torch.tensor(rows, dtype=torch.bool, device=device),
                torch.tensor([self.depths], device=device))

    def children(self, node):
        return [i for i, p in enumerate(self.parents) if p == node]

    def path(self, node):
        """Tokens from the root to `node`, inclusive."""
        tokens = []
        while node >= 0:
            tokens.append(self.tokens[node])
            node = self.parents[node]
        return tokens[::-1]


def prune(tokens, parents, depths, scores, budget):
    """Keep the root and the budget - 1 highest-scoring nodes; ancestors always precede descendants."""
    chosen = sorted(sorted(range(1, len(tokens)), key=lambda i: (-scores[i], i))[:max(0, budget - 1)])
    selected = [0] + chosen
    index = {old: new for new, old in enumerate(selected)}
    return Tree([tokens[i] for i in selected], [-1] + [index[parents[i]] for i in chosen],
                [depths[i] for i in selected], [scores[i] for i in selected])


def stop_extending(frontier, scores, budget):
    """True once no deeper node can enter the pruned tree (children never outscore parents)."""
    best = min(frontier, key=lambda i: (-scores[i], i))
    if len(scores) - 1 < budget - 1:
        return False
    kth = sorted(range(1, len(scores)), key=lambda i: (-scores[i], i))[budget - 2]
    return (-scores[best], best) > (-scores[kth], kth)


@torch.inference_mode()
def build(draft, block, root, end_ids, top_k=8, budget=60, continuations=None):
    """Project the proposal block once, then condition each depth on its concrete parents."""
    base = draft.logits(block)[0]
    tokens, parents, depths, scores, frontier = [int(root)], [-1], [0], [0.], [0]
    for d in range(block.shape[1]):
        live = [p for p in frontier if tokens[p] not in end_ids]
        if not live:
            break
        previous = torch.tensor([tokens[p] for p in live], device=block.device)
        logprobs = draft.markov(base[d].expand(len(live), -1), previous).float().log_softmax(-1)
        if continuations:
            logprobs = promote(draft, logprobs, live, tokens, parents, continuations, d)
        order = logprobs.argsort(dim=-1, descending=True, stable=True)[:, :min(top_k, logprobs.shape[-1])]
        ids, values = draft.token_ids[order].tolist(), logprobs.gather(1, order).tolist()
        new = []
        for row, parent in enumerate(live):
            for token, value in zip(ids[row], values[row]):
                new.append(len(tokens))
                tokens.append(token)
                parents.append(parent)
                depths.append(d + 1)
                scores.append(scores[parent] + value)
        frontier = sorted(new, key=lambda i: (-scores[i], i))[:top_k]
        if frontier and stop_extending(frontier, scores, budget):
            break
    return prune(tokens, parents, depths, scores, budget)


def promote(draft, logprobs, live, tokens, parents, continuations, depth):
    """Raise continuation tokens to the best draft score under the same parent (Eq. 6)."""
    index = getattr(draft, '_vocab_index', None)
    if index is None:
        index = draft._vocab_index = {int(t): i for i, t in enumerate(draft.token_ids.tolist())}
    pairs = set()
    for row, node in enumerate(live):
        path = []
        while node > 0:
            path.append(tokens[node])
            node = parents[node]
        path.reverse()
        for chain in continuations:
            if len(chain) > depth and chain[:depth] == path and int(chain[depth]) in index:
                pairs.add((row, index[int(chain[depth])]))
    if not pairs:
        return logprobs
    rows, cols = map(list, zip(*sorted(pairs)))
    logprobs = logprobs.clone()
    r, c = torch.tensor(rows, device=logprobs.device), torch.tensor(cols, device=logprobs.device)
    logprobs[r, c] = torch.maximum(logprobs[r, c], logprobs[r].max(-1).values.clamp_max(0.))
    return logprobs


def insert(tree, chains, budget):
    """Merge continuation chains under the root, dropping the lowest-scoring draft nodes to stay in budget."""
    chains = [list(c) for c in chains if c]
    if not chains:
        return tree

    def child(parents_, tokens_, node, token):
        return next((i for i, p in enumerate(parents_) if p == node and tokens_[i] == token), None)

    def missing(t):
        new = set()
        for chain in chains:
            node = 0
            for d, token in enumerate(chain):
                nxt = child(t.parents, t.tokens, node, token)
                if nxt is None:
                    new.update(tuple(chain[:e + 1]) for e in range(d, len(chain)))
                    break
                node = nxt
        return new

    keep, pruned = budget - len(missing(tree)), tree
    for _ in range(4):
        pruned = prune(tree.tokens, tree.parents, tree.depths, tree.scores, max(1, keep)) if len(tree.tokens) > keep else tree
        overflow = len(pruned.tokens) + len(missing(pruned)) - budget
        if overflow <= 0:
            break
        keep -= overflow
    tokens, parents, depths, scores = map(list, (pruned.tokens, pruned.parents, pruned.depths, pruned.scores))
    for chain in chains:
        node = 0
        for token in chain:
            nxt = child(parents, tokens, node, token)
            if nxt is None:
                if len(tokens) >= budget:
                    break
                tokens.append(token)
                parents.append(node)
                depths.append(depths[node] + 1)
                scores.append(scores[node] + math.log(.5))
                nxt = len(tokens) - 1
            node = nxt
    return Tree(tokens, parents, depths, scores)


def continuations(tree, predictions, end, correction, end_ids, max_length=7):
    """Target predictions along each rejected sibling of the correction (Sec. 3.4).

    From a rejected node, follow the child that equals the target's prediction and
    keep the final prediction when no child matches.
    """
    siblings = sorted((i for i in tree.children(end) if tree.tokens[i] != correction), key=lambda i: (-tree.scores[i], i))
    if correction in end_ids or not siblings:
        return []
    chains = []
    for node in siblings:
        chain = []
        while len(chain) < max_length:
            token = predictions[node]
            chain.append(token)
            node = next((i for i in tree.children(node) if tree.tokens[i] == token), None)
            if node is None:
                break
        stop = next((i for i, t in enumerate(chain) if t in end_ids), None)
        chains.append(chain if stop is None else chain[:stop + 1])
    return chains
