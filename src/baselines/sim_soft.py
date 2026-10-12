"""ID-only SIM-soft: paper equations 1--5, without unavailable time features.

The embedding is injected so this scorer has no data/Lightning dependencies.
GSU uses unnormalised dot-product pooling, NOT ESU's softmax attention.
"""

import torch
from torch import nn
from torch.nn import functional as F


def prediction_head(input_dim):
    # A binary logit is equivalent to the difference of two softmax logits.
    return nn.Sequential(nn.Linear(input_dim, 200), nn.PReLU(),
                         nn.Linear(200, 80), nn.PReLU(), nn.Linear(80, 1))


def supervised_bce(logits, labels):
    """Same supervised rows as the existing no-hit candidate protocol.

    Natural no-hit rows are excluded; a separate, training-only injected list
    supplies their supervision. Reduction is mean over rows AND candidates.
    """
    valid = labels.sum(-1) > 0
    if not valid.any():
        return logits.sum() * 0.0
    return F.binary_cross_entropy_with_logits(logits[valid].float(), labels[valid].float())


class SIMSoftScorer(nn.Module):
    def __init__(self, item_embedding, embedding_dim=128, top_k=10, num_heads=4):
        super().__init__()
        if min(embedding_dim, top_k, num_heads) <= 0:
            raise ValueError("SIM dimensions, top_k and num_heads must be positive")
        self.item_embedding = item_embedding
        self.embedding_dim, self.top_k, self.num_heads = embedding_dim, top_k, num_heads
        self.gsu_history = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.gsu_query = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.esu_history = nn.Linear(embedding_dim, num_heads * embedding_dim, bias=False)
        self.esu_query = nn.Linear(embedding_dim, num_heads * embedding_dim, bias=False)
        self.gsu_head = prediction_head(2 * embedding_dim)
        self.esu_head = prediction_head((num_heads + 1) * embedding_dim)

    def score(self, hist_itemkey, candidate_itemkey, *, auxiliary=False):
        if hist_itemkey.ndim != 2 or candidate_itemkey.ndim != 2:
            raise ValueError("Expected batch x history and batch x candidates")
        if hist_itemkey.shape[0] != candidate_itemkey.shape[0]:
            raise ValueError("History and candidates must share the batch dimension")
        # Empty histories use one invalid placeholder; all pooling stays zero.
        if not hist_itemkey.shape[1]:
            hist_itemkey = hist_itemkey.new_full((hist_itemkey.shape[0], 1), -1)
        valid = hist_itemkey != -1
        hist = self.item_embedding(hist_itemkey) * valid.unsqueeze(-1)
        query = self.item_embedding(candidate_itemkey)
        relevance = torch.einsum('bcd,bld->bcl', self.gsu_query(query), self.gsu_history(hist))
        indices = relevance.masked_fill(~valid[:, None, :], float('-inf')).topk(
            min(self.top_k, hist.shape[1]), dim=-1).indices
        rows = torch.arange(hist.shape[0], device=hist.device)[:, None, None]
        selected = hist[rows, indices]
        selected_mask = valid[rows, indices]
        keys = self.esu_history(selected).unflatten(-1, (self.num_heads, self.embedding_dim))
        queries = self.esu_query(query).unflatten(-1, (self.num_heads, self.embedding_dim))
        attention = torch.einsum('bcthd,bchd->bcht', keys, queries).float()
        mask = selected_mask.unsqueeze(2)
        attention = attention.masked_fill(~mask, torch.finfo(attention.dtype).min).softmax(-1)
        attention = attention * mask
        interest = torch.einsum('bcht,bctd->bchd', attention.to(selected.dtype), selected).flatten(-2)
        result = {
            'esu_logits': self.esu_head(torch.cat((interest, query), -1)).squeeze(-1),
            'selected_indices': indices, 'selected_mask': selected_mask,
        }
        if auxiliary:
            # Keep equation 2's unnormalised sum, including its gradient to GSU.
            weights = relevance * valid[:, None, :]
            pooled = torch.einsum('bcl,bld->bcd', weights, hist)
            result['gsu_logits'] = self.gsu_head(torch.cat((pooled, query), -1)).squeeze(-1)
        return result

    def forward(self, hist_itemkey, candidate_itemkey, candidate_sid=None, beam_scores=None):
        # These compatibility arguments MUST NOT become SIM features.
        return self.score(hist_itemkey, candidate_itemkey)['esu_logits']
