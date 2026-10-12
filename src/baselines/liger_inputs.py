"""Fixed-SID LIGER adaptation helpers; no dataset splitting or label changes."""

import numpy as np


def pack_sequences(sequences, semantic_ids, content, history_items=7, codebook_size=4096):
    """Right-pad for upstream T5 while preserving GRID's last-item target.

    GRID's 32-token sequence includes four GT tokens: seven history items.
    SID offsets reserve zero for T5 padding; they do not change item identity.
    """
    if history_items < 1 or not sequences:
        raise ValueError('Nonempty sequences and positive history budget required')
    if semantic_ids.ndim != 2 or content.ndim != 2 or len(semantic_ids) != len(content):
        raise ValueError('SID and content row identities must align')
    depth = semantic_ids.shape[1]
    b = len(sequences)
    ids = np.zeros((b, history_items * depth + 1), dtype=np.int64)
    mask = np.zeros_like(ids)
    labels = np.empty((b, depth), dtype=np.int64)
    embeds = np.zeros((b, history_items, content.shape[1]), dtype=np.float32)
    targets, histories = [], []
    offsets = np.arange(depth, dtype=np.int64) * codebook_size + 1
    for i, sequence in enumerate(sequences):
        rows = np.asarray(sequence, dtype=np.int64)
        if rows.ndim != 1 or len(rows) < 2 or np.any(rows < 0) or np.any(rows >= len(content)):
            raise ValueError('Invalid history-last row sequence')
        history = rows[:-1][-history_items:]
        target = int(rows[-1])
        codes = np.asarray(semantic_ids[np.append(history, target)], dtype=np.int64)
        if np.any(codes < 0) or np.any(codes >= codebook_size):
            raise ValueError('SID code is outside the configured codebook')
        tokens = (codes[:-1] + offsets).reshape(-1)
        ids[i, :len(tokens)] = tokens
        mask[i, :len(tokens)] = 1
        labels[i] = codes[-1] + offsets
        embeds[i, :len(history)] = content[history].astype(np.float32)
        targets.append(target)
        histories.append(history.tolist())
    return {'input_sids': ids, 'attention_mask_sids': mask,
            'labels_sids': labels, 'input_embeddings': embeds,
            'target_rows': np.asarray(targets), 'history_rows': histories}


def cosine_catalog_logits(query, projected_items, temperature):
    """Equivalent to upstream broadcast cosine scoring for a 2D catalog.

    Avoids B x N x D materialization, but NOT full-catalog projection/logits.
    Call outside autocast, as upstream does. Both inputs retain gradients.
    """
    import torch.nn.functional as F
    if query.ndim != 2 or projected_items.ndim != 2 or temperature <= 0:
        raise ValueError('Expected query/catalog matrices and positive temperature')
    return (F.normalize(query, dim=1)
            @ F.normalize(projected_items, dim=-1).to(query.dtype).T) / temperature


def catalog_memory_lower_bounds(rows, batch=32, width=128, content_dim=128, hidden=768):
    if min(rows, batch, width, content_dim, hidden) < 1:
        raise ValueError('Dimensions must be positive')
    # Individual buffers, not a summed peak estimate. Excludes optimizer and graph.
    return {'raw_content_fp32_bytes': rows * content_dim * 4,
            'projection_first_hidden_fp32_bytes': rows * hidden * 4,
            'projected_items_fp32_bytes': rows * width * 4,
            'broadcast_product_fp16_bytes': batch * rows * width * 2,
            'matmul_logits_fp16_bytes': batch * rows * 2}
