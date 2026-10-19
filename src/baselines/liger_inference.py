"""GT-free official LIGER generation followed by exact full-cold retrieval."""
import numpy as np
import torch

from src.baselines.liger_candidates import unique_beam_rows
from src.baselines.liger_retrieval import full_cold_rank


@torch.no_grad()
def generate_queries(model, model_inputs, method, official_evaluation, sid_index, device, beam_size=20):
    if model.training or beam_size != 20:
        raise ValueError('Expected eval model and frozen beam20 protocol')
    device = torch.device(device)
    keys = ('input_sids', 'attention_mask_sids', 'input_embeddings')
    batch = {key: torch.as_tensor(model_inputs[key]) for key in keys}
    size = len(batch['input_sids'])
    # Upstream model_forward reads this field even in skip_forward mode. Supply
    # dummy values, never the real GT, and do not pass labels into generate().
    batch['labels_sids'] = torch.zeros((size, 4), dtype=torch.long)
    _, input_kwargs = official_evaluation.model_forward(model, batch, device, 4, method, skip_forward=True)
    with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
        outputs = model.generate(**input_kwargs, num_beams=beam_size, num_return_sequences=beam_size,
                                 max_new_tokens=4, use_cache=True)
    if outputs.ndim != 2 or len(outputs) != size * beam_size or outputs.shape[1] > 5:
        raise ValueError('Unexpected official generation shape')
    tokens = np.zeros((size * beam_size, 4), dtype=np.int64)
    emitted = outputs[:, 1:5].cpu().numpy()
    tokens[:, :emitted.shape[1]] = emitted
    tokens = tokens.reshape(size, beam_size, 4)
    raw_rows = sid_index.lookup_tokens(tokens)
    query = model.predicted_embedding.reshape(size, beam_size, -1)[:, 0].float()
    if not torch.isfinite(query).all().item():
        raise ValueError('Nonfinite encoder query')
    return query, dict(beam_tokens=tokens, raw_beam_rows=raw_rows,
                       valid_beam_rows=unique_beam_rows(raw_rows))


@torch.no_grad()
def predict(model, model_inputs, method, official_evaluation, sid_index, cache, device):
    query, generation = generate_queries(model, model_inputs, method, official_evaluation, sid_index, device)
    ranked = full_cold_rank(cache, query, generation['valid_beam_rows'], k=20)
    return dict(**generation, **{key: value for key, value in ranked.items() if key != 'valid_beam_rows'})
