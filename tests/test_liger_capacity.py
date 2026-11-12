import json
from pathlib import Path

import pytest
import torch

from src.baselines.liger_capacity import (
    CAPACITY384, CAPACITY384_BACKBONE_PARAMETERS, CAPACITY384_PARAMETERS,
    build_capacity_model, parameter_report, validate_architecture, validate_probe_config,
)


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT.parent / 'liger-upstream'
TINY = dict(d_model=16, d_ff=32, num_layers=1, num_decoder_layers=1,
            num_heads=2, d_kv=8)


def test_architecture_validation_copies_and_accepts_original_nondivisible_width():
    original = dict(CAPACITY384, d_model=128, d_ff=1024)
    assert validate_architecture(original) == original
    assert validate_architecture(original) is not original


@pytest.mark.parametrize('key', list(CAPACITY384))
@pytest.mark.parametrize('bad', [0, -1, True, 1.5, '384', None])
def test_invalid_dimensions(key, bad):
    with pytest.raises(ValueError):
        validate_architecture(dict(CAPACITY384, **{key: bad}))


@pytest.mark.parametrize('bad', [None, {}, [], dict(CAPACITY384, checkpoint='old.pt'),
                               {k: v for k, v in CAPACITY384.items() if k != 'd_kv'}])
def test_missing_or_extra_architecture_keys(bad):
    with pytest.raises(ValueError):
        validate_architecture(bad)


@pytest.mark.parametrize('seed,content_dim', [(-1, 128), (True, 128), (1.2, 128),
                                           (2**63, 128), (42, 0), (42, False)])
def test_invalid_seed_or_content(seed, content_dim):
    with pytest.raises(ValueError):
        build_capacity_model(UPSTREAM, TINY, seed=seed, content_dim=content_dim)


def test_384_exact_meta_parameter_counts_tying_and_unchanged_semantics():
    model, method, evaluation, report = build_capacity_model(UPSTREAM, CAPACITY384, device='meta')
    assert all(p.device.type == 'meta' for p in model.parameters())
    assert model.training
    assert report['total_params'] == report['trainable_params'] == CAPACITY384_PARAMETERS
    assert report['backbone_params'] == CAPACITY384_BACKBONE_PARAMETERS
    assert report['parameter_tensors'] == 152
    assert sum(report['disjoint_groups'].values()) == report['total_params']
    assert report['tied_sid_input_output']
    assert model.config.pad_token_id == model.config.decoder_start_token_id == 0
    assert model.config.eos_token_id == 16385 and model.config.vocab_size == 16386
    assert model.config.dropout_rate == .2 and model.config.initializer_factor == .02
    assert model.input_embed_dropout.p == .5
    assert model.semantic_pos.num_embeddings == 4
    assert model.pos_embedding.num_embeddings == 7
    assert method['embedding_loss_weight'] == method['sid_loss_weight'] == 1
    assert method['embedding_head_dict']['text_embedding_dim'] == 128
    assert callable(evaluation.model_forward)


def test_baseline_meta_size_remains_exact():
    model, _, _, report = build_capacity_model(
        UPSTREAM, dict(CAPACITY384, d_model=128, d_ff=1024), device='meta')
    assert report['total_params'] == 10070912
    assert report['backbone_params'] == 6689152


def test_fresh_cpu_seed_reproducibility_and_no_checkpoint_api():
    first, _, _, _ = build_capacity_model(UPSTREAM, TINY, seed=42)
    second, _, _, _ = build_capacity_model(UPSTREAM, TINY, seed=42)
    assert all(p.device.type == 'cpu' for p in first.parameters())
    for key, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[key], atol=0, rtol=0)
    third, _, _, _ = build_capacity_model(UPSTREAM, TINY, seed=43)
    assert not torch.equal(first.shared.weight, third.shared.weight)
    with pytest.raises(TypeError):
        build_capacity_model(UPSTREAM, TINY, checkpoint='old.pt')


def test_parameter_report_rejects_accidental_untied_head():
    model, _, _, _ = build_capacity_model(UPSTREAM, TINY, device='meta')
    model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.clone())
    with pytest.raises(ValueError, match='tied'):
        parameter_report(model)


def test_small_cpu_forward_with_all_four_sid_labels():
    model, _, _, _ = build_capacity_model(UPSTREAM, TINY)
    model.eval()
    labels = torch.tensor([[1, 4097, 8193, 12289], [2, 4098, 8194, 12290]])
    inputs = torch.tensor([[3, 4099, 8195, 12291], [4, 4100, 8196, 12292]])
    with torch.no_grad():
        result = model(input_ids=inputs, attention_mask=torch.ones_like(inputs), labels=labels)
    assert result.logits.shape == (2, 4, 16386)
    assert torch.isfinite(result.loss)
    assert model.predicted_embedding.shape == (2, 16)


def test_probe_configuration_is_bounded_and_does_not_authorize_training():
    config = json.loads((ROOT / 'configs/baselines/liger_capacity384_probe_v1.json').read_text())
    assert validate_probe_config(config) == config
    assert validate_probe_config(config) is not config
    assert config['model'] == CAPACITY384
    assert not config['formal_training_ready'] and not config['automatic_successor']
    assert config['expected_parameters'] == dict(total=41868160, trainable=41868160,
                                                transformer_backbone=34222464)
    train = config['training']
    assert train['logical_batch_size'] == 2048
    assert train['physical_batch_sizes'] == [512, 1024]
    assert train['fixture_prefixes'] == 2048 * (train['warmup_attempts'] + train['measured_attempts'])
    assert train['fixture_prefixes'] <= 32768
    assert train['max_attempts_per_case'] == 13
    assert config['dense_sampling'] == dict(num_negatives=32768, seed=42)
    assert config['validation']['records'] == 65
    assert config['validation']['cold_rows'] == 23915712
    assert not config['validation']['test_access']
    assert config['validation']['benchmark']['records'] == 32
    assert config['validation']['benchmark']['warmup_repeats'] == 2
    assert config['validation']['benchmark']['measured_repeats'] == 10
    assert train['checkpoint_after_step'] == 1
    assert train['resume_check_steps'] == [2, 3]
    assert train['validation_checkpoint_physical_batch_size'] == 512
    assert config['resources']['training_gpus'] == 1
    assert config['resources']['validation_gpus'] == 2
    assert config['resources']['cpu_ceiling'] == 4
    assert config['resources']['memory_gb'] == 24


def probe_config():
    return json.loads((ROOT / 'configs/baselines/liger_capacity384_probe_v1.json').read_text())


def leaves(value, path=()):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from leaves(child, path + (key,))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from leaves(child, path + (index,))
    else:
        yield path, value


@pytest.mark.parametrize('path,value', list(leaves(probe_config())))
def test_every_probe_contract_leaf_is_enforced(path, value):
    config = probe_config()
    parent = config
    for key in path[:-1]:
        parent = parent[key]
    if isinstance(value, bool):
        changed = not value
    elif isinstance(value, (int, float)):
        changed = value + 1
    else:
        changed = value + '_changed'
    parent[path[-1]] = changed
    with pytest.raises(ValueError, match='config'):
        validate_probe_config(config)


@pytest.mark.parametrize('section', [None, 'training', 'validation', 'resources', 'model'])
@pytest.mark.parametrize('field', ['formal', 'resume_from', 'test', 'checkpoint'])
def test_unknown_authority_or_resume_fields_rejected(section, field):
    config = probe_config()
    target = config if section is None else config[section]
    target[field] = True
    with pytest.raises(ValueError, match='fields'):
        validate_probe_config(config)


@pytest.mark.parametrize('section,field,value', [
    (None, 'automatic_successor', 0), (None, 'seed', 42.0),
    ('resources', 'training_gpus', True), ('validation', 'test_access', 0),
    ('optimizer', 'clip_norm', 1), ('training', 'physical_batch_sizes', (512, 1024)),
])
def test_probe_contract_rejects_type_coercion(section, field, value):
    config = probe_config()
    (config if section is None else config[section])[field] = value
    with pytest.raises(ValueError, match='type'):
        validate_probe_config(config)


def test_probe_contract_rejects_missing_nested_field_and_returns_detached_copy():
    config = probe_config()
    checked = validate_probe_config(config)
    checked['training']['physical_batch_sizes'].append(2048)
    assert config['training']['physical_batch_sizes'] == [512, 1024]
    del config['validation']['benchmark']['records']
    with pytest.raises(ValueError, match='fields'):
        validate_probe_config(config)
