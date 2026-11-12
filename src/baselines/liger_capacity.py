"""Fresh configurable LIGER construction for isolated capacity diagnostics.

No checkpoint loading, device allocation policy, data access or training is
hidden in this factory. Historical ``new_model`` and its hashes stay unchanged.
The content adapter, tied SID vocabulary, dropout and initialization are the
same as that factory; only the explicit Transformer architecture can differ.
"""
from collections import Counter
from copy import deepcopy
from numbers import Integral
from pathlib import Path


ARCHITECTURE_KEYS = frozenset({
    'd_model', 'd_ff', 'num_layers', 'num_decoder_layers', 'num_heads', 'd_kv'})
CAPACITY384 = dict(d_model=384, d_ff=2560, num_layers=6,
                   num_decoder_layers=6, num_heads=6, d_kv=64)
CAPACITY384_PARAMETERS = 41868160
CAPACITY384_BACKBONE_PARAMETERS = 34222464


def validate_probe_config(config):
    """Reject scope/budget drift, including unknown keys and bool/int coercion.

    This is deliberately a single authorized diagnostic configuration, not a
    general training configuration. A changed budget requires a new contract.
    """
    expected = dict(
        protocol='liger-fixed-sid-scl-prefix-sampled-capacity384-probe-v1',
        phase='capacity384_resource_and_correctness_only', seed=42,
        formal_training_ready=False, automatic_successor=False,
        initialization='fresh seed42 random weights; no old formal or diagnostic checkpoint initialization',
        model=CAPACITY384,
        expected_parameters=dict(total=41868160, trainable=41868160,
                                 transformer_backbone=34222464),
        optimizer=dict(name='AdamW', learning_rate=.0003, weight_decay=.035, clip_norm=1.0),
        dense_sampling=dict(num_negatives=32768, seed=42),
        training=dict(
            logical_batch_size=2048, physical_batch_sizes=[512, 1024], history_items=7,
            shuffle_buffer=8192, prefix_shuffle_buffer=8192,
            supervision='all_nonempty_history_prefixes_in_last8',
            loss_weighting='equal_prefix_examples_both_sid_and_dense',
            sampling_scope='one forced-positive union and one 32K complement draw per logical batch',
            projection_dropout_scope='one candidate projection draw shared by all microbatches in a logical batch',
            update_scope='prefix-weighted micro losses; one clip, scaler, optimizer and scheduler step per logical batch',
            warmup_attempts=3, measured_attempts=10, fixture_prefixes=26624,
            max_attempts_per_case=13, schedule_max_optimizer_steps=22460,
            schedule_warmup_steps=1123,
            schedule_scope='borrowed one-pass schedule for resource probes only; formal training is not authorized',
            drop_last=False,
            amp='FP16 input/SID path with GradScaler; dense loss FP32, TF32 disabled',
            cases_share_fixture_and_initial_weights=True, resume_check=True,
            resume_check_steps=[2, 3], checkpoint_after_step=1,
            validation_checkpoint_physical_batch_size=512),
        feature_cache=dict(backend='cached_seen', storage='process_local_cpu_ram',
                           verify_all_rows=True, cache_trainable_projection=False),
        standardization=dict(fit_scope='frozen_train_seen', reuse_verified_statistics=True),
        validation=dict(
            split='evaluation', records=65, batch_size=32, beam_size=20, cold_rows=23915712,
            catalog_partition='two balanced contiguous sorted-row shards, one per GPU',
            candidate_mode='full_cold', scoring_dtype='float32',
            tie_break='ascending global asset row', chunk_size=65536, test_access=False,
            purpose='complete catalog and state-preserving integration only, not recommendation quality',
            benchmark=dict(records=32, warmup_repeats=2, measured_repeats=10,
                replay='repeat the first 32 of the 65 validation records with fixed model weights',
                timing_scope='generation plus complete cold ranking; input packing, projection cache preparation and unique-record audit reported separately',
                interpretation='repeated-input latency probe, not independent samples or additional validation metrics')),
        process_environment={'NUMPY_MADVISE_HUGEPAGE': '0'}, deadline_seconds=2400,
        resources=dict(training_gpus=1, validation_gpus=2, gpu_model='RTX 4090',
                       cpu_ceiling=4, library_threads=1, memory_gb=24, job_time_limit_minutes=45))

    def check(actual, wanted, path):
        if type(actual) is not type(wanted):
            raise ValueError(f'{path}: configuration type differs from authorized probe')
        if isinstance(wanted, dict):
            if set(actual) != set(wanted):
                raise ValueError(f'{path}: missing or unauthorized configuration fields')
            for key in wanted:
                check(actual[key], wanted[key], f'{path}.{key}')
        elif isinstance(wanted, list):
            if len(actual) != len(wanted):
                raise ValueError(f'{path}: configuration length changed')
            for index, (value, target) in enumerate(zip(actual, wanted)):
                check(value, target, f'{path}[{index}]')
        elif actual != wanted:
            raise ValueError(f'{path}: configuration differs from authorized probe')

    check(config, expected, 'config')
    return deepcopy(config)


def validate_architecture(architecture):
    if not isinstance(architecture, dict) or set(architecture) != ARCHITECTURE_KEYS:
        raise ValueError('Expected exactly the six explicit Transformer dimensions')
    if any(not isinstance(v, Integral) or isinstance(v, bool) or v < 1
           for v in architecture.values()):
        raise ValueError('Transformer dimensions must be positive integers')
    # T5 uses heads*d_kv for attention projections; d_model need not be divisible
    # by num_heads (the accepted 128-wide baseline uses six heads).
    return {key: int(value) for key, value in architecture.items()}


def parameter_report(model):
    """Count unique parameters once, including T5's shared input/output weight."""
    groups = Counter()
    for name, parameter in model.named_parameters():
        groups[name.split('.')[0]] += parameter.numel()
    tied = (model.shared.weight is model.encoder.embed_tokens.weight
            is model.decoder.embed_tokens.weight is model.lm_head.weight)
    if not tied:
        raise ValueError('Capacity comparison requires the accepted tied SID vocabulary')
    return dict(total_params=sum(p.numel() for p in model.parameters()),
                trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
                backbone_params=groups['encoder'] + groups['decoder'],
                parameter_tensors=len(list(model.parameters())),
                disjoint_groups=dict(groups), tied_sid_input_output=tied)


def build_capacity_model(upstream_root, architecture, *, seed=42, content_dim=128,
                         device='cpu'):
    """Return fresh ``(model, method, official_evaluation, parameter_report)``.

    CPU construction followed by transfer preserves the accepted initialization
    policy for CPU/CUDA. ``meta`` is only for allocation-free parameter audits;
    meta tensors must never be mistaken for an initialized training model.
    Callers bind source/config hashes, enforce CPU/GPU budgets, and choose mode.
    """
    import torch
    from omegaconf import OmegaConf
    from transformers import T5Config
    from scripts.probe_liger_cpu4 import import_official
    from src.baselines.liger_projection import use_linear_residuals

    architecture = validate_architecture(architecture)
    if (not isinstance(seed, Integral) or isinstance(seed, bool) or not 0 <= seed < 2**63
            or not isinstance(content_dim, Integral) or isinstance(content_dim, bool)
            or content_dim < 1):
        raise ValueError('Expected nonnegative int64 seed and positive content dimension')
    destination = torch.device(device)
    if destination.type not in {'cpu', 'cuda', 'meta'}:
        raise ValueError('Capacity diagnostics support CPU, CUDA or meta only')
    upstream = Path(upstream_root)
    official = import_official('_grid_liger_capacity_model', upstream / 'src/tiger.py')
    evaluation = import_official('_grid_liger_capacity_eval', upstream / 'src/evaluation.py')
    method = OmegaConf.to_container(OmegaConf.merge(
        OmegaConf.load(upstream / 'configs/method/base.yaml'),
        OmegaConf.load(upstream / 'configs/method/setting.yaml')), resolve=True)
    expected = dict(use_id='sid', include_user_id=False,
                    flag_add_input_embedding=True, flag_use_output_embedding=True,
                    embedding_loss_weight=1, sid_loss_weight=1)
    if any(method.get(key) != value for key, value in expected.items()):
        raise ValueError('Upstream method no longer matches accepted LIGER semantics')
    head = method['embedding_head_dict']
    expected_head = dict(use_new_init=False, embed_proj_type='mlp',
                         hidden_sizes=[256, 512, 768], embd_proj_dropout_rate=.5,
                         embd_proj_in_dropout_rate=.2, normalize_logits=True,
                         logits_temperature=.07)
    if any(head.get(key) != value for key, value in expected_head.items()):
        raise ValueError('Upstream content projection or scoring semantics changed')
    head['text_embedding_dim'] = int(content_dim)
    config = T5Config(**architecture, dropout_rate=.2, vocab_size=4 * 4096 + 2,
        pad_token_id=0, eos_token_id=4 * 4096 + 1, decoder_start_token_id=0,
        feed_forward_proj='relu', layer_norm_epsilon=1e-8, initializer_factor=.02,
        tie_word_embeddings=True)
    torch.manual_seed(int(seed))
    construction_device = 'meta' if destination.type == 'meta' else 'cpu'
    with torch.device(construction_device):
        model = official.TIGER(config, n_semantic_codebook=3, max_items_per_seq=7,
            flag_use_output_embedding=True, flag_use_learnable_text_embed=True,
            embedding_head_dict=head)
        if use_linear_residuals(model.emb_proj) != 3:
            raise ValueError('Expected all three original residual projection blocks')
    model = model.to(destination).train()
    report = parameter_report(model)
    if architecture == CAPACITY384 and content_dim == 128:
        if (report['total_params'] != CAPACITY384_PARAMETERS
                or report['backbone_params'] != CAPACITY384_BACKBONE_PARAMETERS):
            raise ValueError('384-wide parameter identity changed')
    return model, method, evaluation, report
