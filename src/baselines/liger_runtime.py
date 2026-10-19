"""Compute-node-only context: Torch first, CPU-only TF, no extra workers."""
import itertools
import json
import os
from pathlib import Path
import random

import numpy as np
import torch

from scripts.probe_liger_cpu4 import sha
from scripts.liger_training_manifest import checkpoint_binding
from src.baselines.liger_data import epoch_batches, record_stream
from src.baselines.liger_state import BudgetSchedule
from src.runtime_cpu import THREAD_VARIABLES, configure_torch
from src.runtime_export_cpu import check_export_allocation, export_cpu_snapshot


class Runtime:
    def __init__(self, manifest, *, build_model=True):
        self.m = manifest
        self.config = json.loads(Path(manifest['config']).read_text())
        assert all(os.environ.get(key) == '1' for key in THREAD_VARIABLES)
        configure_torch(torch)
        self.resources = check_export_allocation()
        self.cpu_checks = [export_cpu_snapshot(self.resources['allowed_cpus'])]
        self.binding = checkpoint_binding(manifest)
        self.gpu = None
        self.seen = np.load(manifest['catalog']['path'], mmap_mode='r', allow_pickle=False)
        from src.baselines.liger_candidates import SidRowIndex
        self.index = SidRowIndex(*(np.load(manifest['index_artifacts'][name]['path'], mmap_mode='r', allow_pickle=False)
                                  for name in ('sid_keys', 'asset_rows')))
        if not build_model: return
        assert torch.cuda.device_count() == 1 and '4090' in torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        self.gpu = dict(name=props.name, uuid=str(props.uuid), total_bytes=props.total_memory)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        import tensorflow as tf
        tf.config.set_visible_devices([], 'GPU')
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        self.tf = tf
        self.versions = dict(torch=torch.__version__, tensorflow=tf.__version__, cuda=torch.version.cuda, numpy=np.__version__)
        print('[liger-train] verify frozen assets', flush=True)
        for asset in manifest['assets'].values(): assert sha(asset['path']) == asset['sha256']
        self.sids = np.load(manifest['assets']['frozen_sid']['path'], mmap_mode='r', allow_pickle=False)
        self.content = np.load(manifest['assets']['model_embedding_copy']['path'], mmap_mode='r', allow_pickle=False)
        from scripts.probe_liger_batch_scaling import new_model
        from src.baselines.liger_full_catalog import RowCatalog
        from src.baselines.liger_retrieval import cold_rows_from_seen
        random.seed(manifest['seed'])
        np.random.seed(manifest['seed'])
        self.model, self.method, self.official, self.initial_sha = new_model(manifest, self.content.shape[1])
        self.catalog = RowCatalog(self.content, self.seen)
        self.cold = cold_rows_from_seen(len(self.content), self.seen)
        assert len(self.seen) == 11542755 and len(self.cold) == 23915712
        opt = self.config['optimizer']
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=opt['learning_rate'], weight_decay=opt['weight_decay'])
        t = self.config['training']
        self.scheduler = BudgetSchedule(self.optimizer, total=t['max_optimizer_steps'], warmup=t['warmup_steps'], base_lr=opt['learning_rate'])
        self.scaler = torch.amp.GradScaler('cuda')

    @property
    def components(self):
        return self.model, self.optimizer, self.scheduler, self.scaler

    def batches(self, *, training, epoch=0, start_batch=0, validation_limit=None):
        if training and validation_limit is not None: raise ValueError('Cannot cap train through validation argument')
        entries = self.m['train_files' if training else 'validation_files']
        def stream(files):
            records = record_stream(self.tf, files, self.m['asset_rows'])
            try:
                yield from records if validation_limit is None else itertools.islice(records, validation_limit)
            finally:
                records.close()
        return epoch_batches(entries, stream, batch_size=self.config['training' if training else 'validation']['batch_size'],
            seed=self.m['seed'], epoch=epoch, training=training,
            shuffle_buffer=self.config['training']['shuffle_buffer'], start_batch=start_batch)

    def cpu_check(self):
        value = export_cpu_snapshot(self.resources['allowed_cpus'])
        self.cpu_checks.append(value)
        return value
