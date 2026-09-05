#!/usr/bin/env python3
"""Read one batch per split through the real GRID preprocessing pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from lightning.pytorch.trainer.states import TrainerFn
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if not OmegaConf.has_resolver("sub"):
    OmegaConf.register_new_resolver("sub", lambda x, y: int(x) - int(y))

# Import for resolver registration side effects.
from src.utils import custom_hydra_resolvers as _custom_hydra_resolvers  # noqa: E402,F401


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--semantic-id-path", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Workers per split; use the training value to expose empty-shard workers.",
    )
    return parser.parse_args()


def _unwrap(batch: Any) -> Any:
    return batch[0] if isinstance(batch, tuple) else batch


def _check_batch(batch: Any, split: str, num_hierarchies: int = 4) -> dict[str, Any]:
    batch = _unwrap(batch)
    sid = batch.transformed_sequences["sequence_data"].long()
    hist_itemkey = batch.transformed_sequences["hist_itemkey"].long()
    mask = batch.mask.long()

    assert sid.ndim == 2 and sid.size(1) == 32, sid.shape
    assert mask.shape == sid.shape
    assert hist_itemkey.ndim == 2 and hist_itemkey.size(1) >= 100
    assert torch.all(mask[:, -num_hierarchies:] == 1)
    valid_tokens = mask.sum(dim=1)
    assert torch.all(valid_tokens >= 2 * num_hierarchies)
    assert torch.all(valid_tokens.remainder(num_hierarchies) == 0)
    valid_sid = sid[mask.bool()]
    assert int(valid_sid.min()) >= 1
    assert int(valid_sid.max()) <= 4096

    powers = 4096 ** torch.arange(
        num_hierarchies - 1, -1, -1, dtype=torch.int64
    )
    target_itemkey = (sid[:, -num_hierarchies:] * powers).sum(dim=-1)
    torch.testing.assert_close(target_itemkey, hist_itemkey[:, -1])

    return {
        "split": split,
        "batch_size": sid.size(0),
        "sid_shape": list(sid.shape),
        "hist_itemkey_shape": list(hist_itemkey.shape),
        "minimum_valid_items": int(valid_tokens.min()) // num_hierarchies,
        "maximum_valid_items": int(valid_tokens.max()) // num_hierarchies,
    }


def main() -> None:
    args = _parse_args()
    data_root = args.data_root.resolve()
    semantic_id_path = args.semantic_id_path.resolve()
    overrides = [
        "experiment=[tiger_train_flat,tiger_train_sid]",
        f"data_dir={data_root}",
        f"semantic_id_path={semantic_id_path}",
        "num_hierarchies=4",
        "num_embeddings_per_hierarchy=4096",
    ]
    for stage in ("train", "val", "test"):
        prefix = f"data_loading.{stage}_dataloader_config.dataloader"
        overrides.extend(
            [
                f"{prefix}.batch_size_per_device={args.batch_size}",
                f"{prefix}.num_workers={args.num_workers}",
                f"{prefix}.persistent_workers=false",
            ]
        )
    overrides.extend(
        [
            "data_loading.train_dataloader_config.dataloader.should_shuffle_rows=false",
            "data_loading.train_dataloader_config.dataloader.assign_all_files_per_worker=false",
        ]
    )

    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
        cfg = compose(config_name="train", overrides=overrides)
    datamodule = instantiate(cfg.data_loading.datamodule)
    datamodule.trainer = SimpleNamespace(world_size=1, global_rank=0)
    datamodule.setup()

    loaders = {
        "train": datamodule.get_dataloader(TrainerFn.FITTING),
        "evaluation": datamodule.get_dataloader(TrainerFn.VALIDATING),
        "test": datamodule.get_dataloader(TrainerFn.TESTING),
    }
    reports = []
    for split, wrapped_loader in loaders.items():
        loader = _unwrap(wrapped_loader)
        reports.append(_check_batch(next(iter(loader)), split))

    print({"status": "passed", "batches": reports})


if __name__ == "__main__":
    main()
