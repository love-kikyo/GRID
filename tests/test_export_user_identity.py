from functools import partial

import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from scripts.export_paired_evaluation import (
    collate_with_user_ids,
    configure_dataloader,
    extract_user_ids,
)
from src.data.loading.components.collate_functions import collate_fn_train


@pytest.mark.parametrize("per_row", [True, False])
def test_export_ids_do_not_change_model_input(per_row):
    rows = [
        {"user_id": torch.tensor([uid]), "sequence_data": torch.tensor(sid),
         "hist_itemkey": torch.tensor(keys)}
        for uid, sid, keys in [(0, [1, 2, 3, 4], [12, 34]),
                               (2**40, [5, 6], [56])]
    ]
    bare = [{k: v for k, v in row.items() if k != "user_id"} for row in rows]
    collator = partial(collate_fn_train, sequence_length=6, hist_itemkey_length=3)
    expected = collator(bare)
    batch = rows if per_row else {k: [row[k] for row in rows] for k in rows[0]}
    actual = collate_with_user_ids(batch, collator, "user_id")
    assert extract_user_ids(actual).tolist() == [0, 2**40]
    assert set(actual.transformed_sequences) == set(expected.transformed_sequences)
    torch.testing.assert_close(actual.mask, expected.mask)
    for key in expected.transformed_sequences:
        torch.testing.assert_close(actual.transformed_sequences[key], expected.transformed_sequences[key])
    assert all("user_id" in row for row in rows)  # no mutation of caller data


def test_export_collator_rejects_missing_and_noninteger_ids():
    with pytest.raises(KeyError):
        collate_with_user_ids([{"sequence_data": torch.tensor([1])}], None, "user_id")
    with pytest.raises(ValueError, match="scalar integers"):
        collate_with_user_ids([{"user_id": torch.tensor([1.5])}], None, "user_id")


@pytest.mark.parametrize("split", ["val", "test"])
def test_export_config_preserves_training_settings_and_wraps_once(split):
    # Use a self-contained minimal config; no dependency on local run artifacts.
    loader = {
        "batch_size_per_device": 32, "num_workers": 2, "persistent_workers": True,
        "drop_last": False, "should_shuffle_rows": False, "assign_all_files_per_worker": False,
        "dataset_config": {"keep_user_id": False, "user_id_field": "user_id"},
        "collate_fn": {"_target_": "src.data.loading.components.collate_functions.collate_fn_train",
                       "_partial_": True, "sequence_length": 6, "hist_itemkey_length": 3},
    }
    cfg = OmegaConf.create({"data_loading": {
        f"{name}_dataloader_config": {"dataloader": loader} for name in ("train", "val", "test")
    }})
    original = OmegaConf.to_container(cfg, resolve=True)
    configure_dataloader(cfg, split, batch_size=2, num_workers=0)
    configure_dataloader(cfg, split, batch_size=2, num_workers=0)
    selected = cfg.data_loading[f"{split}_dataloader_config"].dataloader
    assert selected.dataset_config.keep_user_id is True
    assert selected.persistent_workers is False
    assert selected.collate_fn.sequence_collate._target_.endswith("collate_fn_train")
    collator = instantiate(selected.collate_fn)
    result = collator([{"user_id": torch.tensor([42]), "sequence_data": torch.tensor([1, 2]),
                        "hist_itemkey": torch.tensor([12])}],
                      labels={}, sequence_length=8, masking_token=-1, padding_token=0, oov_token=None)
    assert extract_user_ids(result).tolist() == [42]
    assert result.transformed_sequences["sequence_data"].shape == (1, 8)
    for name in {"train", "val", "test"} - {split}:
        key = f"{name}_dataloader_config"
        assert OmegaConf.to_container(cfg.data_loading[key], resolve=True) == original["data_loading"][key]
