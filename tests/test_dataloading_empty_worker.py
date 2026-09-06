from types import SimpleNamespace

import pytest

from src.data.loading.components import dataloading


class _DummyIterator:
    def __init__(self):
        self.list_of_file_paths = None
        self.should_shuffle_rows = None
        self.iterrows_called = False
        self.iter_batches_called = False

    def update_list_of_file_paths(self, paths):
        self.list_of_file_paths = paths

    def shuffle(self, seed=42):
        raise AssertionError("an empty worker must not shuffle its iterator")

    def iterrows(self):
        self.iterrows_called = True
        raise AssertionError("an empty worker must not initialize its iterator")

    def iter_batches(self, batch_size):
        self.iter_batches_called = True
        raise AssertionError("an empty worker must not initialize its iterator")


@pytest.mark.parametrize("iterate_per_row", [False, True])
@pytest.mark.parametrize("is_for_training", [False, True])
@pytest.mark.parametrize("should_shuffle_rows", [False, True])
def test_worker_without_assigned_files_finishes_cleanly(
    monkeypatch, iterate_per_row, is_for_training, should_shuffle_rows
):
    data_iterator = _DummyIterator()
    config = SimpleNamespace(
        data_iterator=data_iterator,
        iterate_per_row=iterate_per_row,
        preprocessing_functions=[],
    )
    dataset = dataloading.UnboundedSequenceIterable(
        dataset_config=config,
        data_folder="unused",
        batch_size=32,
        is_for_training=is_for_training,
        should_shuffle_rows=should_shuffle_rows,
    )
    dataset.set_list_of_files(["only-shard.tfrecord.gz"])
    dataset.set_distributed_params(total_workers=1, global_worker_id=0)

    # Worker 1 receives files[1::2], hence an empty list.
    monkeypatch.setattr(
        dataloading,
        "get_worker_info",
        lambda: SimpleNamespace(id=1, num_workers=2),
    )

    assert list(dataset) == []
    # Reusing the worker (e.g. persistent validation workers) must stay safe.
    assert list(dataset) == []
    assert dataset.dataset_to_iterate is None
    assert data_iterator.list_of_file_paths == []
    assert data_iterator.iterrows_called is False
    assert data_iterator.iter_batches_called is False


class _NonemptyIterator(_DummyIterator):
    def iterrows(self):
        self.iterrows_called = True
        return iter({"file": path} for path in self.list_of_file_paths)

    def iter_batches(self, batch_size):
        self.iter_batches_called = True
        return iter({"file": path} for path in self.list_of_file_paths)


@pytest.mark.parametrize("iterate_per_row", [False, True])
def test_nonempty_worker_preserves_assignment_and_repeated_evaluation(
    monkeypatch, iterate_per_row
):
    data_iterator = _NonemptyIterator()
    dataset = dataloading.UnboundedSequenceIterable(
        dataset_config=SimpleNamespace(
            data_iterator=data_iterator,
            iterate_per_row=iterate_per_row,
            preprocessing_functions=[],
        ),
        data_folder="unused",
        batch_size=32,
        is_for_training=False,
    )
    dataset.set_list_of_files(["shard-0", "shard-1", "shard-2"])
    dataset.set_distributed_params(total_workers=1, global_worker_id=0)
    monkeypatch.setattr(
        dataloading, "get_worker_info", lambda: SimpleNamespace(id=0, num_workers=2)
    )

    expected = [{"file": "shard-0"}, {"file": "shard-2"}]
    assert list(dataset) == expected
    assert list(dataset) == expected
    assert data_iterator.iterrows_called is iterate_per_row
    assert data_iterator.iter_batches_called is not iterate_per_row
