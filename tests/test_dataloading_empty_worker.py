from types import SimpleNamespace

from src.data.loading.components import dataloading


class _DummyIterator:
    def __init__(self):
        self.list_of_file_paths = None
        self.should_shuffle_rows = None
        self.iter_batches_called = False

    def update_list_of_file_paths(self, paths):
        self.list_of_file_paths = paths

    def shuffle(self, seed=42):
        return self

    def iterrows(self):
        raise AssertionError("an empty worker must not initialize its iterator")

    def iter_batches(self, batch_size):
        self.iter_batches_called = True
        raise AssertionError("an empty worker must not initialize its iterator")


def test_worker_without_assigned_files_finishes_cleanly(monkeypatch):
    data_iterator = _DummyIterator()
    config = SimpleNamespace(
        data_iterator=data_iterator,
        iterate_per_row=False,
        preprocessing_functions=[],
    )
    dataset = dataloading.UnboundedSequenceIterable(
        dataset_config=config,
        data_folder="unused",
        batch_size=32,
        is_for_training=True,
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
    assert data_iterator.list_of_file_paths == []
    assert data_iterator.iter_batches_called is False
