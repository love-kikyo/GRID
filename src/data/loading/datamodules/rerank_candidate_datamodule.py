from typing import Optional

from lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset
from torch.utils.data.distributed import DistributedSampler

from src.data.loading.components.interfaces import RerankCandidateDataloaderConfig


class RerankCandidateDataModule(LightningDataModule):
    def __init__(
        self,
        train_dataloader_config: Optional[RerankCandidateDataloaderConfig] = None,
        val_dataloader_config: Optional[RerankCandidateDataloaderConfig] = None,
        test_dataloader_config: Optional[RerankCandidateDataloaderConfig] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.train_dataloader_config = train_dataloader_config
        self.val_dataloader_config = val_dataloader_config
        self.test_dataloader_config = test_dataloader_config

    def _build_dataloader(
        self, dataloader_config: RerankCandidateDataloaderConfig, drop_last: bool
    ) -> DataLoader:
        dataset = dataloader_config.dataset_class(
            dataset_config=dataloader_config.dataset_config
        )
        if isinstance(dataset, IterableDataset) and hasattr(self, "trainer") and self.trainer is not None:
            dataset.set_distributed_params(
                total_workers=self.trainer.world_size,
                global_worker_id=self.trainer.global_rank,
            )

        sampler = None
        shuffle = dataloader_config.shuffle
        if (
            not isinstance(dataset, IterableDataset)
            and
            dataloader_config.use_distributed_sampler
            and hasattr(self, "trainer")
            and self.trainer is not None
            and self.trainer.world_size > 1
        ):
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.trainer.world_size,
                rank=self.trainer.global_rank,
                shuffle=shuffle,
                drop_last=drop_last,
            )
            shuffle = False

        return DataLoader(
            dataset=dataset,
            batch_size=dataloader_config.batch_size_per_device,
            num_workers=dataloader_config.num_workers,
            pin_memory=dataloader_config.pin_memory,
            persistent_workers=(
                dataloader_config.persistent_workers
                if dataloader_config.num_workers > 0
                else False
            ),
            drop_last=drop_last,
            timeout=dataloader_config.timeout,
            collate_fn=dataloader_config.collate_fn,
            sampler=sampler,
            shuffle=shuffle if sampler is None else False,
        )

    def train_dataloader(self) -> DataLoader:
        return self._build_dataloader(
            self.train_dataloader_config,
            drop_last=self.train_dataloader_config.drop_last,
        )

    def val_dataloader(self) -> DataLoader:
        return self._build_dataloader(
            self.val_dataloader_config,
            drop_last=self.val_dataloader_config.drop_last,
        )

    def test_dataloader(self) -> DataLoader:
        return self._build_dataloader(
            self.test_dataloader_config,
            drop_last=self.test_dataloader_config.drop_last,
        )
