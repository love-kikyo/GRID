from typing import Optional

from lightning import LightningDataModule
from torch.utils.data import DataLoader

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
