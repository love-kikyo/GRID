from typing import Dict, Optional, Tuple
import time

import torch
import transformers
from torchmetrics.aggregation import BaseAggregator

from src.components.eval_metrics import RetrievalEvaluator
from src.data.loading.components.interfaces import (
    SequentialModelInputData,
    SequentialModuleLabelData,
)
from src.models.components.interfaces import SharedKeyAcrossPredictionsOutput
from src.models.components.network_blocks.embedding_aggregator import (
    EmbeddingAggregator,
)
from src.models.modules.base_module import BaseModule
from enum import Enum, auto


class ModelMode(Enum):
    TRAIN = auto()
    INFER = auto()


class TransformerBaseModule(BaseModule):
    def __init__(
        self,
        huggingface_model: transformers.PreTrainedModel,
        postprocessor: torch.nn.Module,
        aggregator: EmbeddingAggregator,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        loss_function: torch.nn.Module,
        evaluator: RetrievalEvaluator,
        weight_tying: bool,
        compile: bool,
        training_loop_function: callable = None,
        feature_to_model_input_map: Dict[str, str] = {},
        decoder: torch.nn.Module = None,
    ) -> None:

        super().__init__(
            model=huggingface_model,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_function=loss_function,
            evaluator=evaluator,
            training_loop_function=training_loop_function,
        )

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        # we remove the nn.Modules as they are already checkpointed to avoid doing it twice

        self.save_hyperparameters(
            logger=False,
            ignore=[
                "huggingface_model",
                "postprocessor",
                "aggregator",
                "decoder",
                "loss_function",
            ],
        )

        self.encoder = huggingface_model
        self.embedding_post_processor = postprocessor
        self.decoder = decoder
        self.aggregator = aggregator
        self.feature_to_model_input_map = feature_to_model_input_map

    def get_embedding_table(self):
        if self.hparams.weight_tying:  # type: ignore
            return self.encoder.get_input_embeddings().weight
        else:
            return self.decoder.weight

    def training_step(
        self,
        batch: Tuple[SequentialModelInputData],
        batch_idx: int,
    ) -> torch.Tensor:
        pass

    def eval_step(
        self,
        batch: Tuple[SequentialModelInputData, SequentialModuleLabelData],
        loss_to_aggregate: BaseAggregator,
    ):
        pass

    def predict_step(
        self,
        batch: Tuple[SequentialModelInputData, SequentialModuleLabelData],
        batch_idx: int,
    ):
        pass
