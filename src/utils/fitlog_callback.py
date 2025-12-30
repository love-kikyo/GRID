import fitlog
from lightning.pytorch.callbacks import Callback


class FitlogCallback(Callback):
    def __init__(self):
        self._logged_this_val = False
        self._last_train_step = None

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_rank != 0:
            return

        step = trainer.global_step
        if step == self._last_train_step:
            return

        self._last_train_step = step
        metrics = trainer.callback_metrics
        for key in [
            "train/loss",
            "train/sid_loss",
            "train/item_loss",
            "train/pairwise_loss",
            "train/auc", 
            "train/margin", 
            "train/hard_ratio"
        ]:
            if key in metrics:
                fitlog.add_loss(
                    metrics[key].item(),
                    name=key,
                    step=step,
                )

    def on_validation_epoch_start(self, trainer, pl_module):
        # reset metrics
        for m in pl_module.evaluator.metrics.values():
            m.reset()
        # reset flag
        self._logged_this_val = False

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.global_rank != 0:
            return

        if self._logged_this_val:
            return
        self._logged_this_val = True

        for name, metric in pl_module.evaluator.metrics.items():
            value = metric.compute().mean().item()
            fitlog.add_metric(
                value,
                name=f"val/{name}",
                step=trainer.global_step,
            )

    def on_fit_end(self, trainer, pl_module):
        if trainer.global_rank == 0:
            fitlog.finish()
