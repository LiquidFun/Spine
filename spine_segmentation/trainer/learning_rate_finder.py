import mlflow
from lightning.pytorch.callbacks import LearningRateFinder

from spine_segmentation.utils.log_dir import get_next_log_dir


class FineTuneLearningRateFinder(LearningRateFinder):
    def __init__(self, milestones, *args, update_attr=True, **kwargs):
        super().__init__(*args, update_attr=update_attr, early_stop_threshold=None, num_training_steps=300, **kwargs)
        self.milestones = milestones

    def on_fit_start(self, *args, **kwargs):
        return

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch in self.milestones or trainer.current_epoch == 0:
            lr_before = trainer.optimizers[0].param_groups[0]["lr"]
            self.lr_find(trainer, pl_module)
            trainer.optimizers[0].param_groups[0]["lr"] = self.optimal_lr.suggestion()
            fig = self.optimal_lr.plot(suggest=True)
            fig_path = get_next_log_dir() / f"lr_finder{trainer.current_epoch:03}.png"
            fig.savefig(fig_path)
            mlflow.log_artifact(fig_path)
            lr_after = trainer.optimizers[0].param_groups[0]["lr"]
            print(f"Optimizer learning rate before {lr_before}, and after: {lr_after}")
