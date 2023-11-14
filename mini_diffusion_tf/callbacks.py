from pathlib import Path
from typing import Dict, Optional

import tensorflow as tf


keras = tf.keras


class TensorBoard(keras.callbacks.TensorBoard):
    """
    Tensorboard Callback for logging all curves to tensorboard
    """

    def _collect_learning_rate(self, logs):
        """
        Collect learning rate from the logs produced by the model
        """
        lr_schedule = getattr(self.model.optimizer, "lr", None)

        if isinstance(lr_schedule, keras.optimizers.schedules.LearningRateSchedule):
            iterations = self.model.optimizer.iterations
            logs["learning_rate"] = lr_schedule(iterations)
        else:
            logs["learning_rate"] = lr_schedule

        return logs


class RenderImagesClearML(tf.keras.callbacks.Callback):
    def __init__(
        self,
        task,
        model,
        num_diffusion_steps: int = 25,
        every_n_epochs: int = 1,
    ):
        super(RenderImagesClearML, self).__init__()
        from clearml import Task

        if not isinstance(task, Task):
            raise ValueError("task must be an instance of clearml.Task")
        self.task = task
        self.model = model
        self.every_n_epochs = every_n_epochs
        self.num_diffusion_steps = num_diffusion_steps

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, int]] = None):
        if epoch % self.every_n_epochs != 0:
            return
        image = self.model.plot_images(
            epoch=epoch, logs=logs, num_diffusion_steps=self.num_diffusion_steps
        )
        self.task.logger.report_image(
            f"Generated images", "sample", iteration=epoch, image=image
        )


class ModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    def __init__(self, model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_model = model

    def _save_model(self, epoch, batch, logs):
        self.set_model(self._saved_model)
        return super()._save_model(epoch, batch, logs)


def get_default_callbacks(
    model, save_dir: Path, monitor: tuple[str, str] = ("loss", "min")
) -> list[tf.keras.callbacks.Callback]:
    save_dir = Path(save_dir)
    callbacks = [
        ModelCheckpoint(
            model=model,
            filepath=str(Path(save_dir) / "models"),
            monitor=monitor[0],
            mode=monitor[1],
            save_best_only=True,
            # save_weights_only=True,
        ),
        TensorBoard(
            log_dir=str(Path(save_dir) / "logs"),
            histogram_freq=0,
            embeddings_freq=0,
            update_freq="epoch",
            write_steps_per_second=True,
            write_graph=False,
        ),
    ]
    return callbacks
