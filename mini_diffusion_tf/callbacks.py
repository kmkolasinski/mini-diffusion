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
        class_label: int = None,
    ):
        super(RenderImagesClearML, self).__init__()
        from clearml import Task

        if not isinstance(task, Task):
            raise ValueError("task must be an instance of clearml.Task")
        self.task = task
        self.model = model
        self.class_label = class_label
        self.every_n_epochs = every_n_epochs
        self.num_diffusion_steps = num_diffusion_steps

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, int]] = None):
        if epoch % self.every_n_epochs != 0:
            return

        class_labels = None
        if self.class_label is not None:
            class_labels = tf.constant([self.class_label] * 18)
        image = self.model.plot_images(
            epoch=epoch,
            logs=logs,
            num_rows=3,
            num_cols=6,
            num_diffusion_steps=self.num_diffusion_steps,
            class_labels=class_labels,
        )
        if self.class_label is None:
            self.task.logger.report_image(
                f"Generated images", "sample", iteration=epoch, image=image
            )
        else:
            self.task.logger.report_image(
                f"Generated images for class {self.class_label}",
                "sample",
                iteration=epoch,
                image=image,
            )


def get_default_callbacks(
    save_dir: Path, monitor: tuple[str, str] = ("loss", "min")
) -> list[tf.keras.callbacks.Callback]:
    save_dir = Path(save_dir)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(Path(save_dir) / "models"),
            monitor=monitor[0],
            mode=monitor[1],
            save_best_only=True,
            save_weights_only=True,
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
