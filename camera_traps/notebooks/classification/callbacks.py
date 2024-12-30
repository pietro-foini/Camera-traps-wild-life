import datetime
import time

import tensorflow as tf
from tensorflow.keras.callbacks import Callback


def scheduler(epoch, lr):
    """Learning scheduler."""
    if epoch <= 10:
        return lr
    else:
        return lr * tf.math.exp(-0.05)


class TimeStopping(Callback):
    """Stop training when a specified amount of time has passed."""

    def __init__(self, seconds: int = 86400, verbose: int = 0):
        super().__init__()

        self.seconds = seconds
        self.verbose = verbose
        self.stopped_epoch = None
        self.stopping_time = None

    def on_train_begin(self, logs=None):
        self.stopping_time = time.time() + self.seconds

    def on_epoch_end(self, epoch, logs=None):
        if time.time() >= self.stopping_time:
            self.model.stop_training = True
            self.stopped_epoch = epoch

    def on_train_end(self, logs=None):
        if self.stopped_epoch is not None and self.verbose > 0:
            formatted_time = datetime.timedelta(seconds=self.seconds)
            msg = "Timed stopping at epoch {} after training for {}".format(
                self.stopped_epoch + 1, formatted_time
            )
            print(msg)

    def get_config(self):
        config = {
            "seconds": self.seconds,
            "verbose": self.verbose,
        }

        base_config = super().get_config()
        return {**base_config, **config}
