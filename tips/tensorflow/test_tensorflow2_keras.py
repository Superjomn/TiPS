import tensorflow as tf
import numpy as np
import warnings
import math
import pytest

from distutils.version import LooseVersion

from tensorflow import keras
from tensorflow.python.keras.optimizer_v2 import optimizer_v2

import tips.tensorflow.keras as tips
from tips.tensorflow.ops import tips_basics
from tips.tensorflow.keras.callbacks import BroadcastGlobalVariablesCallback

_PRE_TF_2_4_0 = LooseVersion(tf.__version__) < LooseVersion("2.4.0")
_PRE_TF_2_2_0 = LooseVersion(tf.__version__) < LooseVersion("2.2.0")


@pytest.mark.skipif(
    LooseVersion(tf.__version__) < LooseVersion('2.0.0'),
    reason='TensorFlow v2 tests')
class Tf2KerasTests(tf.test.TestCase):
    def __init__(self, *args, **kwargs):
        super(Tf2KerasTests, self).__init__(*args, **kwargs)
        warnings.simplefilter("module")

    def test_train_model_lr_schedule(self):
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            0.001 * tips_basics.size(),
            decay_steps=100000,
            decay_rate=0.96,
            staircase=True)
        opt = tf.keras.optimizers.Adam(lr_schedule)
        opt = tips.DistributedOptimizer(opt)

        model = keras.models.Sequential()
        model.add(keras.layers.Dense(2, input_shape=(3, )))
        model.add(keras.layers.RepeatVector(3))
        model.add(keras.layers.ThresholdedReLU(0.5))
        model.compile(
            loss=keras.losses.mean_squared_error,
            optimizer=opt,
            metrics=[keras.metrics.categorical_accuracy],
            experimental_run_tf_function=False)

        x = np.random.random((1, 3))
        y = np.random.random((1, 3, 2))

        # No assertions, we just need to verify that it doesn't hang or error
        callbacks = [BroadcastGlobalVariablesCallback(0)]
        model.fit(x, y, steps_per_epoch=10, callbacks=callbacks, epochs=1)


if __name__ == '__main__':
    tf.test.main()
