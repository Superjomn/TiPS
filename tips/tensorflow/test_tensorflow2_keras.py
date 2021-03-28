import tensorflow as tf
import numpy as np
import warnings

from distutils.version import LooseVersion

from tensorflow import keras
from tensorflow.python.keras.optimizer_v2 import optimizer_v2

import tips.tensorflow.keras as tips
from tips.tensorflow.ops import tips_basics

_PRE_TF_2_4_0 = LooseVersion(tf.__version__) < LooseVersion("2.4.0")
_PRE_TF_2_2_0 = LooseVersion(tf.__version__) < LooseVersion("2.2.0")


class Tf2KerasTests(tf.test.TestCase):
    def __init__(self, *args, **kwargs):
        super(Tf2KerasTests, self).__init__(*args, **kwargs)
        warnings.simplefilter("module")

    def test_gradient_aggregation(self):
        class TestingOptimizer(optimizer_v2.OptimizerV2):
            def get_config(self):
                config = super(TestingOptimizer, self).get_config()
                return config

            def _create_slots(self, var_list):
                # Only needed for TF < 2.2.
                pass

            def _resource_apply_dense(self, grad, handle, apply_state=None):
                return handle.assign_add(grad)

        backward_passes_per_step = 4
        tips_optimizer = tips.DistributedOptimizer(
            optimizer=TestingOptimizer("test"),
            backward_passes_per_step=backward_passes_per_step,
            average_aggregated_gradients=True,
        )

        _ = tips_optimizer.iterations

        @tf.function
        def apply_gradients_in_tf_function(gradient_updates, model_variables,
                                           **kwargs):
            tips_optimizer.apply_gradients(
                zip(gradient_updates, model_variables), **kwargs)

        gradients = [tf.constant([float(tips_basics.rank())])]
        variables = [tf.Variable([0.0])]

        for idx in range(10):
            #updated_gradients = tips_optimizer._aggregate_gradients(
            #zip(gradients, variables))
            apply_gradients_in_tf_function(
                gradients, variables, experimental_aggregate_gradients=False)

        updated_variable_value = variables[0][0].numpy()


if __name__ == '__main__':
    tf.test.main()
