import tips.tensorflow as tips
import tensorflow as tf
from tips.tensorflow.gradient_aggregation import LocalGradientAggregationHelper
from tips.tensorflow.gradient_aggregation_eager import LocalGradientAggregationHelperEager
from tips.tensorflow.ops import rank_op
from tensorflow import keras
from distutils.version import LooseVersion

_PRE_TF_2_4_0 = LooseVersion(tf.__version__) < LooseVersion('2.4.0')


def DistributedOptimizer(optimizer,
                         name=None,
                         device_dense='',
                         device_sparse='',
                         backward_passes_per_step=1,
                         average_aggregated_gradients=False,
                         sparse_as_dense=False,
                         op='Average'):
    """
    An optimizer that wraps another keras.optimizers.Optimizer, using an allreduce to average gradient
    values before applying gradients to model weights.
    :param optimizer:
    :param name:
    :param device_dense:
    :param device_sparse:
    :param sparse_as_dense:
    :param op:
    :return:
    """
    if op != 'Average' and op != 'Sum':
        raise ValueError('Op currently only supports Average and Sum.')

    return create_distributed_optimizer(
        optimizer=optimizer,
        name=name,
        device_dense=device_dense,
        device_sparse=device_sparse,
        backward_passes_per_step=backward_passes_per_step,
        average_aggregated_gradients=average_aggregated_gradients,
        sparse_as_dense=sparse_as_dense,
        op=op)


def create_distributed_optimizer(optimizer,
                                 name,
                                 device_dense,
                                 device_sparse,
                                 sparse_as_dense,
                                 op,
                                 backward_passes_per_step,
                                 average_aggregated_gradients=False):
    class _DistributedOptimizer(keras.optimizers.Optimizer):
        _HAS_AGGREGATE_GRAD = True

        def __init__(self, **kwargs):
            self._name = name or "Distributed%s" % self.__class__.__base__.__name__
            self._aggregated_gradients = False

            self._allreduce_grads = tips._make_allreduce_grads_fn(
                self._name, device_dense, device_sparse, sparse_as_dense, op)

            self._agg_helper = None
            if backward_passes_per_step > 1:
                if tips.utils.executing_eagerly():
                    self._agg_helper = LocalGradientAggregationHelperEager(
                        backward_passes_per_step=backward_passes_per_step,
                        allreduce_func=self._allreduce_grads,
                        sparse_as_dense=sparse_as_dense,
                        average_aggregated_gradients=
                        average_aggregated_gradients)
                else:
                    self._agg_helper = LocalGradientAggregationHelper(
                        backwrod_passes_per_step=backward_passes_per_step,
                        allreduce_func=self._allreduce_grads,
                        sparse_as_dense=sparse_as_dense,
                        average_aggregated_gradients_by_passes=
                        average_aggregated_gradients,
                        rank=tips.tips_basics.rank(),
                        optimizer_type=LocalGradientAggregationHelper.
                        _OPTIMIZER_TYPE_KERAS)
                    super(self.__class__, self).__init__(**kwargs)

        def _compute_gradients(self, loss, var_list, grad_loss=None,
                               tape=None):
            """
            Compute gradients of all trainable variables.
            :param loss:
            :param var_list:
            :param grad_loss:
            :param tape:
            :return:
            """
            if _PRE_TF_2_4_0:
                raise NotImplementedError(
                    'tensorflow version earlier than 2.4.0 is not supported')

            assert not tape
            grads_and_vars = super(self.__class__, self)._compute_gradients(
                loss, var_list, grad_loss, tape=tape)

            grads, weights = list(zip(*grads_and_vars))

            allreduced_grads = self._allreduce(grads, weights)
            return list(zip(allreduced_grads, weights))

        def get_gradients(self, loss, params):
            """
            Compute gradients of all trainable variables.
            :param loss:
            :param params:
            :return:
            """
            gradients = super(self.__class__, self).get_gradients(loss, params)
            return self._allreduce(gradients, params)

        def _aggregate_gradients(self, grads_and_vars):
            if _PRE_TF_2_4_0:
                # TODO(Superjomn) Support latter.
                raise NotImplementedError(
                    'tensorflow version earlier than 2.4.0 is not supported')

            else:
                return super(_DistributedOptimizer,
                             self)._aggregate_gradients(grads_and_vars)

        def apply_gradients(self, *args, **kwargs):
            if self._agg_helper:
                if isinstance(args[0], zip):
                    args = list(args)
                    args[0] = list(args[0])
                    args = tuple(args)

                results = self._agg_helper.apply_gradients(
                    lambda: super(self.__class__, self).apply_gradients(
                        *args, **kwargs), self, *args, **kwargs)
            else:
                results = super(self.__class__, self).apply_gradients(
                    *args, **kwargs)

            if _PRE_TF_2_4_0:
                # TODO(Superjomn) Support latter.
                raise NotImplementedError(
                    'tensorflow version earlier than 2.4.0 is not supported')

            return results

    cls = type(optimizer.__class__.__name__, (optimizer.__class__, ),
               dict(_DistributedOptimizer.__dict__))

    config = optimizer.get_config()
    if not _PRE_TF_2_4_0 and hasattr(optimizer, 'lr') and issubclass(
            optimizer.lr.__class__,
            keras.optimizers.schedules.LearningRateSchedule):
        lr_cls = type(optimizer.lr.__class__.__name__,
                      (optimizer.lr.__class__, ), dict(optimizer.lr.__dict__))
        config['learning_rate'] = lr_cls.from_config(
            config['learning_rate']['config'])

    return cls.from_config(config)


def _eval(backend, op_or_result):
    if tips.utils.executing_eagerly():
        return op_or_result
    return backend.get_session().run(op_or_result)
