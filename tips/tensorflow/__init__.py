import os
import warnings
import tensorflow as tf

from tips.tensorflow.ops import allreduce_op, size_op, rank_op, tips_basics
from tips.tensorflow.utils import cache


def allreduce(tensor: tf.Tensor,
              device_dense: str = '',
              device_sparse: str = '',
              op: str = None,
              name: str = None) -> tf.Tensor:
    """
    Perform an allreduce on a tf.Tensor or tf.IndexedSlice.

    :param tensor: The shape must be identical accross all ranks.
    :param device_dense: The device to be used for dense tensors.
    :param device_sparse: The device to be used for sparse tensors.
    :param op: The reduction op.
    :param name: The name of the allreduce operation.
    :return: The result tensor.
    """
    if isinstance(tensor, tf.IndexedSlices):
        raise NotImplementedError('tf.IndexSlices not supported yet')
        if op == 'Adasum':
            raise NotImplementedError(
                'The Adasum does not support sparse tensors.')

        with tf.device(device_sparse):
            # need to support Allgatherv ?
            pass

    else:
        with tf.device(device_dense):
            mpi_size: int = tips_basics.size()  # cosntant value
            summed_tensor = allreduce_op(tensor, name)

            if op == 'Average':
                return summed_tensor / mpi_size
            else:
                raise NotImplementedError(
                    "Allreduce don't support op other than Average")


def _allreduce_cond(tensor: tf.Tensor, *args, **kwargs):
    ''' Only perform allreduce when there are more than one node. '''

    def allreduce_fn():
        return allreduce(tensor, *args, **kwargs)

    def id_fn():
        return tensor

    return tf.cond(
        tf.convert_to_tensor(tips_basics.size() > 1), allreduce_fn, id_fn)


@cache
def _make_cached_allreduce_grads_fn(name: str, device_dense: str,
                                    device_sparse: str, sparse_as_dense: bool,
                                    op: str):
    def allreduce_grads(grads, vars=None):
        with tf.name_scope(name + '_Allreduce'):
            if sparse_as_dense:
                grads = [
                    tf.convert_to_tensor(grad) if grad is not None
                    and isinstance(grad, tf.IndexedSlices) else grad
                    for grad in grads
                ]

            return [
                _allreduce_cond(
                    grad,
                    device_dense=device_dense,
                    device_sparse=device_sparse,
                    op=op) if grad is not None else grad for grad in grads
            ]


def _make_allreduce_grads_fn(name: str, device_dense: str, device_sparse: str,
                             sparse_as_dense: bool, op: str):
    return _make_cached_allreduce_grads_fn(name, device_dense, device_sparse,
                                           sparse_as_dense, op)


def DistributedOptimizer(optimizer,
                         name=None,
                         use_locking=False,
                         device_dense='',
                         device_sparse='',
                         sparse_as_dense=False,
                         backward_passes_per_step=1,
                         op='Average',
                         average_aggregated_gradients=False):
    """
    Construct a new DistributedOptimizer, which uses another optimizer under the hood for computing single-process
    gradient values and applying gradient updates after the gradient values have been combined accross all the ranks.

    :param optimizer:
    :param name:
    :param use_locking:
    :param device_dense:
    :param device_sparse:
    :param sparse_as_dense:
    :param backward_passes_per_step:
    :param op:
    :param average_aggregated_gradients:
    :return:
    """
    assert optimizer is tf.keras.optimizers.Optimizer, "legancy optimizer is not supported"
    if op == "Adasum":
        raise ValueError("op == Adasum is not supported yet with Keras")

    import tips.tensorflow.keras as _keras
    return _keras.DistributedOptimizer(
        optimizer=optimizer,
        name=name,
        device_dense=device_dense,
        device_sparse=device_sparse,
        sparse_as_dense=sparse_as_dense,
    )
