import os
import warnings
import tensorflow as tf
import inspect
from tensorflow import keras
from tensorflow.python.keras import backend as K

from tips.tensorflow.ops import allreduce_op, size_op, rank_op, rank, size, shutdown, allgather_op, broadcast_op
from tips.tensorflow.utils import cache, vars_to_refs, refs_to_vars, make_subgraph
from tips.tensorflow.compression import Compression
from tips.tensorflow.utils import executing_eagerly
from tips.tensorflow.functions import broadcast_variables
from tips.tensorflow.gradient_aggregation import LocalGradientAggregationHelper
from tips.tensorflow.gradient_aggregation_eager import LocalGradientAggregationHelperEager

Average = 'Average'
Sum = 'Sum'


def allreduce(tensor,
              average=None,
              device_dense='',
              device_sparse='',
              compression=Compression.none,
              op=None,
              prescale_factor=1.0,
              postscale_factor=1.0,
              name=None):
    """Perform an allreduce on a tf.Tensor or tf.IndexedSlices.

    This function performs a bandwidth-optimal ring allreduce on the input
    tensor. If the input is an tf.IndexedSlices, the function instead does an
    allgather on the values and the indices, effectively doing an allreduce on
    the represented tensor.

    Arguments:
        tensor: tf.Tensor, tf.Variable, or tf.IndexedSlices to reduce.
                The shape of the input must be identical across all ranks.
        average:
            .. warning:: .. deprecated:: 0.19.0

                Use `op` instead. Will be removed in v0.21.0.

        device_dense: Device to be used for dense tensors. Uses GPU by default
                      if Horovod was built with HOROVOD_GPU_OPERATIONS.
        device_sparse: Device to be used for sparse tensors. Uses GPU by default
                       if Horovod was built with HOROVOD_GPU_OPERATIONS.
        compression: Compression algorithm used to reduce the amount of data
                     sent and received by each worker node.  Defaults to not
                     using compression.
        op: The reduction operation to combine tensors across different ranks.
            Defaults to Average if None is given.
        prescale_factor: Multiplicative factor to scale tensor before allreduce.
        postscale_factor: Multiplicative factor to scale tensor after allreduce.
        name: A name of the allreduce operation

    Returns:
        A tensor of the same shape and type as `tensor`, summed across all
        processes.
    """
    if isinstance(tensor, tf.IndexedSlices):
        # TODO: Need to fix this to actuall call Adasum
        with tf.device(device_sparse):
            # For IndexedSlices, do two allgathers instead of an allreduce.
            tips_size = tf.cast(
                size_op()
                if int(os.environ.get("HOROVOD_ELASTIC", 0)) else size(),
                dtype=tensor.values.dtype)
            values = allgather_op(tensor.values)
            indices = allgather_op(tensor.indices)

            # To make this operation into an average, divide allgathered values by
            # the Horovod size.
            new_values = (values / tips_size) if op == Average else values
        return tf.IndexedSlices(
            new_values, indices, dense_shape=tensor.dense_shape)
    else:
        average_in_framework = False
        with tf.device(device_dense):
            #tips_size = tf.cast(size_op(), dtype=tensor.dtype)
            tensor_compressed, ctx = compression.compress(tensor)
            summed_tensor_compressed = allreduce_op(
                tensor_compressed,
                #op=op,
                #prescale_factor=prescale_factor,
                #postscale_factor=postscale_factor,
                name=name)
            summed_tensor = compression.decompress(summed_tensor_compressed,
                                                   ctx)
            new_tensor = summed_tensor
        return new_tensor


def _allreduce_cond(tensor, *args, **kwargs):
    def allreduce_fn():
        return allreduce(tensor, *args, **kwargs)

    def id_fn():
        return tensor

    return tf.cond(
        (size_op() > 1) if int(os.environ.get("HOROVOD_ELASTIC", 0)) else
        tf.convert_to_tensor(size() > 1), allreduce_fn, id_fn)


try:
    _global_variables = tf.compat.v1.global_variables
except AttributeError:
    try:
        _global_variables = tf.global_variables
    except AttributeError:
        _global_variables = None

if _global_variables is not None:

    def broadcast_global_variables(root_rank):
        """Broadcasts all global variables from root rank to all other processes.

        **NOTE:** deprecated in TensorFlow 2.0.

        Arguments:
            root_rank: rank of the process from which global variables will be broadcasted
                       to all other processes.
        """
        if executing_eagerly():
            raise RuntimeError(
                "hvd.broadcast_global_variables() does not support eager execution. "
                "Please use `hvd.broadcast_variables(<model/optimizer variables>)` instead."
            )

        return broadcast_variables(_global_variables(), root_rank)


try:
    _get_default_graph = tf.compat.v1.get_default_graph
except AttributeError:
    try:
        _get_default_graph = tf.get_default_graph
    except AttributeError:
        _get_default_graph = None

try:
    _SessionRunHook = tf.estimator.SessionRunHook
except AttributeError:
    try:
        _SessionRunHook = tf.train.SessionRunHook
    except AttributeError:
        _SessionRunHook = None

if _SessionRunHook is not None and _get_default_graph is not None:

    class BroadcastGlobalVariablesHook(_SessionRunHook):
        """
        SessionRunHook that will broadcast all global variables from root rank
        to all other processes during initialization.

        This is necessary to ensure consistent initialization of all workers when
        training is started with random weights or restored from a checkpoint.

        **NOTE:** deprecated in TensorFlow 2.0.
        """

        def __init__(self, root_rank, device=''):
            """Construct a new BroadcastGlobalVariablesHook that will broadcast all
            global variables from root rank to all other processes during initialization.

            Args:
              root_rank:
                Rank that will send data, other ranks will receive data.
              device:
                Device to be used for broadcasting. Uses GPU by default
                if Horovod was built with HOROVOD_GPU_OPERATIONS.
            """
            super(BroadcastGlobalVariablesHook, self).__init__()
            self.root_rank = root_rank
            self.bcast_op = None
            self.device = device

        def begin(self):
            if not self.bcast_op or self.bcast_op.graph != _get_default_graph(
            ):
                with tf.device(self.device):
                    self.bcast_op = broadcast_global_variables(self.root_rank)

        def after_create_session(self, session, coord):
            session.run(self.bcast_op)


@cache
def _make_cached_allreduce_grads_fn(name, device_dense, device_sparse,
                                    compression, sparse_as_dense, op,
                                    gradient_predivide_factor, groups):
    groups = refs_to_vars(groups) if isinstance(groups, tuple) else groups
    if op == Average:
        # Split average operation across pre/postscale factors
        # C++ backend will apply additional 1 / size() factor to postscale_factor for op == Average.
        prescale_factor = 1.0 / gradient_predivide_factor
        postscale_factor = gradient_predivide_factor
    else:
        prescale_factor = 1.0
        postscale_factor = 1.0

    def allreduce_grads(grads, vars=None):
        with tf.name_scope(name + "_Allreduce"):
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
                    compression=compression,
                    op=op,
                    prescale_factor=prescale_factor,
                    postscale_factor=postscale_factor)
                if grad is not None else grad for grad in grads
            ]

    if executing_eagerly():
        return make_subgraph(allreduce_grads)
    else:
        return allreduce_grads


def _make_allreduce_grads_fn(name, device_dense, device_sparse, compression,
                             sparse_as_dense, op, gradient_predivide_factor,
                             groups):
    groups = vars_to_refs(groups) if isinstance(groups, list) else groups
    return _make_cached_allreduce_grads_fn(name, device_dense, device_sparse,
                                           compression, sparse_as_dense, op,
                                           gradient_predivide_factor, groups)


try:
    # TensorFlow 2.x
    _LegacyOptimizer = tf.compat.v1.train.Optimizer
except AttributeError:
    try:
        # TensorFlow 1.x
        _LegacyOptimizer = tf.train.Optimizer
    except AttributeError:
        # Future TensorFlow versions
        _LegacyOptimizer = None

if _LegacyOptimizer is not None:

    class _DistributedOptimizer(_LegacyOptimizer):
        """An optimizer that wraps another tf.Optimizer, using an allreduce to
        combine gradient values before applying gradients to model weights."""

        def __init__(self,
                     optimizer,
                     name=None,
                     use_locking=False,
                     device_dense='',
                     device_sparse='',
                     compression=Compression.none,
                     sparse_as_dense=False,
                     op=Average,
                     gradient_predivide_factor=1.0,
                     backward_passes_per_step=1,
                     average_aggregated_gradients=False,
                     groups=None):
            if name is None:
                name = "Distributed{}".format(type(optimizer).__name__)
            super(_DistributedOptimizer, self).__init__(
                name=name, use_locking=use_locking)

            self._optimizer = optimizer
            self._allreduce_grads = _make_allreduce_grads_fn(
                name, device_dense, device_sparse, compression,
                sparse_as_dense, op, gradient_predivide_factor, groups)

            self._agg_helper = None
            if backward_passes_per_step > 1:
                if executing_eagerly():
                    raise ValueError(
                        "backward_passes_per_step > 1 is not yet supported "
                        "for _LegacyOptimizer with eager execution.")

                self._agg_helper = LocalGradientAggregationHelper(
                    backward_passes_per_step=backward_passes_per_step,
                    allreduce_func=self._allreduce_grads,
                    sparse_as_dense=sparse_as_dense,
                    average_aggregated_gradients=average_aggregated_gradients,
                    rank=rank(),
                    optimizer_type=LocalGradientAggregationHelper.
                    _OPTIMIZER_TYPE_LEGACY,
                )

        def compute_gradients(self, *args, **kwargs):
            """Compute gradients of all trainable variables.

            See Optimizer.compute_gradients() for more info.

            In DistributedOptimizer, compute_gradients() is overriden to also
            allreduce the gradients before returning them.
            """
            gradients = self._optimizer.compute_gradients(*args, **kwargs)
            grads, vars = zip(*gradients)
            if self._agg_helper:
                avg_grads = self._agg_helper.compute_gradients(grads, vars)
            else:
                avg_grads = self._allreduce_grads(grads, vars)
            return list(zip(avg_grads, vars))

        def apply_gradients(self, *args, **kwargs):
            """Calls this same method on the underlying optimizer."""
            if self._agg_helper:
                return self._agg_helper.apply_gradients(
                    lambda: self._optimizer.apply_gradients(*args, **kwargs),
                    self._optimizer,
                    *args,
                    **kwargs,
                )

            return self._optimizer.apply_gradients(*args, **kwargs)

        def get_slot(self, *args, **kwargs):
            """Calls this same method on the underlying optimizer."""
            return self._optimizer.get_slot(*args, **kwargs)

        def get_slot_names(self, *args, **kwargs):
            """Calls this same method on the underlying optimizer."""
            return self._optimizer.get_slot_names(*args, **kwargs)

        def variables(self, *args, **kwargs):
            """Calls this same method on the underlying optimizer."""
            return self._optimizer.variables(*args, **kwargs)


def DistributedOptimizer(optimizer,
                         name=None,
                         use_locking=False,
                         device_dense='',
                         device_sparse='',
                         compression=Compression.none,
                         sparse_as_dense=False,
                         backward_passes_per_step=1,
                         op=Average,
                         gradient_predivide_factor=1.0,
                         average_aggregated_gradients=False,
                         num_groups=0,
                         groups=None):
    """Construct a new DistributedOptimizer, which uses another optimizer
    under the hood for computing single-process gradient values and
    applying gradient updates after the gradient values have been combined
    across all the Horovod ranks.

    Args:
      optimizer:
        Optimizer to use for computing gradients and applying updates.
      name:
        Optional name prefix for the operations created when applying
        gradients. Defaults to "Distributed" followed by the provided
        optimizer type.
      use_locking:
        Whether to use locking when updating variables.
        See Optimizer.__init__ for more info.
      device_dense:
        Device to be used for dense tensors. Uses GPU by default
        if Horovod was built with HOROVOD_GPU_OPERATIONS.
      device_sparse:
        Device to be used for sparse tensors. Uses GPU by default
        if Horovod was built with HOROVOD_GPU_OPERATIONS.
      compression:
        Compression algorithm used during allreduce to reduce the amount
        of data sent during each parameter update step.  Defaults to
        not using compression.
      sparse_as_dense:
        Treat all sparse gradients as dense tensors.  This can help improve
        performance and memory utilization if the original sparse gradient
        has high density.  Defaults to false.
      backward_passes_per_step:
        Number of backward passes to perform before calling hvd.allreduce.
        This allows accumulating updates over multiple mini-batches before
        reducing and applying them.
      op:
        The reduction operation to use when combining gradients across
        different ranks.
      gradient_predivide_factor:
        If op == Average, gradient_predivide_factor splits the averaging
        before and after the sum. Gradients are scaled by
        1.0 / gradient_predivide_factor before the sum and
        gradient_predivide_factor / size after the sum.
      average_aggregated_gradients:
        Whether to average the aggregated gradients that have been accumulated
        over multiple mini-batches. If true divides gradients updates by
        backward_passes_per_step. Only applicable for backward_passes_per_step > 1.
      num_groups:
        Number of groups to assign gradient allreduce ops to for explicit
        grouping. Defaults to no explicit groups.
      groups:
        The parameter to group the gradient allreduce ops. Accept values is a
        non-negative integer or a list of list of tf.Variable.
        If groups is a non-negative integer, it is the number of groups to assign
        gradient allreduce ops to for explicit grouping.
        If groups is a list of list of tf.Variable. Variables in the same
        inner list will be assigned to the same group, while parameter that does
        not appear in any list will form a group itself.
        Defaults as None, which is no explicit groups.
    """
    if gradient_predivide_factor != 1.0:
        if op != Average:
            raise ValueError(
                'gradient_predivide_factor not supported with op != Average')

    if num_groups != 0:
        warnings.warn(
            'Parameter `num_groups` has been replaced by `groups` '
            'and will be removed in v0.23.0.', DeprecationWarning)
        if groups is None:
            groups = num_groups

    if groups is not None:
        if not (isinstance(groups, list) or groups > 0):
            raise ValueError('groups should be a non-negative integer or '
                             'a list of list of tf.Variable.')

    if isinstance(optimizer, _LegacyOptimizer):

        return _DistributedOptimizer(
            optimizer=optimizer,
            name=name,
            use_locking=use_locking,
            device_dense=device_dense,
            device_sparse=device_sparse,
            compression=compression,
            sparse_as_dense=sparse_as_dense,
            op=op,
            gradient_predivide_factor=gradient_predivide_factor,
            backward_passes_per_step=backward_passes_per_step,
            average_aggregated_gradients=average_aggregated_gradients,
            groups=groups)
    elif isinstance(optimizer, tf.keras.optimizers.Optimizer):
        import horovod.tensorflow.keras as hvd_k
        return hvd_k.DistributedOptimizer(
            optimizer=optimizer,
            name=name,
            device_dense=device_dense,
            device_sparse=device_sparse,
            compression=compression,
            sparse_as_dense=sparse_as_dense,
            gradient_predivide_factor=gradient_predivide_factor,
            backward_passes_per_step=backward_passes_per_step,
            average_aggregated_gradients=average_aggregated_gradients,
        )
    else:
        raise ValueError(
            'Provided optimizer doesn\'t inherit from either legacy '
            'TensorFlow or Keras optimizer: %s' % optimizer)


if hasattr(tf, 'GradientTape'):

    class _DistributedGradientTape(tf.GradientTape):
        def __init__(self,
                     tape,
                     device_dense,
                     device_sparse,
                     compression,
                     sparse_as_dense,
                     op,
                     gradient_predivide_factor,
                     groups,
                     persistent=False,
                     watch_accessed_variables=True):
            if hasattr(tape, '_watch_accessed_variables'):
                super(self.__class__, self).__init__(persistent,
                                                     watch_accessed_variables)
            else:
                super(self.__class__, self).__init__(persistent)

            self._tape = tape
            self._allreduce_grads = _make_allreduce_grads_fn(
                'DistributedGradientTape', device_dense, device_sparse,
                compression, sparse_as_dense, op, gradient_predivide_factor,
                groups)

        def gradient(self, target, sources, output_gradients=None):
            gradients = super(self.__class__, self).gradient(
                target, sources, output_gradients)
            return self._allreduce_grads(gradients, sources)

    def DistributedGradientTape(gradtape,
                                device_dense='',
                                device_sparse='',
                                compression=Compression.none,
                                sparse_as_dense=False,
                                op=Average,
                                gradient_predivide_factor=1.0,
                                num_groups=0,
                                groups=None):
        """A tape that wraps another tf.GradientTape, using an allreduce to
        combine gradient values before applying gradients to model weights.

        Args:
          gradtape:
            GradientTape to use for computing gradients and applying updates.
          device_dense:
            Device to be used for dense tensors. Uses GPU by default
            if Horovod was built with HOROVOD_GPU_OPERATIONS.
          device_sparse:
            Device to be used for sparse tensors. Uses GPU by default
            if Horovod was built with HOROVOD_GPU_OPERATIONS.
          compression:
            Compression algorithm used during allreduce to reduce the amount
            of data sent during each parameter update step.  Defaults to
            not using compression.
          sparse_as_dense:
            Treat all sparse gradients as dense tensors.  This can help improve
            performance and memory utilization if the original sparse gradient
            has high density.  Defaults to false.
          op:
            The reduction operation to use when combining gradients across
            different ranks.
          gradient_predivide_factor:
            If op == Average, gradient_predivide_factor splits the averaging
            before and after the sum. Gradients are scaled by
            1.0 / gradient_predivide_factor before the sum and
            gradient_predivide_factor / size after the sum.
          num_groups:
            Number of groups to assign gradient allreduce ops to for explicit
            grouping. Defaults to no explicit groups.
          groups:
            The parameter to group the gradient allreduce ops. Accept values is a
            non-negative integer or a list of list of tf.Variable.
            If groups is a non-negative integer, it is the number of groups to assign
            gradient allreduce ops to for explicit grouping.
            If groups is a list of list of tf.Variable. Variables in the same
            inner list will be assigned to the same group, while parameter that does
            not appear in any list will form a group itself.
            Defaults as None, which is no explicit groups.
        """
        if gradient_predivide_factor != 1.0:
            if op != Average:
                raise ValueError(
                    'gradient_predivide_factor not supported with op != Average'
                )

        if num_groups != 0:
            warnings.warn(
                'Parameter `num_groups` has been replaced by `groups` '
                'and will be removed in v0.23.0.', DeprecationWarning)
            if groups is None:
                groups = num_groups

        if groups is not None:
            if not (isinstance(groups, list) or groups > 0):
                raise ValueError('groups should be a non-negative integer or '
                                 'a list of list of tf.Variable.')

        cls = type(gradtape.__class__.__name__, (gradtape.__class__, ),
                   dict(_DistributedGradientTape.__dict__))
        if hasattr(gradtape, '_watch_accessed_variables'):
            return cls(gradtape._tape, device_dense, device_sparse,
                       compression, sparse_as_dense, op,
                       gradient_predivide_factor, groups, gradtape._persistent,
                       gradtape._watch_accessed_variables)
        else:
            return cls(gradtape._tape, device_dense, device_sparse,
                       compression, sparse_as_dense, op,
                       gradient_predivide_factor, groups, gradtape._persistent)
