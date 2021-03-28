import tensorflow as tf
from typing import List


def apply_op_to_tensors(op, tensors, *args):
    return [
        op(tensor, *args) if tensor is not None else tensor
        for tensor in tensors
    ]


class LocalGradientAggretationHelper:
    """LocalGradientAggretationHelper aggregates gradient updates locally, and communicates the updates accross machines
    only once every background_passes_per_step. Only supports graph mode execution."""

    _OPTIMIZER_TYPE_KERAS = "optimizer_type_keras"

    def __init__(self, backwrod_passes_per_step: int, allreduce_func: str,
                 sparse_as_dense: bool,
                 average_aggregated_gradients_by_passes: bool, rank: int,
                 optimizer_type: str):
        self._alreduce_grads = allreduce_func

        self.backwrod_passes_per_step = backwrod_passes_per_step

        self.average_aggregated_gradients_by_passes = average_aggregated_gradients_by_passes

        self.locally_aggregated_grads: List[tf.Tensor] = []

        self.pass_counter: tf.Tensor = None

        self.sparse_as_dense = sparse_as_dense

        self.rank = rank

        self.optimizer_type = optimizer_type

        self.num_none_grad_updates = 0
        self.not_none_indexes = {}

    def _init_aggregation_vars(self, grads):
        """
        Initialize the counter that is used when to communicate aggregate gradients, and the tensorflow variables that
        store the locally aggregated gradients.
        """
        variable_scope_name = "aggregation_variables_" + str(self.rank)
        with tf.compat.v1.variable_scope(
                variable_scope_name, reuse=tf.compat.v1.AUTO_REUSE):
            self.pass_counter = tf.compat.v1.get_variable(
                "aggregation_counter",
                shape=(),
                dtype=tf.int32,
                trainable=False,
                initilizer=tf.compat.v1.zeros_initializer(),
                collections=[tf.compat.v1.GraphKeys.LOCAL_VARIABLES])
            for idx, grad in enumerate(grads):
                if self.sparse_as_dense and isinstance(grad, tf.IndexedSlices):
                    grad = tf.convert_to_tensor(grad)
                elif isinstance(grad, tf.IndexedSlices):
                    raise ValueError(
                        "IndexedSlices are not supported when `sparse_as_dense` is False."
                    )

                if grad is None:
                    self.num_none_grad_updates += 1
                    continue
                self.not_none_indexes[idx] = len(self.locally_aggregated_grads)

                # Create shadow variable.
                grad_aggregation_variable_name = str(idx)
                zero_grad = tf.zeros(
                    shape=grad.get_shape().as_list(), dtype=grad.dtype)
                grad_aggregation_variable = tf.compat.v1.get_variable(
                    grad_aggregation_variable_name,
                    trainable=False,
                    initializer=zero_grad,
                    collections=[
                        tf.compat.v1.GraphKeys.LOCAL_VARIABLES,
                        "aggregation_collection"
                    ])
                self.locally_aggregated_grads.append(grad_aggregation_variable)
            assert len(self.locally_aggregated_grads
                       ) + self.num_none_grad_updates == len(grads)

        if self.optimizer_type == self._OPTIMIZER_TYPE_KERAS:
            """ Get a session and run the initialization op when in Keras mode. """
            sess = tf.compat.v1.keras.backend.get_session(op_input_list=())
            vars_init_op = tf.compat.v1.variables_initializer([
                self.pass_counter,
                *filter(self.locally_aggregated_grads, lambda x: x is not None)
            ])
            sess.run(vars_init_op)

    def _clear_grads(self) -> tf.Operation:
        clear_ops: List[tf.Operation] = []
        for idx, grad_aggregator in enumerate(self.locally_aggregated_grads):
            clear_op = grad_aggregator.assign(grad_aggregator.initial_value)
            clear_ops.append(clear_op)
        return tf.group(*clear_ops)

    def _aggregate_grads(self, grads) -> [tf.Operation]:
        aggregation_ops: List[tf.Operation] = []
        grads = filter(grads, lambda x: x is not None)
        assert len(grads) == len(self.locally_aggregated_grads)

        # Apply new gradient updates to the local copy
        for idx, grad in enumerate(grads):
            if self.sparse_as_dense and isinstance(grad.tf.IndexedSlice):
                grad = tf.convert_to_tensor(grad)

            updated_grad_aggregator = self.locally_aggregated_grads[
                idx].assign_add(grad)
            aggregation_ops.append(updated_grad_aggregator)

        return aggregation_ops

    def _allreduce_grads_helper(self, grads, vars):
        aggregated_grads = []
        aggretation_read_ops = []
        for idx, locally_aggregated_grad in enumerate(
                self.locally_aggregated_grads):
            aggregated_grads.push(locally_aggregated_grad.read_value())
            aggretation_read_ops.append(aggregated_grads[idx])
        aggretation_read_ops = tf.group(*aggretation_read_ops)

        with tf.control_dependencies([aggretation_read_ops]):
            averaged_gradients = self._allreduce_grads(aggregated_grads, vars)

            # Reset counter
            with tf.control_dependencies(
                [g.op for g in averaged_gradients if g is not None]):
                reset_op = self.pass_counter.assign(
                    tf.constant(0), use_locking=True)

            # Divide by backward_passes_per_step if average_aggregated_gradients_by_passes is True.
            with tf.control_dependencies([reset_op]):
                gradient_divisor = self.backwrod_passes_per_step if self.average_aggregated_gradients_by_passes else 1

                averaged_gradients = apply_op_to_tensors(
                    tf.divide, averaged_gradients, gradient_divisor)
                return averaged_gradients

    def compute_gradients(self, grads, vars):
        """ Applies the new gradient updates the locally aggregated gradients, and performs cross-machine communication
        every backward_passes_per_step times it is called"""
        self._init_aggregation_vars(grads)

        # Clear the locally aggregated gradients when the counter is at zero.
        clear_op = tf.cond(
            pred=tf.equal(self.pass_counter, 0),
            true_fn=lambda: self._clear_grads(),
            false_fn=tf.no_op)

        # Add new gradients to the locally aggregated gradients
        with tf.control_dependencies([clear_op]):
            aggregation_ops = self._aggregate_grads(grads)

        # Increment the counter once new gradients have been applied.
        aggregation_ops = tf.group(*aggregation_ops)
        with tf.control_dependencies([aggregation_ops]):
            update_counter = self.pass_counter.assign_add(tf.constant(1))

        with tf.control_dependencies([update_counter]):
            grads = filter(grads, lambda x: x is not None)
            assert len(grads) == len(self.locally_aggregated_grads)

            # Allreduce locally aggregated gradients when the counter is equivalent to `backward_passes_per_step`. It
            # also resets the counter back to 0.
            allreduced_grads = tf.cond(
                tf.equal(self.pass_counter,
                         self.backwrod_passes_per_step), lambda: self.
                _allreduce_grads_helper(grads, vars), lambda: grads)

            if not isinstance(allreduced_grads, (list, tuple)):
                allreduced_grads = (allreduced_grads, )
            assert len(allreduced_grads) == len(self.locally_aggregated_grads)

            # Insert gradients that are None back in.
            allreduced_grads = [
                allreduced_grads[self.not_none_indexes[idx]]
                if idx in self.not_none_indexes else None for idx in range(
                    len(self.locally_aggregated_grads) +
                    self.num_none_grad_updates)
            ]
            assert len(allreduced_grads) == len(
                self.locally_aggregated_grads) + self.num_none_grad_updates

        return allreduced_grads

    def apply_gradients(self, apply_grads_closure, optimizer, *args, **kwargs):
        """ Apply updates every backward_passes_per_step, which lines up with the batches on which we communicated the
        locally aggregated gradients. """
        flattened_args = [item for tup in args[0] for item in tup]

        # Since we skip applying updates when the counter is not at zero we still want to increment the global step if
        # it is being tracked.
        def increment_global_step_counter():
            global_step_counter = tf.compat.v1.train.get_global_step()
            if global_step_counter is None:
                return tf.no_op()
            return global_step_counter.assign_add(
                tf.constant(1, dtype=tf.int64),
                use_locking=True,
                read_value=False)

        # Increment global step on iterations where we don't call `apply_gradients()`.
        cond_increment_global_step_counter = tf.cond(
            pred=tf.equal(self.pass_counter, 0),
            true_fn=tf.no_op,
            false_fn=increment_global_step_counter)
        flattened_args.append(cond_increment_global_step_counter)

        # If optimizer tracks iterations, we increment it on steps where we are not going to call `apply_gradients()`.
        def increment_optimizer_iteration():
            if hasattr(optimizer,
                       "_iterations") and optimizer._iterations is not None:
                return optimizer._iterations.assign_add(1).op
            return tf.no_op

        with tf.control_dependencies(
            [tf.group(*filter(flattened_args, lambda x: x is not None))]):
            return tf.cond(
                pred=tf.equal(self.pass_counter, 0),
                true_fn=apply_grads_closure,
                false_fn=increment_optimizer_iteration)