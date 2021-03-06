import tensorflow as tf
from tensorflow.python.framework import ops
import tips.tensorflow as tips
from tips.tensorflow import ops as tips_ops
from tips.tensorflow.utils import executing_eagerly


class TensorFlowTests(tf.test.TestCase):
    def __init__(self, *args, **kwargs):
        super(TensorFlowTests, self).__init__(*args, **kwargs)
        if hasattr(tf, 'contrib') and hasattr(tf.contrib, 'eager'):
            self.tfe = tf.contrib.eager
        else:
            self.tfe = tf

    def evaluate(self, tensors):
        if executing_eagerly():
            return self._eval_helper(tensors)
        sess = ops.get_default_session()
        if sess is None:
            with self.test_session() as sess:
                return sess.run(tensors)
        else:
            return sess.run(tensors)

    def test_tips_rank_op(self):
        rank = self.evaluate(tips_ops.rank_op())
        print('rank', rank)

    def test_tips_size_op(self):
        size = self.evaluate(tips_ops.size_op())
        print('size', size)

    def test_tips_allreduce_cpu(self):
        dim = 3
        with tf.device("/cpu:0"):
            tensor = self._random_uniform(
                [3] * dim, -100, 100, dtype=tf.float32)
            summed = tips_ops.allreduce_op(tensor, name="allreduce__")
            print(summed)

    def test_tips_allgather_cpu(self):
        rank = self.evaluate(tips_ops.rank_op())

        dim = 3
        with tf.device("/cpu:0"):
            tensor = tf.constant(rank, tf.float32, [rank + 1, 3, 3])
            gathered = tips_ops.allgather_op(tensor)
            print(gathered)

    def test_tips_broadcast_cpu(self):
        rank = self.evaluate(tips_ops.rank_op())

        dim = 3
        with tf.device("/cpu:0"):
            tensor = tf.constant(rank, tf.float32, [4, 3, 3])
            gathered = tips_ops.broadcast_op(tensor)
            print(gathered)

    def test_tips_broadcast_scalar_cpu(self):
        with tf.device("/cpu:0"):
            i64 = tf.constant(tips_ops.rank(), tf.int64, [])
            dbl = tf.constant(tips_ops.rank(), tf.float64, [])
            gathered = tips_ops.broadcast_op(i64, name="i64_broadcast")
            print(gathered)
            #gathered = tips_ops.broadcast_op(i64, name="i64_broadcast")
            gathered1 = tips_ops.broadcast_op(dbl, name="double_broadcast")

    def test_tips_allreduce_scalar_cpu(self):
        with tf.device("/cpu:0"):
            i64 = tf.constant(tips_ops.rank(), tf.int64, [], name="a")
            dbl = tf.constant(tips_ops.rank(), tf.float64, [])
            gathered = tips_ops.allreduce_op(i64)
            gathered1 = tips_ops.allreduce_op(dbl)

    def _random_uniform(self, *args, **kwargs):
        if hasattr(tf, 'random') and hasattr(tf.random, 'set_seed'):
            tf.random.set_seed(1234)
            return tf.random.uniform(*args, **kwargs)
        else:
            tf.set_random_seed(1234)
            return tf.random_uniform(*args, **kwargs)


from tensorflow.python.framework.test_util import run_all_in_graph_and_eager_modes
run_all_in_graph_and_eager_modes(TensorFlowTests)

if __name__ == '__main__':
    tf.test.main()
