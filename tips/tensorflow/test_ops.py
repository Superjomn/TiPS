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
                [17] * dim, -100, 100, dtype=tf.float32)
            summed = tips_ops.allreduce_op(tensor)
            print(summed)

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
