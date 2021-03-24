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

    def test_tips_rank(self):
        rank = self.evaluate(tips_ops.rank())
        print('rank', rank)

    def test_tips_size(self):
        size = self.evaluate(tips_ops.size())
        print('size', size)


from tensorflow.python.framework.test_util import run_all_in_graph_and_eager_modes
run_all_in_graph_and_eager_modes(TensorFlowTests)

if __name__ == '__main__':
    tf.test.main()
