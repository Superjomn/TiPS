import tensorflow as tf
from tensorflow.python.eager import context


def executing_eagerly():
    return context.executing_eagerly()


def make_subgraph(f):
    return tf.function(f)


def cache(f):
    cache = dict()

    def wrapper(*args):
        key = (args, executing_eagerly())

        if key in cache:
            return cache[key]
        else:
            ret = f(*args)
            cache[key] = ret
            return ret

    return wrapper
