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


def vars_to_refs(vars):
    if isinstance(vars, list):
        return tuple(vars_to_refs(v) for v in vars)
    return vars.ref()


def refs_to_vars(refs):
    if isinstance(refs, tuple):
        return [refs_to_vars(r) for r in refs]
    return refs.deref()
