import re
import logging
import tensorflow as tf
from tensorflow.python.framework import load_library
from tensorflow.python.framework import ops
from tensorflow.python.platform import resource_loader
from .basics import TipsBasics
from .utils import executing_eagerly


def _load_library(name, op_list=None):
    """
    Loads a .so file containing some operators.
    """
    try:
        filename = resource_loader.get_path_to_datafile(name)
        library = load_library.load_op_library(filename)
        return library

    except FileNotFoundError:
        logging.warning("%s file could not be loaded.", name)


MPI_LIB = _load_library('libtipscore.so')

tips_basics = TipsBasics("libtipscore.so")

size = tips_basics.size
rank = tips_basics.rank
shutdown = tips_basics.shutdown


def _normialize_name(name):
    """Normalizes operation name to TensorFlow rules."""
    return re.sub('[^a-zA-Z0-9_]', '_', name)


def _allreduce(tensor: tf.Tensor, name=None):
    if name is None and not executing_eagerly():
        name = 'TipsAllreduce_%s' % _normialize_name(tensor.name)

    return MPI_LIB.tips_allreduce(tensor, name=name)


def size_op(name=None):
    """An op which returns the number of MPI processes."""
    return MPI_LIB.mpi_size(name=name)


ops.NotDifferentiable("MPISize")


def rank_op(name=None):
    """An op which returns the number of MPI processes."""
    return MPI_LIB.mpi_rank(name=name)


ops.NotDifferentiable("MPIRank")


def allreduce_op(tensor, name=None):
    """An op which sums an input tensor over all the MPI processes."""
    return MPI_LIB.mpi_allreduce(tensor, name=name)


ops.NotDifferentiable("MPIAllreduce")


def allgather_op(tensor, name=None):
    """An op which broadcast an input tensor over all the MPI processes."""
    return MPI_LIB.mpi_allgather(tensor, name=name)


ops.NotDifferentiable("MPIAllgather")


def broadcast_op(tensor, root_rank=0, name=None):
    """An op which broadcast an input tensor over all the MPI processes."""
    return MPI_LIB.mpi_broadcast(tensor, name=name, root_rank=root_rank)


ops.NotDifferentiable("MPIBroadcast")
