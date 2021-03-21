#pragma once
#include <mpi.h>
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <tensorflow/stream_executor/lib/statusor.h>

#include "tips/core/mpi/tips_mpi.h"

namespace tips {
namespace collective {

using tensorflow::OpKernelContext;
using tensorflow::Status;
using tensorflow::Tensor;

template <typename T>
using StatusOr = stream_executor::port::StatusOr<T>;

enum class CollectiveOpKind {
  SUM = 0,
  MAX = 1,
  MIN = 2,
};

MPI_Op CollectiveOpKindToMpiOp(CollectiveOpKind op);

/**
 * Do ring allreduce on CPU device.
 * We utilize the open_mpi allreduce method directlly.
 */
template <typename dtype>
Status AllreduceCpu(const Tensor* input, Tensor* output, CollectiveOpKind op) {
  const dtype* buffer = reinterpret_cast<const dtype*>(input->tensor_data().data());
  CHECK(output->shape() == input->shape());

  // TODO(Superjomn) try the inplace way.
  ZCHECK(MPI_Allreduce(buffer,
                       output->data(),
                       input->tensor_data().size(),
                       mpi_type_trait<dtype>::type(),
                       CollectiveOpKindToMpiOp(op),
                       mpi_comm()));
}

// TODO(Superjomn) Replace with allgatherv ?
template <typename dtype>
Status AllgatherCpu(const Tensor* input, Tensor* output) {
  const dtype* buffer = reinterpret_cast<const dtype*>(input->tensor_data().data());
  // TODO(Supejomn) do shape check.

  // TODO(Superjomn) try the inplace way.
  ZCHECK(MPI_Allgather(buffer,
                       input->tensor_data().size(),
                       mpi_type_trait<dtype>::type(),
                       output->data(),
                       input->tensor_data().size(),
                       mpi_type_trait<dtype>::type(),
                       mpi_comm()));
}

template <typename dtype>
Status AllreduceGpu(const Tensor* input, Tensor* output, CollectiveOpKind op);

}  // namespace collective
}  // namespace tips
