#pragma once
#include <mpi.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <tensorflow/stream_executor/lib/statusor.h>

#include "tips/core/message/collective_messages_generated.h"
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

static Status TF_DataTypeToMessageDataType(tensorflow::DataType dtype, message::DataType* out) {
  switch (dtype) {
    case tensorflow::DataType::DT_FLOAT:
      *out = message::DataType_TF_FLOAT32;
      break;
    case tensorflow::DataType::DT_INT32:
      *out = message::DataType_TF_INT32;
      break;
    default:
      return tensorflow::errors::FailedPrecondition("Unknown type found");
  }
  return Status::OK();
}

/**
 * Do ring allreduce on CPU device.
 * We utilize the open_mpi allreduce method directlly.
 */
template <typename dtype>
Status AllreduceCpu(const Tensor* input, Tensor* output, CollectiveOpKind op) {
  CHECK(output->shape() == input->shape());
  CHECK(input->dtype() == output->dtype());

  // TODO(Superjomn) try the inplace way.
  // TODO(Superjomn) try the compress way(float16?)
  bool suc = MPI_Allreduce(input->data(),
                           output->data(),
                           input->shape().num_elements(),
                           mpi_type_trait<dtype>::type(),
                           CollectiveOpKindToMpiOp(op),
                           mpi_comm()) == 0;
  return suc ? Status::OK() : tensorflow::errors::FailedPrecondition("MPI_Allreduce failed");
}

// TODO(Superjomn) Replace with allgatherv ?
template <typename dtype>
Status AllgatherCpu(const Tensor* input, Tensor* output) {
  // TODO(Superjomn) try the inplace way.
  bool suc = MPI_Allgather(input->data(),
                           input->NumElements(),
                           mpi_type_trait<dtype>::type(),
                           output->data(),
                           input->NumElements(),
                           mpi_type_trait<dtype>::type(),
                           mpi_comm()) == 0;

  return suc ? Status::OK() : tensorflow::errors::FailedPrecondition("MPI_Allgather failed");
}

template <typename dtype>
Status AllreduceGpu(const Tensor* input, Tensor* output, CollectiveOpKind op);

}  // namespace collective
}  // namespace tips
