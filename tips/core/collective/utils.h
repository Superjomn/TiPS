#pragma once
#include <absl/types/span.h>
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

template <typename T>
Status AllgathervCpu(const Tensor* input, absl::Span<int> first_ranks, Tensor* output) {
  int first_rank = std::accumulate(first_ranks.begin(), first_ranks.end(), 0, [](int a, int b) { return a + b; });
  CHECK_EQ(first_ranks.size(), mpi_size());
  if (input->dims() != output->dims()) {
    return tensorflow::errors::FailedPrecondition("input and output tensors shape not match");
  }
  if (first_rank != output->dim_size(0)) {
    return tensorflow::errors::FailedPrecondition("output tensor first rank not match");
  }

  int elems_of_remain_rank = 1;
  for (int i = 1; i < input->dims(); i++) {
    if (input->dim_size(i) != output->dim_size(i)) {
      return tensorflow::errors::FailedPrecondition(
          "input and output tensor %d-rank not match %d vs %d", i, input->dim_size(i), output->dim_size(i));
    }
    elems_of_remain_rank *= input->dim_size(i);
  }
  if (input->dim_size(0) != first_ranks[mpi_rank()]) {
    return tensorflow::errors::FailedPrecondition(
        "input and first_ranks not match %d vs %d", input->dim_size(0), first_ranks[mpi_rank()]);
  }

  std::vector<int> disps(first_ranks.size(), 0);
  for (int i = 1; i < disps.size(); i++) {
    disps[i] = disps[i - 1] + elems_of_remain_rank * first_ranks[i];
  }

  bool suc = MPI_Allgatherv(input->data(),
                            input->NumElements(),
                            mpi_type_trait<T>::type(),
                            output->data(),
                            &first_ranks[0],
                            disps.data(),
                            mpi_type_trait<T>::type(),
                            mpi_comm()) == 0;

  return suc ? Status::OK() : tensorflow::errors::FailedPrecondition("MPI_Allgather failed");
}

template <typename T>
Status BroadcastCpu(const Tensor* input, int root) {
  bool suc = MPI_Bcast(input->data(), input->NumElements(), mpi_type_trait<T>::type(), root, mpi_comm()) == 0;
  return suc ? Status::OK() : tensorflow::errors::FailedPrecondition("MPI_Bcast failed");
}

template <typename dtype>
Status AllreduceGpu(const Tensor* input, Tensor* output, CollectiveOpKind op);

}  // namespace collective
}  // namespace tips
