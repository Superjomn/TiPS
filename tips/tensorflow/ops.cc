#include "tips/tensorflow/ops.h"

#include "tips/core/collective/coordinator.h"
#include "tips/core/mpi/tips_mpi.h"

namespace tips {
namespace collective {
using namespace tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;

Status IsMpiIntialized() {
  if (!MpiContext::Global().IsInitialized()) return errors::FailedPrecondition("MPI is not initialized");
  if (!RpcServer::Global().initialized()) return errors::FailedPrecondition("Global RPC server is not initialized");
  return Status::OK();
}

template <typename Device>
class MpiSizeOp : public tensorflow::OpKernel {
 public:
  explicit MpiSizeOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    OP_REQUIRES_OK(context, IsMpiIntialized());
    // Write integer to output tensor
    Tensor* output;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({}), &output));

    auto flat = output->flat<int>();
    flat(0)   = mpi_size();
  }
};

REGISTER_KERNEL_BUILDER(Name("MPISize").Device(DEVICE_CPU), MpiSizeOp<CPUDevice>);

REGISTER_OP("MPISize")
    .Output("size: int32")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext* ctx) {
      ctx->set_output(0, ctx->Scalar());
      return Status::OK();
    })
    .Doc(R"doc(
Returns the number of running MPI processes.

size: Size of the MPI group.
)doc");

template <typename Device>
class MpiRankOp : public OpKernel {
 public:
  explicit MpiRankOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    OP_REQUIRES_OK(context, IsMpiIntialized());

    Tensor* output;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({}), &output));

    auto flat = output->flat<int>();
    flat(0)   = mpi_rank();
  }
};

REGISTER_KERNEL_BUILDER(Name("MPIRank").Device(DEVICE_CPU), MpiRankOp<CPUDevice>);

REGISTER_OP("MPIRank")
    .Output("rank: int32")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Scalar());
      return Status::OK();
    })
    .Doc(R"doc(
Returns the index of the current process in the MPI group.
)doc");

template <typename Device>
class MpiAllreduceOp : public AsyncOpKernel {
 public:
  explicit MpiAllreduceOp(OpKernelConstruction* context) : AsyncOpKernel(context) {}

  bool IsExpensive() override { return false; }

  void ComputeAsync(OpKernelContext* context, DoneCallback done) override {
    OP_REQUIRES_OK_ASYNC(context, IsMpiIntialized(), done);
    const auto* input_tensor = &context->input(0);
    Tensor* output_tensor;
    OP_REQUIRES_OK_ASYNC(context, context->allocate_output(0, input_tensor->shape(), &output_tensor), done);

    LOG(INFO) << "running op: " << name();
    OpRecord record;
    record.name       = name();
    record.op_context = context;
    record.in_tensor  = input_tensor;
    record.out_tensor = output_tensor;
    record.on_gpu     = false;
    record.dtype      = message::DataType_TF_UNK;
    OP_REQUIRES_OK_ASYNC(context, TF_DataTypeToMessageDataType(input_tensor->dtype(), &record.dtype), done);

    const size_t temp_size = (input_tensor->NumElements() + mpi_size() - 1) / mpi_size();
    TensorShape temp_shape;
    temp_shape.AddDim(temp_size);
    OP_REQUIRES_OK_ASYNC(context, context->allocate_temp(input_tensor->dtype(), temp_shape, &record.temp_tensor), done);

    record.callback = [done, context](StatusOr<Tensor> status) {
      context->SetStatus(status.status());
      done();
    };

    auto allreduce_launch_callback = [record] { EnqueueTensorCollective(record, message::RequestType_ALLREDUCE); };

    allreduce_launch_callback();
  }
};

REGISTER_KERNEL_BUILDER(Name("MPIAllreduce").Device(DEVICE_CPU), MpiAllreduceOp<CPUDevice>);

REGISTER_OP("MPIAllreduce")
    .Attr("T: {int32, int64, float32}")
    .Input("tensor: T")
    .Output("sum: T")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    })
    .Doc(R"doc(
Perform an MPI Allreduce on a tensor.

Arguments
    tensor:   A tensor to reduce.

Output
    sum:      A tensor with the same shape as `tensor`, summed accross all MPI processes.
)doc");

}  // namespace collective
}  // namespace tips
