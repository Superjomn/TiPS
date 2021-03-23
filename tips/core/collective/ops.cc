#include "tips/core/collective/ops.h"
#include "tips/core/collective/coordinator.h"
#include "tips/core/mpi/tips_mpi.h"

namespace tips {
namespace collective {
using namespace tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;

template <typename Device>
class MpiSizeOp : public tensorflow::OpKernel {
 public:
  void Compute(OpKernelContext* context) override {
    CHECK(MpiContext::Global().IsInitialized()) << "MPI is not initialized";
    // Write integer to output tensor
    Tensor* output;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({}), &output));

    auto flat = output->flat<int>();
    flat(0)   = mpi_size();
  }
};
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
    CHECK(MpiContext::Global().IsInitialized());

    Tensor* output;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({}), &output));

    auto flat = output->flat<int>();
    flat(0)   = mpi_rank();
  }
};

REGISTER_KERNEL_BUILDER(Name("MPIRank").Device(DEVICE_GPU), MpiRankOp<CPUDevice>);

}  // namespace collective
}  // namespace tips
