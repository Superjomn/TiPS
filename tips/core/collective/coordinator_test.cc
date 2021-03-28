#include "tips/core/collective/coordinator.h"

#include "tips/core/operations.h"

namespace tips {
namespace collective {

using namespace tensorflow;

void TestAllreduce() {
  tips_init();

  TensorShape shape({2, 4});
  Tensor tensor(DataType::DT_FLOAT, shape);
  Tensor output(DataType::DT_FLOAT, shape);

  for (int i = 0; i < tensor.NumElements(); i++) {
    static_cast<float*>(tensor.data())[i] = i * 0.1;
    static_cast<float*>(output.data())[i] = 0.;
  }

  OpRecord record;
  record.name     = "a";
  record.callback = [&](StatusOr<tensorflow::Tensor> x) {
    CHECK(x.ok()) << "failed";
    LOG(INFO) << "output: " << x->DebugString(10);
    auto* tensor_data = static_cast<float*>(tensor.data());
    auto* output_data = static_cast<float*>(output.data());
    for (int i = 0; i < tensor.NumElements(); i++) {
      CHECK_NEAR(tensor_data[i] * mpi_size(), output_data[i], 1e-4);
    }
  };
  record.in_tensor  = &tensor;
  record.out_tensor = &output;
  record.dtype      = message::DataType_TF_FLOAT32;
  record.on_gpu     = false;
  record.rank       = mpi_rank();
  record.sizes_vec  = {2, 4};
  record.op_context = nullptr;

  EnqueueTensorCollective(record, message::RequestType_ALLREDUCE);

  mpi_barrier();
  tips_shutdown();
}

void TestAllgather() {
  tips_init();

  TensorShape shape({2, 4});
  Tensor tensor(DataType::DT_FLOAT, shape);

  for (int i = 0; i < tensor.NumElements(); i++) {
    static_cast<float*>(tensor.data())[i] = i * 0.1;
  }

  OpRecord record;
  record.name       = "a";
  record.callback   = [&](StatusOr<tensorflow::Tensor> x) {};
  record.in_tensor  = &tensor;
  record.dtype      = message::DataType_TF_FLOAT32;
  record.on_gpu     = false;
  record.rank       = mpi_rank();
  record.sizes_vec  = {2, 4};
  record.op_context = nullptr;

  EnqueueTensorCollective(record, message::RequestType_ALLGATHER);

  mpi_barrier();
  tips_shutdown();
}

}  // namespace collective
}  // namespace tips

int main() {
#ifdef TEST_ALLREDUCE
  tips::collective::TestAllreduce();
#endif

#ifdef TEST_ALLGATHER
  tips::collective::TestAllgather();
#endif

  return 0;
}