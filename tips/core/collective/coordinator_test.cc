#include "tips/core/collective/coordinator.h"
#include "tips/core/operations.h"

namespace tips {
namespace collective {

using namespace tensorflow;

void Test() {
  tips_init();

  TensorShape shape({2, 4});
  Tensor tensor(DataType::DT_FLOAT, shape);
  Tensor output(DataType::DT_FLOAT, shape);

  OpRecord record;
  record.name       = "a";
  record.callback   = [](StatusOr<tensorflow::Tensor> x) {};
  record.in_tensor  = &tensor;
  record.out_tensor = &output;
  record.dtype      = message::DataType_TF_FLOAT32;
  record.on_gpu     = false;
  record.rank       = mpi_rank();
  record.sizes_vec  = {2, 4};
  record.op_context = nullptr;

  EnqueueTensorCollective(record, message::RequestType_ALLREDUCE);

  tips_shutdown();
}

}  // namespace collective
}  // namespace tips

int main() {
  tips::collective::Test();

  return 0;
}