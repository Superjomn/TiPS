#include "tips/core/collective/utils.h"
#include "tips/core/mpi/tips_mpi.h"

namespace tips {
namespace collective {

using namespace tensorflow;

void TestAllreduceOp() {
  MPI_Init(nullptr, nullptr);

  TensorShape shape({2, 2});
  Tensor tenspr;
  Tensor tensor(tensorflow::DataType::DT_FLOAT, shape);
  Tensor output(tensorflow::DataType::DT_FLOAT, shape);

  for (int i = 0; i < shape.num_elements(); i++) {
    static_cast<float*>(tensor.data())[i] = i * 0.1 * mpi_rank();
  }

  CHECK(collective::AllreduceCpu<float>(&tensor, &output, CollectiveOpKind::SUM).ok());

  if (mpi_rank() == 0) {
    for (int i = 0; i < shape.num_elements(); i++) {
      float x = static_cast<float*>(output.data())[i];
      LOG(INFO) << x;
      CHECK_NEAR(x, i * 0.1 * ((mpi_size() - 1) * mpi_size() / 2), 1e-5);
    }
  }

  mpi_barrier();

  MPI_Finalize();
}

}  // namespace collective
}  // namespace tips

int main() {
  tips::collective::TestAllreduceOp();

  return 0;
}
