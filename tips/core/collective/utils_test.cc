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

void TestAllgatherOp() {
  MPI_Init(nullptr, nullptr);

  // int first_rank = mpi_rank() + 1;

  TensorShape shape({2, 3});
  Tensor tensor(DataType::DT_FLOAT, shape);
  for (int i = 0; i < tensor.NumElements(); i++) {
    static_cast<float*>(tensor.data())[i] = mpi_rank();
  }

  /*
  std::vector<int64_t> sizes(mpi_size());
  sizes[mpi_rank()] = first_rank;

  ZCHECK(MPI_Allgather(&sizes[mpi_rank()], 1, MPI_INT64_T, &sizes[0], 1, MPI_INT64_T, mpi_comm()));

  int total = 0;
  for (int v : sizes) {
    LOG(INFO) << "v: " << v;
    total += v;
  }

  LOG(INFO) << "total size: " << total;
   */

  Tensor output(DataType::DT_FLOAT, TensorShape({2 * mpi_size(), 3}));
  CHECK(collective::AllgatherCpu<float>(&tensor, &output).ok());

  if (mpi_rank() == 0) {
    LOG(INFO) << output.DebugString(100);
  }

  mpi_barrier();

  MPI_Finalize();
}

}  // namespace collective
}  // namespace tips

int main() {
#ifdef TEST_ALLREDUCE
  tips::collective::TestAllreduceOp();
#endif

#ifdef TEST_ALLGATHER
  tips::collective::TestAllgatherOp();
#endif

  return 0;
}
