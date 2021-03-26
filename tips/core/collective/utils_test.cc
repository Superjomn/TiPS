#include "tips/core/collective/utils.h"

#include <absl/types/span.h>
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

  TensorShape shape({2, 3});
  Tensor tensor(DataType::DT_FLOAT, shape);
  for (int i = 0; i < tensor.NumElements(); i++) {
    static_cast<float*>(tensor.data())[i] = mpi_rank();
  }

  Tensor output(DataType::DT_FLOAT, TensorShape({2 * mpi_size(), 3}));
  CHECK(collective::AllgatherCpu<float>(&tensor, &output).ok());

  if (mpi_rank() == 0) {
    LOG(INFO) << output.DebugString(100);
  }

  // Check result.
  auto* output_data = static_cast<float*>(output.data());
  for (int i = 0; i < mpi_size(); i++) {
    auto* slice_data = &output_data[i * tensor.NumElements()];
    for (int j = 0; j < tensor.NumElements(); j++) {
      CHECK_NEAR(slice_data[j], i, 1e-5);
    }
  }

  MPI_Finalize();
}

void TestAllgathervOp() {
  MPI_Init(nullptr, nullptr);

  Tensor tensor(DataType::DT_FLOAT, TensorShape({2, 4}));

  Tensor first_rank{mpi_rank() + 1};

  Tensor output(DataType::DT_FLOAT, TensorShape({*first_rank.scalar<int32_t>().data(), 4}));

  Tensor first_ranks(DataType::DT_INT32, TensorShape({mpi_size()}));

  // allgather the size
  CHECK(AllgatherCpu<int32_t>(&first_rank, &first_ranks).ok());

  // allgather the tensor
  // TODO

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
