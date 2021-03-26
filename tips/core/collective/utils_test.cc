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

  Tensor first_rank(DataType::DT_INT32, TensorShape({1}));
  static_cast<int*>(first_rank.data())[0] = mpi_rank() + 1;

  Tensor first_ranks(DataType::DT_INT32, TensorShape({mpi_size()}));

  Tensor tensor(DataType::DT_FLOAT, TensorShape({mpi_rank() + 1, 4}));
  for (int i = 0; i < tensor.NumElements(); i++) {
    static_cast<float*>(tensor.data())[i] = mpi_rank() + 1;
  }

  LOG(INFO) << "first_rank: " << first_rank.DebugString();

  // allgather the size
  CHECK(AllgatherCpu<int32_t>(&first_rank, &first_ranks).ok());
  LOG(INFO) << "first_ranks: " << first_ranks.DebugString();

  int total_first_ranks = 0;
  for (int i = 0; i < mpi_size(); i++) {
    total_first_ranks += static_cast<int*>(first_ranks.data())[i];
  }

  Tensor output(DataType::DT_FLOAT, TensorShape({total_first_ranks, 4}));

  LOG(INFO) << "tensor: " << tensor.DebugString(10);

  // allgather the tensor
  CHECK(
      AllgathervCpu<float>(&tensor, absl::Span<int>(static_cast<int*>(first_ranks.data()), mpi_size()), &output).ok());

  LOG(INFO) << "output: " << output.DebugString(40);

  // Check the result.
  int stride     = 4;
  int pre_offset = 0;
  for (int i = 0; i < mpi_size(); i++) {
    for (int j = 0; j < stride * i + 1; j++) {
      CHECK_NEAR(static_cast<float*>(output.data())[pre_offset + j], i + 1, 1e-5);
    }
    pre_offset += stride * (i + 1);
  }

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

#ifdef TEST_ALLGATHERV
  tips::collective::TestAllgathervOp();
#endif

  return 0;
}
