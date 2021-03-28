#include <mpi.h>

#include <vector>

#include "tips/core/mpi/tips_mpi.h"

using namespace tips;  // NOLINT
void test() {
  float value = mpi_rank() * 0.1;

  std::vector<float> gathered_data(mpi_size(), 0);

  ZCHECK(MPI_Allgather(&value, 1, MPI_FLOAT, &gathered_data[0], 1, MPI_FLOAT, mpi_comm()));

  for (int i = 0; i < mpi_size(); i++) {
    CHECK_NEAR(gathered_data[i], i * 0.1, 1e-5);

    if (mpi_rank() == 0) {
      LOG(INFO) << i << " " << gathered_data[i];
    }
  }
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  test();

  MPI_Finalize();
  return 0;
}
