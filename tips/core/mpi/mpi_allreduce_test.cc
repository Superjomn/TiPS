#include <glog/logging.h>
#include <mpi.h>

#include <random>
#include <vector>

void test() {
  std::vector<float> rands;

  const int n = 10;

  for (int i = 0; i < n; i++) {
    rands.push_back(i * 0.1);
  }

  std::vector<float> out(10);

  CHECK_EQ(MPI_Allreduce(&rands[0], &out[0], n, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD), 0);

  int size;
  CHECK_EQ(MPI_Comm_size(MPI_COMM_WORLD, &size), 0);
  int rank;
  CHECK_EQ(MPI_Comm_rank(MPI_COMM_WORLD, &rank), 0);

  for (int i = 0; i < n; i++) {
    CHECK_NEAR(out[i], i * 0.1 * size, 1e-5);

    if (rank == 0) {
      LOG(INFO) << out[i];
    }
  }
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  test();

  MPI_Finalize();
  return 0;
}
