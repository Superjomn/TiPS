#include "tips/core/mpi/mpi_group.h"

#include "tips/core/operations.h"

namespace tips {

void MpiGroup_basic() {
  mpi_barrier();

  CHECK_GE(mpi_size(), 4);
  MpiGroup group;
  group.AddRank(3);
  group.AddRank(2);
  group.Initialize();

  if (group.valid()) {
    CHECK_EQ(mpi_rank(), group.ToWorldRank());
    LOG(INFO) << "my group rank: " << mpi_rank() << " -> " << group.mpi_rank();
  }

  mpi_barrier();

  group.Finalize();

  MPI_LOG << "after barrirer";
}

}  // namespace tips

int main() {
  tips::tips_init();

  tips::MpiGroup_basic();

  CHECK_EQ(tips::mpi_size(), 4);

  tips::tips_shutdown();
}
