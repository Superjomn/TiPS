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
    // CHECK_EQ(mpi_rank(), group.ToWorldRank());
    LOG(INFO) << "my group rank: " << mpi_rank() << " -> " << group.mpi_rank();
  }

  mpi_barrier();

  MPI_LOG << "after barrirer";
}

void MpiGroup_basic1() {
  // Get the rank and size in the original communicator
  int world_rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // Get the group of processes in MPI_COMM_WORLD
  MPI_Group world_group;
  MPI_Comm_group(MPI_COMM_WORLD, &world_group);

  int n              = 2;
  const int ranks[2] = {1, 3};

  // Construct a group containing all of the prime ranks in world_group
  MPI_Group prime_group;
  MPI_Group_incl(world_group, 2, ranks, &prime_group);

  // Create a new communicator based on the group
  MPI_Comm prime_comm;
  MPI_Comm_create_group(MPI_COMM_WORLD, prime_group, 0, &prime_comm);

  int prime_rank = -1, prime_size = -1;
  // If this rank isn't in the new communicator, it will be
  // MPI_COMM_NULL. Using MPI_COMM_NULL for MPI_Comm_rank or
  // MPI_Comm_size is erroneous
  if (MPI_COMM_NULL != prime_comm) {
    MPI_Comm_rank(prime_comm, &prime_rank);
    MPI_Comm_size(prime_comm, &prime_size);
  }

  printf("WORLD RANK/SIZE: %d/%d \t PRIME RANK/SIZE: %d/%d\n", world_rank, world_size, prime_rank, prime_size);

  if (prime_comm != MPI_COMM_NULL) {
    MPI_Group_free(&world_group);
    MPI_Group_free(&prime_group);
    MPI_Comm_free(&prime_comm);
  }
}

}  // namespace tips

int main() {
  tips::tips_init();

  tips::MpiGroup_basic();

  CHECK_EQ(tips::mpi_size(), 4);

  tips::tips_shutdown();
}
