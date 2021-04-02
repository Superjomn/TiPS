#include "tips/core/mpi/mpi_group.h"

namespace tips {

void MpiGroup::Initialize() {
  CHECK(!initialized_) << "Duplicate initialization found";
  CHECK(!data_.empty());
  initialized_ = true;

  // Get the group of processes in MPI_COMM_WORLD
  MPI_Comm_group(MPI_COMM_WORLD, &world_group_);

  // The ranks of this group must sort, or MPI_Comm_create_group will hang.
  rank_order_.assign(data_.begin(), data_.end());
  std::sort(rank_order_.begin(), rank_order_.end());

  // Construct a group containing all of the prime ranks in world_group
  MPI_Group_incl(world_group_, rank_order_.size(), rank_order_.data(), &my_group_);

  // Create a new communicator based on the group
  MPI_Comm_create_group(MPI_COMM_WORLD, my_group_, 0, &mpi_comm_);

  // If this rank isn't in the new communicator, it will be
  // MPI_COMM_NULL. Using MPI_COMM_NULL for MPI_Comm_rank or
  // MPI_Comm_size is erroneous
  if (MPI_COMM_NULL != mpi_comm_) {
    MPI_Comm_rank(mpi_comm_, &mpi_rank_);
    MPI_Comm_size(mpi_comm_, &mpi_size_);
  }
}

void MpiGroup::AddRanks(absl::Span<int> &&ranks) {
  CHECK(!initialized_);
  for (int x : ranks) {
    data_.insert(x);
  }
}

bool MpiGroup::AddRank(int rank) {
  CHECK(!initialized_);
  auto res = data_.insert(rank);
  return res.second;
}

void MpiGroup::AllReduce(void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op) {
  ZCHECK(MPI_Allreduce(sendbuf, recvbuf, count, datatype, op, mpi_comm()));
}

}  // namespace tips
