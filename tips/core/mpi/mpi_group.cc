#include "tips/core/mpi/mpi_group.h"

namespace tips {

void MpiGroup::Initialize() {
  CHECK(!initialized_) << "Duplicate initialization found";
  initialized_ = true;
  rank_order_.assign(data_.begin(), data_.end());

  // create the target group
  MPI_Group world_group;
  MPI_Group my_group;
  ZCHECK(MPI_Comm_group(::tips::mpi_comm(), &world_group));
  ZCHECK(MPI_Group_incl(world_group, rank_order_.size(), rank_order_.data(), &my_group));

  // create the communicator
  ZCHECK(MPI_Comm_create_group(::tips::mpi_comm(), my_group, 0, &mpi_comm_));

  if (mpi_comm_ != MPI_COMM_NULL) {
    ZCHECK(MPI_Comm_rank(mpi_comm_, &mpi_rank_));
    ZCHECK(MPI_Comm_size(mpi_comm_, &mpi_size_));

    int rank = ::tips::mpi_rank();

    ZCHECK(MPI_Allgather(
        &rank, 1, mpi_type_trait<int>::type(), &rank_order_[0], 1, mpi_type_trait<int>::type(), this->mpi_comm_));

    CHECK_EQ(mpi_size_, data_.size());
  }
}

}  // namespace tips