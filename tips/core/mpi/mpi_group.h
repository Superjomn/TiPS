#pragma once

#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>
#include <absl/container/inlined_vector.h>

#include "tips/core/common/common.h"
#include "tips/core/mpi/tips_mpi.h"

namespace tips {

/**
 * A route for a group.
 */
class MpiGroup {
 public:
  /**
   * Add a rank to the route.
   * @param rank A MPI rank.
   * @returns true if not duplicate or else.
   */
  bool AddRank(int rank);

  void AddRanks(absl::Span<int>&& ranks);

  template <typename Iterator>
  void AddRanks(Iterator begin, Iterator end) {
    for (auto it = begin; it != end; it++) {
      AddRank(*it);
    }
  }

  void Initialize();

  void Finalize() {
    if (valid()) {
      MPI_Comm_free(&mpi_comm_);
      mpi_comm_ = MPI_COMM_NULL;
    }

    if (world_group_ != MPI_GROUP_NULL) MPI_Group_free(&world_group_);
    if (my_group_ != MPI_GROUP_NULL) MPI_Group_free(&my_group_);

    world_group_ = MPI_GROUP_NULL;
    my_group_    = MPI_GROUP_NULL;
  }

  bool valid() const { return initialized_ && mpi_comm_ != MPI_COMM_NULL; }

  bool empty() const {
    CHECK(initialized_);
    return rank_order_.empty();
  }

  MPI_Comm mpi_comm() const {
    CHECK(valid());
    return mpi_comm_;
  }

  int mpi_size() const { return data_.size(); }

  int mpi_rank() const {
    CHECK(valid());
    return mpi_rank_;
  }

  void Barrier() const { ::tips::mpi_barrier(this->mpi_comm()); }

  void AllReduce(void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op);

  /**
   * Get the rank in MPI_COMM_WORLD
   * @param group_rank The rank in the group.
   */
  int ToWorldRank(int group_rank) const {
    CHECK_GE(group_rank, 0);
    CHECK_LT(group_rank, rank_order_.size());
    return rank_order_[group_rank];
  }

  int ToWorldRank() const { return ToWorldRank(this->mpi_rank()); }

  ~MpiGroup() { Finalize(); }

 private:
  bool initialized_{};
  absl::flat_hash_set<int> data_;
  std::vector<int> rank_order_;
  MPI_Comm mpi_comm_{MPI_COMM_NULL};

  MPI_Group world_group_{MPI_GROUP_NULL};
  MPI_Group my_group_{MPI_GROUP_NULL};

  int mpi_rank_{-1};
  int mpi_size_{-1};
};

}  // namespace tips
