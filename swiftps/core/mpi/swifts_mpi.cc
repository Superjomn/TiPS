#include "swiftps/core/mpi/swifts_mpi.h"

#include "swiftps/core/common/common.h"

namespace swifts {

MpiContext &MpiContext::Global() {
  static MpiContext ctx;
  return ctx;
}

MpiContext::MpiContext() {
  LOG(INFO) << "Initalize global MPI Context";
  int rank;
  int size;
  CHECK_EQ(MPI_Comm_rank(mpi_comm(), &rank), 0);
  CHECK_EQ(MPI_Comm_size(mpi_comm(), &size), 0);

  ip_table_.resize(mpi_size(), "");
  ip_table_[rank] = GetLocalIp();
  LOG(INFO) << "local ip: " << ip_table_[rank];

  LOG(INFO) << "to broadcast ip_table ...";
  for (int i = 0; i < mpi_size(); i++) {
    mpi_broadcast(&ip_table_[i], 1, i);
  }
  mpi_barrier();
}

int mpi_rank() {
  int rank{-1};
  CHECK_EQ(MPI_Comm_rank(mpi_comm(), &rank), 0);
  return rank;
}

}  // namespace swifts