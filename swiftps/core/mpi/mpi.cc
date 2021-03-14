#include "swiftps/core/mpi/mpi.h"

#include "swiftps/core/common/common.h"

namespace swifts {

MpiCtx &MpiCtx::Global() {
  static MpiCtx ctx;
  return ctx;
}

MpiCtx::MpiCtx() {
  int rank;
  int size;
  CHECK_EQ(MPI_Comm_rank(mpi_comm(), &rank), 0);
  CHECK_EQ(MPI_Comm_size(mpi_comm(), &size), 0);

  ip_table_.resize(mpi_size(), "");
  ip_table_[rank] = GetLocalIp();
}

}  // namespace swifts