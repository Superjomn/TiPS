#include "tips/core/mpi/tips_mpi.h"

#include <absl/strings/str_format.h>

#include "tips/core/common/common.h"

namespace tips {

MpiContext &MpiContext::Global() {
  static MpiContext ctx;
  return ctx;
}

MpiContext::MpiContext() {
  if (!IsInitialized()) {
    Initialize();
  }

  ZCHECK(MPI_Comm_rank(mpi_comm(), &rank_));
  ZCHECK(MPI_Comm_size(mpi_comm(), &size_));

  ip_table_.resize(size_, "");
  ip_table_[rank_] = GetLocalIp();

  for (int i = 0; i < size_; i++) {
    mpi_broadcast(&ip_table_[i], 1, i, rank_);
  }
  mpi_barrier(mpi_comm(), size_);
}

bool MpiContext::IsInitialized() {
  int flag;
  ZCHECK(MPI_Initialized(&flag));
  return flag;
}

bool MpiContext::IsFinalized() {
  int flag;
  ZCHECK(MPI_Finalized(&flag));
  return flag;
}

void MpiContext::Initialize(int *argc, char ***argv) { ZCHECK(MPI_Init(argc, argv)) << "MPI init failed"; }

int mpi_rank() { return MpiContext::Global().rank(); }

std::string mpi_rank_repr() { return absl::StrFormat("#rank-[%d/%d]", mpi_rank(), mpi_size()); }

void mpi_barrier(MPI_Comm comm) {
  int size;
  ZCHECK(MPI_Comm_size(comm, &size));
  mpi_barrier(comm, size);
}

void mpi_barrier(MPI_Comm comm, int size) {
  // MPI_Barrier uses busy waiting. Try to avoid.
  // MPI_Barrier(comm);

  std::vector<MPI_Request> reqs(size, MPI_REQUEST_NULL);
  int dummy = 0;

  for (int i = 0; i < size; i++) {
    MPI_Irecv(&dummy, 1, MPI_INT, i, 0, comm, &reqs[i]);
  }

  for (int i = 0; i < size; i++) {
    MPI_Send(&dummy, 1, MPI_INT, i, 0, comm);
  }

  for (int i = 0; i < size; i++) {
    for (unsigned long x = 1;; x = std::min(x * 2, 2000UL)) {
      int flag = 0;
      MPI_Test(&reqs[i], &flag, MPI_STATUSES_IGNORE);

      if (flag) {
        break;
      }

      usleep(x);
    }
  }
}

}  // namespace tips
