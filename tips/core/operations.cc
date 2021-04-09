#include "tips/core/operations.h"

#include <chrono>
#include "tips/core/collective/coordinator.h"
#include "tips/core/common/naive_rpc.h"
#include "tips/core/mpi/tips_mpi.h"

namespace tips {
using namespace std::chrono_literals;

extern "C" {
void tips_init() {
  LOG(WARNING) << "Initialize TiPS service";
  LOG(INFO) << "Initialize MPI ...";
  MpiContext::Global();
  LOG(INFO) << "Initialize RPC ...";
  RpcServer::Global().Initialize();
  LOG(INFO) << "Initialize CollectiveState";
  collective::CollectiveState::Global().Initialize();

  LOG(WARNING) << "Initialize TiPS done";
}

void tips_shutdown() {
  using namespace std::chrono_literals;
  MPI_WARN << "to run tips_shutdown";

  mpi_barrier();
  MPI_WARN << "Shuting down collective state";
  collective::CollectiveState::Global().Finalize();
  MPI_WARN << "DONE Shuting down collective state";

  mpi_barrier();
  MPI_WARN << "Shuting down global RPC server";
  RpcServer::Global().Finalize();
  MPI_WARN << "DONE Shuting down global RPC server";

  mpi_barrier();
  MPI_WARN << "Shuting down MPI";
  MpiContext::Global().Finalize();
  MPI_WARN << "DONE Shuting down MPI";

  MPI_WARN << "tips is shutdown";
}

bool tips_is_initialize() { return RpcServer::Global().initialized() && MpiContext::Global().IsInitialized(); }

int tips_size() { return mpi_size(); }

int tips_rank() { return mpi_rank(); }
}

}  // namespace tips
