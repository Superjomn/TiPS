#include "tips/core/operations.h"

#include "tips/core/collective/coordinator.h"
#include "tips/core/common/naive_rpc.h"
#include "tips/core/mpi/tips_mpi.h"

namespace tips {

extern "C" {
void tips_init() {
  LOG(WARNING) << "Initialize TiPS service";
  LOG(INFO) << "Initialize MPI ...";
  MpiContext::Global();
  LOG(INFO) << "Initialize RPC ...";
  RpcServer::Global().Initialize();
  LOG(WARNING) << "Initialize TiPS done";

  collective::CollectiveState::Global().Initialize();
}

void tips_shutdown() {
  using namespace std::chrono_literals;
  MPI_WARN << "to run tips_shutdown";

  mpi_barrier();
  MPI_WARN << "Shutdown collective state";
  collective::CollectiveState::Global().Finalize();

  mpi_barrier();
  MPI_WARN << "Shutdown global RPC server";
  RpcServer::Global().Finalize();

  mpi_barrier();
  MPI_WARN << "Shutdown MPI";
  MpiContext::Global().Finalize();
}

bool tips_is_initialize() { return RpcServer::Global().initialized() && MpiContext::Global().IsInitialized(); }

int tips_size() { return mpi_size(); }

int tips_rank() { return mpi_rank(); }
}

}  // namespace tips