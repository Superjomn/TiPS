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

  std::this_thread::sleep_for(1000ms);
  LOG(WARNING) << "Shutdown collective state";
  collective::CollectiveState::Global().Finalize();

  std::this_thread::sleep_for(1000ms);
  LOG(WARNING) << "Shutdown global RPC server";
  RpcServer::Global().Finalize();

  LOG(WARNING) << "Shutdown MPI";
  MpiContext::Global().Finalize();

  LOG(WARNING) << "TIPS service is shutdown";
}

bool tips_is_initialize() { return RpcServer::Global().initialized() && MpiContext::Global().IsInitialized(); }

int tips_size() { return mpi_size(); }

int tips_rank() { return mpi_rank(); }
}

}  // namespace tips