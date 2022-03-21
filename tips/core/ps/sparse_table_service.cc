#include "tips/core/ps/sparse_table_service.h"
namespace tips {
namespace ps {

void SparseTableService::StartService() {
  route().Initialize();
  auto server_group = route().GetGroup(Route::NodeKind::PS_SERVER);
  // The current rank is not contained in the server mpi group, that means ME is not a server, ignore all the following.
  if (!server_group.IsMeContained()) return;

  server_group.Barrier();
  CHECK(!is_service_start()) << "Duplicate initializing Table found";
  CHECK_GT(server_group.mpi_size(), 0);
  MPI_LOG << "mpi_size: " << server_group.mpi_size();
  MPI_LOG << "local_shard_num: " << table().shard_num();

  CHECK_EQ(table().shard_num(), table().local_shard_num() * server_group.mpi_size())
      << "Total count of shards should equal to MPI server nodes X local_shard_num";

  // Global shardid to local shardid
  for (int i = 0; i < table().shard_num(); i++) {
    table().shard_info(i).shard_id       = i;
    table().shard_info(i).local_shard_id = (i / server_group.mpi_size());
  }

  // Local shard id to global shard id
  for (int i = 0; i < table().local_shard_num(); i++) {
    table().local_shard_info(i).shard_id       = i * server_group.mpi_size() + server_group.mpi_rank();
    table().local_shard_info(i).local_shard_id = i;
  }

  server_channels_.resize(table().local_shard_num());
  for (int i = 0; i < table().local_shard_num(); i++) {
    server_channels_[i] = MakeChannel<std::function<void()>>();
  }

  client_channel_ = MakeChannel<std::function<void()>>();
  // TODO(Superjomn) Make this shared by all the threads in the same process?
  local_thread_group_.SetThreadNum(table().local_shard_num());
  local_thread_group_.Start([this](int tid) {
    MPI_LOG << "Local shard thread #" << tid << " start";
    std::function<void()> task;
    while (client_channel_->Read(&task)) {
      task();
    }
    MPI_LOG << "Local shard thread #" << tid << " quit";
  });

  server_thread_group_.SetThreadNum(table().local_shard_num());
  server_thread_group_.Start([this](int tid) {
    MPI_LOG << "Server shard thread #" << tid << " start";
    auto channel = server_channels_[tid];
    std::function<void()> func;
    while (channel->Read(&func)) {
      func();
    }
    MPI_LOG << "Server shard thread #" << tid << " quit";
  });

  server_group.Barrier();

  is_service_start_ = true;
}

void SparseTableService::StopService() {
  // This node is not a server, ignore.
  if (!route().GetGroup(Route::NodeKind::PS_SERVER).IsMeContained()) return;

  CHECK(is_service_start());
  is_service_start_ = false;

  for (auto& channel : server_channels_) {
    channel->Close();
  }
  server_thread_group_.Join();

  client_channel_->Close();
  local_thread_group_.Join();
}

std::string SparseTableService::Summary() const { return table().Summary(); }

}  // namespace ps
}  // namespace tips