#include "tips/core/ps/table.h"

namespace tips {
namespace ps {

void Table::StartService() {
  server_group().Barrier();
  CHECK(!is_service_start()) << "Duplicate initializing Table found";
  CHECK_GT(server_group().mpi_size(), 0);
  set_local_shard_num(TABLE_SHARD_NUM);
  MPI_LOG << "mpi_size: " << server_group().mpi_size();
  MPI_LOG << "local_shard_num: " << local_shard_num_;

  CHECK_GT(shard_num(), 0);
  shards_.resize(shard_num());
  local_shards_.resize(local_shard_num_);

  // Global shardid to local shardid
  for (int i = 0; i < shard_num(); i++) {
    shards_[i].shard_id       = i;
    shards_[i].local_shard_id = i / server_group().mpi_size();
  }

  // Local shard id to global shard id
  for (int i = 0; i < local_shard_num_; i++) {
    local_shards_[i].shard_id       = i * server_group().mpi_size() + server_group().mpi_rank();
    local_shards_[i].local_shard_id = i;
  }

  server_channels_.resize(local_shard_num_);
  for (int i = 0; i < local_shard_num_; i++) {
    server_channels_[i] = MakeChannel<std::function<void()>>();
  }

  client_channel_ = MakeChannel<std::function<void()>>();
  // TODO(Superjomn) Make this shared by all the threads in the same process?
  local_thread_group_.SetThreadNum(local_shard_num_);
  local_thread_group_.Start([this](int tid) {
    MPI_LOG << "Local shard thread #" << tid << " start";
    std::function<void()> task;
    while (client_channel_->Read(&task)) {
      task();
    }
    MPI_LOG << "Local shard thread #" << tid << " quit";
  });

  server_thread_group_.SetThreadNum(local_shard_num_);
  server_thread_group_.Start([this](int tid) {
    MPI_LOG << "Server shard thread #" << tid << " start";
    auto channel = server_channels_[tid];
    std::function<void()> func;
    while (channel->Read(&func)) {
      func();
    }
    MPI_LOG << "Server shard thread #" << tid << " quit";
  });

  server_group().Barrier();
}

void Table::StopService() {
  finalized_ = true;

  CHECK(is_service_start());

  for (auto& channel : server_channels_) {
    channel->Close();
  }
  server_thread_group_.Join();

  client_channel_->Close();
  local_thread_group_.Join();
}

}  // namespace ps
}  // namespace tips
