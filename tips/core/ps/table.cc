#include "tips/core/ps/table.h"

namespace tips {
namespace ps {

void Table::Initialize() {
  mpi_barrier();
  CHECK(!Initialized()) << "Duplicate initializing Table found";
  CHECK_GT(mpi_size(), 0);

  shards_.resize(shard_num_);
  local_shards_.resize(local_shard_num_);

  // Global shardid to local shardid
  for (int i = 0; i < shard_num_; i++) {
    shards_[i].shard_id       = i;
    shards_[i].local_shard_id = i / mpi_size();
  }

  // Local shard id to global shard id
  for (int i = 0; i < local_shard_num_; i++) {
    local_shards_[i].shard_id       = i * mpi_size() + mpi_rank();
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
    LOG(INFO) << "Local shard thread #" << tid << " start";
    std::function<void()> task;
    while (client_channel_->Read(&task)) {
      task();
    }
    LOG(INFO) << "Local shard thread #" << tid << " quit";
  });

  server_thread_group_.SetThreadNum(local_shard_num_);
  server_thread_group_.Start([this](int tid) {
    LOG(INFO) << "Server shard thread #" << tid << " start";
    auto channel = server_channels_[tid];
    std::function<void()> func;
    while (channel->Read(&func)) {
      func();
    }
    LOG(INFO) << "Server shard thread #" << tid << " quit";
  });

  mpi_barrier();
}

void Table::Finalize() {
  mpi_barrier();
  CHECK(Initialized());

  for (auto& channel : server_channels_) {
    channel->Close();
  }

  server_thread_group_.Join();
  client_channel_->Close();
  local_thread_group_.Join();

  mpi_barrier();
}

const Table::ShardInfo& Table::shard_info(int i) const {
  CHECK_LT(i, shard_num());
  return shards_[i];
}

}  // namespace ps
}  // namespace tips
