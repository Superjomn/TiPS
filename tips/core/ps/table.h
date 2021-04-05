#pragma once
#include <gflags/gflags.h>

#include "tips/core/common/channel.h"
#include "tips/core/common/common.h"
#include "tips/core/common/thread_group.h"
#include "tips/core/mpi/tips_mpi.h"

#define TABLE_SHARD_NUM 8

namespace tips {
namespace ps {

/**
 * Table defines some basic meta for all the Table implementations.
 */
class Table {
 public:
  struct ShardInfo {
    int shard_id{-1};
    int local_shard_id{-1};
  };

  explicit Table() {
    // TODO(Superjomn) make it a config.
    local_shard_num_ = TABLE_SHARD_NUM;
  }

  //! Get number of shards locate in local.
  int local_shard_num() const { return local_shard_num_; }
  //! Get number of overall shards across the whole world for this application.
  int shard_num() const {
    // TODO(Superjomn) Replace mpi_size() to server node number.
    return mpi_size() * local_shard_num();
  }

  /**
   * Initialize a Table.
   */
  void Initialize();

  /**
   * Finalize a Table.
   */
  void Finalize();

  bool Initialized() const { return !shards_.empty(); }

  /**
   * Get the shard information for \param i -th global shard.
   */
  const ShardInfo& shard_info(int i) const {
    CHECK_LT(i, shard_num());
    return shards_[i];
  }

  /**
   * Get the shard information for the \param i -th local shard.
   */
  const ShardInfo& local_shard_info(int i) const {
    CHECK_LT(i, local_shards_.size());
    return local_shards_[i];
  }

  Channel<std::function<void()>>& server_channel(int i) { return *server_channels_[i]; }

  Channel<std::function<void()>>& client_channel() { return *client_channel_; }

 private:
  void set_local_shrard_num(int x) {
    CHECK_GE(x, 1);
    local_shard_num_ = x;
  }

  int local_shard_num_{};
  int shard_num_{};

  std::vector<ShardInfo> local_shards_;
  std::vector<ShardInfo> shards_;

  // We allocate each shard a thread.
  ThreadGroup server_thread_group_;
  ThreadGroup local_thread_group_;

  std::vector<std::shared_ptr<Channel<std::function<void()>>>> server_channels_;
  std::shared_ptr<Channel<std::function<void()>>> client_channel_;
};

}  // namespace ps
}  // namespace tips
