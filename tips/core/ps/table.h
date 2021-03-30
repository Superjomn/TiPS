#pragma once
#include "tips/core/common/channel.h"
#include "tips/core/common/common.h"
#include "tips/core/common/thread_group.h"
#include "tips/core/mpi/tips_mpi.h"

namespace tips {
namespace ps {

class Table {
 public:
  struct ShardInfo {
    int shard_id{-1};
    int local_shard_id{-1};
  };

  Table() = default;

  int local_shard_num() const { return local_shard_num_; }
  int shard_num() const { return shard_num_; }

  /**
   * Initialize a Table.
   */
  void Initialize();

  /**
   * Finalize a Table.
   */
  void Finalize();

  bool Initialized() const { return !shards_.empty(); }

  const ShardInfo& shard(int i) const { return shards_[i]; }

  const ShardInfo& local_shard(int i) const { return local_shards_[i]; }

  Channel<std::function<void()>>& server_channel(int i) { return *server_channels_[i]; }

  Channel<std::function<void()>>& client_channel() { return *client_channel_; }

  static Table& Global();

 private:
  int local_shard_num_{};
  int shard_num_{};

  std::vector<ShardInfo> local_shards_;
  std::vector<ShardInfo> shards_;

  ThreadGroup server_thread_group_;
  ThreadGroup local_thread_group_;

  std::vector<std::shared_ptr<Channel<std::function<void()>>>> server_channels_;
  std::shared_ptr<Channel<std::function<void()>>> client_channel_;
};

}  // namespace ps
}  // namespace tips
