#pragma once
#include <absl/types/optional.h>
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

  /**
   * @param num_nodes Number of the MPI nodes.
   * @param local_shard_num Number of shards inside a node.
   */
  explicit Table(int num_nodes, int local_shard_num)
      : local_shard_num_(local_shard_num), shard_num_(local_shard_num * num_nodes) {}

  //! Get number of shards locate in local.
  int local_shard_num() const { return local_shard_num_; }
  //! Get number of overall shards across the whole world for this application.
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

  /**
   * Get the shard information for \param i -th global shard.
   */
  const ShardInfo& shard_info(int i) const;

  /**
   * Get the shard information for the \param i -th local shard.
   */
  const ShardInfo& local_shard_info(int i) const {
    CHECK_LT(i, local_shards_.size());
    return local_shards_[i];
  }

  //! Set the optional table name for better debugging.
  void SetTableName(absl::string_view name) { table_name_ = std::string(name); }

  //! Get the table name, get an empty string_view if table_name is not set.
  absl::string_view GetTableName() const {
    if (table_name_.has_value()) return absl::string_view(table_name_->data(), table_name_->size());
    return absl::string_view();
  }

  Channel<std::function<void()>>& server_channel(int i) { return *server_channels_[i]; }

  Channel<std::function<void()>>& client_channel() { return *client_channel_; }

 private:
  const int local_shard_num_;
  const int shard_num_;

  std::vector<ShardInfo> local_shards_;
  std::vector<ShardInfo> shards_;

  ThreadGroup server_thread_group_;
  ThreadGroup local_thread_group_;

  std::vector<std::shared_ptr<Channel<std::function<void()>>>> server_channels_;
  std::shared_ptr<Channel<std::function<void()>>> client_channel_;
  absl::optional<std::string> table_name_;
};

}  // namespace ps
}  // namespace tips
