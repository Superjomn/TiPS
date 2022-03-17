#pragma once
#include <absl/container/flat_hash_map.h>
#include <absl/container/inlined_vector.h>
#include <absl/hash/hash.h>
#include <absl/types/any.h>

#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <unordered_map>

#include "tips/core/common/common.h"
#include "tips/core/common/flatbuffers_utils.h"
#include "tips/core/common/rwlock.h"
#include "tips/core/message/ps_messages_generated.h"
#include "tips/core/ps/sparse_table.h"
#include "tips/core/ps/table.h"

namespace tips {
namespace ps {

/**
 * SparseTableService: SparseTable with some threads as a service.
 */
class SparseTableService : public SparseTable {
 public:
  using pull_done_handle_t = std::function<void(FBS_TypeBufferOwned<ps::message::PullResponse> &&)>;
  using push_done_handle_t = std::function<void>;

  SparseTableService(int num_nodes, int num_local_shards, int num_threads) : SparseTable(num_nodes, num_local_shards) {
    threads_.resize(num_threads);
  }

  ~SparseTableService() {
    LOG(WARNING) << "service thread to terminate ...";
    for (auto &t : threads_) {
      t.Terminate();
    }

    LOG(WARNING) << "service thread to join ...";
    for (auto &t : threads_) {
      t.Join();
    }
  }

  void Pull(const ps::message::PullRequest &msg, pull_done_handle_t handle) {
    auto all_shard_keys = std::make_shared<absl::InlinedVector<std::vector<uint64_t>, 4>>();
    all_shard_keys->resize(shard_num());

    for (const auto &key : *msg.keys()) {
      const auto shard_id = ToShardId(key);
      (*all_shard_keys)[shard_id].push_back(key);
    }

    for (int shard_id = 0; shard_id < shard_num(); ++shard_id) {
      // The shard_keys is shared here.
      server_channel(shard_id).Write([this, all_shard_keys, shard_id] {
        // Here we need a agent to access the shard?
        auto &cur_shard_keys = (*all_shard_keys)[shard_id];
        for (auto key : cur_shard_keys) {
          float *val;
          local_shard(shard_id).Get(key, val);
        }
      });
    }
  }

  void Push(const ps::message::PushRequest &msg, push_done_handle_t handle);

 private:
  absl::InlinedVector<ManagedThread, 2> threads_;
};

}  // namespace ps
}  // namespace tips
