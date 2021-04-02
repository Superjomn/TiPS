#pragma once
#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>
#include <absl/container/inlined_vector.h>
#include <mpi.h>

#include <unordered_set>
#include <vector>

#include "tips/core/common/common.h"
#include "tips/core/common/rwlock.h"
#include "tips/core/mpi/tips_mpi.h"

namespace tips {
namespace ps {

/**
 * ServerWorkerRoute records mark the mpi_ranks as (parameter)-server or worker.
 */
class ServerWorkerRoute {
 public:
  void RegisterServerNode(int node) { server_ids_.insert(node); }
  void RegisterWorkerNode(int node) { worker_ids_.insert(node); }

  size_t num_servers() const { return server_ids_.size(); }
  size_t num_workers() const { return worker_ids_.size(); }

  std::unordered_set<int> server_ids() const { return server_ids_; }
  std::unordered_set<int> worker_ids() const { return worker_ids_; }

  static ServerWorkerRoute& Global() {
    static ServerWorkerRoute x;
    return x;
  }

 private:
  std::unordered_set<int> server_ids_;
  std::unordered_set<int> worker_ids_;
};

}  // namespace ps
}  // namespace tips
