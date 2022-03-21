#pragma once
#include <memory>
#include "tips/core/ps/route.h"
#include "tips/core/ps/sparse_table.h"

namespace tips {
namespace ps {

/**
 * SparseTableService: Service for a SparseTable.
 * It contains a Route and a SparseTable with some threads as service. It helps to create a distributed table service.
 */
class SparseTableService {
 public:
  SparseTableService(const std::shared_ptr<Route>& route,
                     absl::string_view table_name,
                     int num_nodes,
                     int num_local_shards)
      : table_(num_nodes, num_local_shards), route_(route) {
    table_.SetTableName(table_name);
  }

  void StartService();
  void StopService();

  //! Display some information about the table and route.
  std::string Summary() const;

  bool is_service_start() const { return is_service_start_; }
  bool is_service_stop() const { return !is_service_start(); }

  Route& route() { return *route_; }
  const Route& route() const { return *route_; }

  SparseTable& table() { return table_; }
  const SparseTable& table() const { return table_; }

 private:
  SparseTable table_;
  std::shared_ptr<Route> route_;
  absl::InlinedVector<std::shared_ptr<Channel<std::function<void()>>>, 4> server_channels_;
  std::shared_ptr<Channel<std::function<void()>>> client_channel_;
  bool is_service_start_{};

  // We allocate each shard a thread.
  ThreadGroup server_thread_group_;
  ThreadGroup local_thread_group_;
};

}  // namespace ps
}  // namespace tips
