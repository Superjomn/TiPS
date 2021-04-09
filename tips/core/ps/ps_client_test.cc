#include "tips/core/ps/ps_client.h"
#include "tips/core/operations.h"
#include "tips/core/ps/access_method.h"
#include "tips/core/ps/ps_server.h"
#include "tips/core/ps/sparse_access_method.h"
#include "tips/core/ps/sparse_table.h"

namespace tips {
namespace ps {

void TestBasic() {
  PullRequest req;
  req.data.emplace_back(0, DatatypeTypetrait<float>(), 1);
  req.data.emplace_back(3, DatatypeTypetrait<float>(), 4);
  req.data.emplace_back(200, DatatypeTypetrait<float>(), 13);

  using val_t         = AnyVec;
  using key_t         = uint64_t;
  using param_t       = AnyVec;
  using table_t       = SparseTable<key_t, val_t>;
  using pull_access_t = SparseTablePullAccess<key_t, param_t, val_t>;
  using push_access_t = SparseTableSgdPushAccess<key_t, param_t, param_t>;
  using server_t      = PsServer<table_t, pull_access_t, push_access_t>;

  auto& route = Route::Global();
  route.RegisterNode<Route::NodeKind::PS_SERVER>(0);
  // route.RegisterNode<Route::NodeKind::PS_SERVER>(1);
  // route.RegisterNode<Route::NodeKind::PS_SERVER>(2);
  route.Initialize();

  std::unique_ptr<table_t> table;
  std::unique_ptr<server_t> server;

  pull_access_t pull_access(table.get());
  push_access_t push_access(table.get(), 0.01);
  table.reset(new table_t(route.GetGroup<Route::NodeKind::PS_SERVER>()));
  server.reset(new server_t(pull_access, push_access, table.get()));

  // Create PsServer instance only for the server nodes.
  CHECK(route.IsInGroup<Route::NodeKind::PS_SERVER>(0));
  if (route.IsInGroup<Route::NodeKind::PS_SERVER>(mpi_rank())) {
    MPI_LOG << "Initialize PsServer services";
    table->StartService();
    MPI_LOG << "After initialize PsServer";
    server->StartService();
  }

  mpi_barrier();

  PullCache cache;
  std::condition_variable cv;
  std::mutex mu;

  PsClient<table_t> client(route, table.get());
  client.Pull(req, &cache, [&] { cv.notify_one(); });

  std::unique_lock<std::mutex> lock(mu);
  cv.wait(lock);

  auto display_cache = [&] {
    for (auto& item : cache.data) {
      LOG(INFO) << "pull: " << item.first << " " << item.second.size() << ":\t" << item.second.ToVec<float>();
    }
  };

  auto check_cache = [&] {
    CHECK_EQ(cache.data[0].size(), 1);
    CHECK_EQ(cache.data[3].size(), 4);
    CHECK_EQ(cache.data[200].size(), 13);
  };

  display_cache();
  check_cache();

  mpi_barrier();

  if (route.IsInGroup<Route::NodeKind::PS_SERVER>(mpi_rank())) {
    table->StopService();
    server->StopService();
  }

  route.Finalize();

  mpi_barrier();
}

}  // namespace ps
}  // namespace tips

int main() {
  tips::tips_init();

  tips::ps::TestBasic();

  tips::tips_shutdown();
}
