#include "tips/core/ps/sparse_table_service.h"

#include <gtest/gtest.h>

#include "tips/core/operations.h"

namespace tips {
namespace ps {

void TestBasic() {
  CHECK_EQ(mpi_size(), 4UL);

  auto route = std::make_shared<Route>();
  route->RegisterNode(Route::NodeKind::PS_SERVER, 0);
  route->RegisterNode(Route::NodeKind::PS_SERVER, 1);
  route->RegisterNode(Route::NodeKind::PS_WORKER, 2);
  route->RegisterNode(Route::NodeKind::PS_WORKER, 3);
  route->Initialize();

  // only 3 server nodes
  SparseTableService service(route, "test0", 2, 3);
  CHECK(!service.is_service_start());

  service.StartService();

  auto& server_group = service.route().GetGroup(Route::NodeKind::PS_SERVER);
  if (mpi_rank() == 0 || mpi_rank() == 1) {
    CHECK(server_group.IsMeContained());
  }

  if (mpi_rank() == 2 || mpi_rank() == 3) {
    CHECK(!server_group.IsMeContained());
  }

  if (mpi_rank() == 0 || mpi_rank() == 1) {
    CHECK(service.is_service_start());
    LOG(INFO) << service.Summary();
  }
  service.StopService();
}

}  // namespace ps
}  // namespace tips

int main() {
  tips::tips_init();

  tips::ps::TestBasic();

  tips::tips_shutdown();
}
