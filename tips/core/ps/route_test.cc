#include "tips/core/ps/route.h"
#include <gtest/gtest.h>
#include "tips/core/common/logging.h"
#include "tips/core/mpi/tips_mpi.h"
#include "tips/core/operations.h"

namespace tips {
namespace ps {

void TestBasic() {
  CHECK_EQ(mpi_size(), 4UL);
  Route route;
  route.RegisterNode(Route::NodeKind::PS_WORKER, 0);
  route.RegisterNode(Route::NodeKind::PS_WORKER, 1);
  route.RegisterNode(Route::NodeKind::PS_SERVER, 2);
  route.RegisterNode(Route::NodeKind::PS_SERVER, 3);
  route.Initialize();
  route.Finalize();
}

}  // namespace ps
}  // namespace tips

int main() {
  tips::tips_init();

  tips::ps::TestBasic();

  tips::tips_shutdown();
}
