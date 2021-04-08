#include "tips/core/ps/ps_client.h"
#include "tips/core/operations.h"
#include "tips/core/ps/access_method.h"
#include "tips/core/ps/sparse_table.h"

namespace tips {
namespace ps {

void TestBasic() {
  PullRequest req;
  req.data.emplace_back(0, DatatypeTypetrait<float>(), 12);
  req.data.emplace_back(3, DatatypeTypetrait<float>(), 4);

  using val_t   = AnyVec;
  using key_t   = uint64_t;
  using param_t = AnyVec;
  using table_t = SparseTable<key_t, val_t>;

  auto& route = Route::Global();
  route.RegisterNode<Route::NodeKind::PS_SERVER>(0);
  route.RegisterNode<Route::NodeKind::PS_SERVER>(1);
  route.RegisterNode<Route::NodeKind::PS_SERVER>(2);

  PullCache cache;
  PsClient<table_t>::Global().Pull(req, &cache);
}

}  // namespace ps
}  // namespace tips

int main() {
  tips::tips_init();

  tips::tips_shutdown();
}
