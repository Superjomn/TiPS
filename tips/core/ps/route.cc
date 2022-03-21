#include "tips/core/ps/route.h"
#include <absl/strings/str_format.h>

namespace tips {
namespace ps {

void Route::Finalize() {
  CHECK(initialized_) << "Duplicated Route initialization found";
  CHECK(!finalized_) << "Duplicated Route finalization found";
  mpi_barrier();

  for (int kind = 0; kind < static_cast<int>(NodeKind::__NUM__); kind++) {
    auto& group_ids = GetGroupIds(static_cast<NodeKind>(kind));
    if (!group_ids.empty()) {
      groups_[kind].Finalize();
    }
  }

  finalized_ = true;
}

void Route::Initialize() {
  if (initialized_) return;

  mpi_barrier();

  for (int kind = 0; kind < static_cast<int>(NodeKind::__NUM__); kind++) {
    auto& group_ids = GetGroupIds(static_cast<NodeKind>(kind));
    if (!group_ids.empty()) {
      LOG(INFO) << absl::StrFormat("Initialize Group #%d with %d nodes", kind, group_ids.size());
      groups_[kind].AddRanks(group_ids.begin(), group_ids.end());
      groups_[kind].Initialize();
    }
  }

  mpi_barrier();
  initialized_ = true;
}

}  // namespace ps
}  // namespace tips
