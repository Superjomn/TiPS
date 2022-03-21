#pragma once
#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>
#include <absl/container/inlined_vector.h>
#include <mpi.h>

#include <mutex>
#include <unordered_set>
#include <vector>

#include "tips/core/common/common.h"
#include "tips/core/common/logging.h"
#include "tips/core/common/rwlock.h"
#include "tips/core/mpi/mpi_group.h"
#include "tips/core/mpi/tips_mpi.h"

namespace tips {
namespace ps {

/**
 * Route records mark the mpi_ranks as (parameter)-server or worker for a single SparseTable.
 */
class Route {
 public:
  // Extend this to hold more node kind.
  enum class NodeKind : short {
    PS_WORKER = 0,
    PS_SERVER,
    __NUM__  // this should be placed in the last
  };

  void RegisterNode(NodeKind kind, int node) { GetGroupIds(kind).insert(node); }

  const MpiGroup& GetGroup(NodeKind kind) const {
    CheckKindValid(kind);
    return groups_[static_cast<short>(kind)];
  }

  size_t GetGroupSize(NodeKind kind) const { return GetGroupIds(kind).size(); }

  /**
   * Tell whether an id is contained in a group.
   */
  bool IsInGroup(NodeKind kind, int id) const {
    CheckKindValid(kind);
    return GetGroupIds(kind).count(id);
  }

  void Initialize();

  void Finalize();

  static constexpr size_t NumGroups() { return static_cast<int>(NodeKind::__NUM__); }

  static Route& Global() {
    static Route x;
    return x;
  }

 private:
  inline std::unordered_set<int>& GetData(NodeKind kind) {
    CheckKindValid(kind);
    return data_[static_cast<int>(kind)];
  }

  inline const std::unordered_set<int>& GetGroupIds(NodeKind kind) const {
    CheckKindValid(kind);
    return data_[static_cast<int>(kind)];
  }
  inline std::unordered_set<int>& GetGroupIds(NodeKind kind) {
    CheckKindValid(kind);
    return data_[static_cast<int>(kind)];
  }

  inline static void CheckKindValid(NodeKind kind) { CHECK(kind != NodeKind::__NUM__) << "Invalid kind"; }

 private:
  absl::InlinedVector<std::unordered_set<int>, 2> data_{static_cast<int>(NodeKind::__NUM__)};
  absl::InlinedVector<MpiGroup, 2> groups_{Route::NumGroups()};
  absl::InlinedVector<absl::InlinedVector<int, 8>, 2> node_orders_{Route::NumGroups()};
  bool initialized_{};
  bool finalized_{};
};

}  // namespace ps
}  // namespace tips
