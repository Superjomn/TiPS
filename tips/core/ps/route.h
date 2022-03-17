#pragma once
#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>
#include <absl/container/inlined_vector.h>
#include <mpi.h>

#include <mutex>
#include <unordered_set>
#include <vector>

#include "tips/core/common/common.h"
#include "tips/core/common/rwlock.h"
#include "tips/core/mpi/mpi_group.h"
#include "tips/core/mpi/tips_mpi.h"

namespace tips {
namespace ps {

/**
 * Route records mark the mpi_ranks as (parameter)-server or worker.
 */
class Route {
 public:
  // Extend this to hold more node kind.
  enum class NodeKind : short {
    PS_WORKER = 0,
    PS_SERVER = 1,
    __NUM__  // this should be placed in the last
  };

#define ROUTE_NODE_KIND_FOREACH(expr__) expr__(PS_WORKER) expr__(PS_SERVER)

  template <NodeKind kind>
  void RegisterNode(int node) {
    GetGroupIds<kind>().insert(node);
  }

  template <NodeKind kind>
  const MpiGroup& GetGroup() const {
    CheckKindValid<kind>();
    return groups_[static_cast<short>(kind)];
  }

  template <NodeKind kind>
  size_t GetGroupSize() const {
    return GetGroupIds<kind>().size();
  }

  /**
   * Tell whether an id is contained in a group.
   */
  template <NodeKind kind>
  bool IsInGroup(int id) const {
    CheckKindValid<kind>();
    return GetGroupIds<kind>().count(id);
  }

  void Initialize() {
    CHECK(!initialized_) << "Duplicated Route initialization found";
    mpi_barrier();

#define ___(item__)                                                                             \
  {                                                                                             \
    auto& group_ids = GetGroupIds<NodeKind::item__>();                                          \
    if (!group_ids.empty()) {                                                                   \
      groups_[static_cast<int>(NodeKind::item__)].AddRanks(group_ids.begin(), group_ids.end()); \
      groups_[static_cast<int>(NodeKind::item__)].Initialize();                                 \
    }                                                                                           \
  }

    ROUTE_NODE_KIND_FOREACH(___)

#undef ___

    initialized_ = true;
  }

  void Finalize() {
    CHECK(initialized_) << "Duplicated Route initialization found";
    CHECK(!finalized_) << "Duplicated Route finalization found";
    mpi_barrier();

#define ___(item__)                                           \
  {                                                           \
    auto& group_ids = GetGroupIds<NodeKind::item__>();        \
    if (!group_ids.empty()) {                                 \
      groups_[static_cast<int>(NodeKind::item__)].Finalize(); \
    }                                                         \
  }

    ROUTE_NODE_KIND_FOREACH(___)

#undef ___

    finalized_ = true;
  }

  static size_t NumGroups() { return static_cast<int>(NodeKind::__NUM__); }

  static Route& Global() {
    static Route x;
    return x;
  }

 private:
  template <NodeKind kind>
  inline std::unordered_set<int>& GetData() {
    CheckKindValid<kind>();
    return data_[static_cast<int>(kind)];
  }

  template <NodeKind kind>
  inline const std::unordered_set<int>& GetGroupIds() const {
    CheckKindValid<kind>();
    return data_[static_cast<int>(kind)];
  }
  template <NodeKind kind>
  inline std::unordered_set<int>& GetGroupIds() {
    CheckKindValid<kind>();
    return data_[static_cast<int>(kind)];
  }

  template <NodeKind kind>
  inline static void CheckKindValid() {
    static_assert(kind != NodeKind::__NUM__, "Invalid kind");
  }

 private:
  absl::InlinedVector<std::unordered_set<int>, 2> data_{static_cast<int>(NodeKind::__NUM__)};
  absl::InlinedVector<MpiGroup, 2> groups_{Route::NumGroups()};
  absl::InlinedVector<absl::InlinedVector<int, 8>, 2> node_orders_{Route::NumGroups()};
  bool initialized_{};
  bool finalized_{};
};

}  // namespace ps
}  // namespace tips
