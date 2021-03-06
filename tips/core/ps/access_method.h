#pragma once

#include <memory>
#include <vector>

#include "tips/core/common/common.h"
#include "tips/core/ps/sparse_table.h"

namespace tips {
namespace ps {

/**
 * PullAccessMethod defines an Interface for pulling operation on a Table.
 */
template <typename Key, typename Param, typename PullVal>
class PullAccessMethod {
 public:
  using key_t      = Key;
  using param_t    = Param;
  using pull_val_t = PullVal;

  /**
   * @brief assign an initial value to param
   */
  virtual void InitParam(const key_t &key, param_t &param, Datatype dtype, int length) = 0;

  /**
   * @brief assign param to val
   */
  virtual void GetPullValue(const key_t &key, const param_t &param, pull_val_t &val) = 0;

  /**
   * Worker apply pull_value to local parameter cache.
   */
  virtual void ApplyPullValue(const key_t &key, param_t &param, const pull_val_t &val) = 0;
};

/**
 * PullAccessMethod defines an Interface for pushing operation on a Table.
 */
template <typename Key, typename Param, typename Grad>
class PushAccessMethod {
 public:
  typedef Key key_t;
  typedef Param param_t;
  typedef Grad grad_t;

  /**
   * @brief update Server-side parameter with grad
   */
  virtual void ApplyPushValue(const key_t &key, param_t &param, const grad_t &grad) = 0;
};

/**
 * @brief Server-side operation agent
 *
 * Pull: worker parameter query request.
 *
 * @param Table subclass of SparseTable
 * @param AccessMethod Server-side operation on parameters
 */
template <typename Table, typename AccessMethod>
class PullAccessAgent {
 public:
  using table_t = Table;
  using key_t   = typename Table::key_t;
  using value_t = typename Table::value_t;

  using access_method_t = AccessMethod;
  using pull_val_t      = typename AccessMethod::pull_val_t;
  using pull_param_t    = typename AccessMethod::param_t;

  explicit PullAccessAgent() {}
  PullAccessAgent(table_t *table, access_method_t &&access_method)
      : table_(table), access_method_(std::move(access_method)) {}

  void Init(table_t &table, access_method_t &&access_method) {
    table_         = &table;
    access_method_ = std::move(access_method);
  }

  explicit PullAccessAgent(table_t &table, access_method_t &&access_method)
      : table_(&table), access_method_(std::move(access_method)) {}

  int ToShardId(const key_t &key) { return table_->ToShardId(key); }

  /**
   * Server-side query parameter
   */
  void GetPullValue(const key_t &key, pull_val_t &val, Datatype dtype = Datatype::kFp32, int length = 1) {
    pull_param_t param;
    if (!table_->Find(key, param)) {
      access_method_.InitParam(key, param, dtype, length);
      table_->Assign(key, param);
      table_->Find(key, param);
    }

    access_method_.GetPullValue(key, param, val);
  }

  /**
   * @brief Worker-side get pull value
   */
  void ApplyPullValue(const key_t &key, pull_param_t &param, const pull_val_t &val) {
    access_method_.ApplyPullValue(key, param, val);
  }

 private:
  table_t *table_{};
  AccessMethod access_method_;
};  // class AccessAgent

/**
 * @brief Server-side push agent
 */
template <typename Table, typename AccessMethod>
class PushAccessAgent {
 public:
  using table_t = Table;
  using key_t   = typename Table::key_t;
  using value_t = typename Table::value_t;

  using push_val_t   = typename AccessMethod::grad_t;
  using push_param_t = typename AccessMethod::param_t;

  using access_method_t = AccessMethod;

  explicit PushAccessAgent() {}
  PushAccessAgent(table_t *table, access_method_t &&access) : table_(table), access_method_(std::move(access)) {}

  explicit PushAccessAgent(table_t &table, access_method_t &&access_method)
      : table_(&table), access_method_(std::move(access_method)) {}

  /**
   * @brief update parameters with the value from remote worker nodes
   */
  void ApplyPushValue(const key_t &key, const push_val_t &push_val) {
    push_param_t *param = nullptr;
    // TODO improve this in fix mode?
    CHECK(table_->Find(key, param)) << "new key should be inited before:\t" << key;
    CHECK(param);
    access_method_.ApplyPushValue(key, *param, push_val);
  }

  int ToShardId(const key_t &key) { return table_->ToShardId(key); }

 private:
  table_t *table_{};
  AccessMethod access_method_;
};  // class PushAccessAgent

template <class Key, class Value>
SparseTable<Key, Value> &global_sparse_table() {
  static SparseTable<Key, Value> table;
  return table;
}

template <typename Table, typename AccessMethod>
auto MakePullAccess(Table &table, AccessMethod access_method) -> std::unique_ptr<PullAccessAgent<Table, AccessMethod>> {
  std::unique_ptr<PullAccessAgent<Table, AccessMethod>> res(
      new PullAccessAgent<Table, AccessMethod>(table, std::move(access_method)));
  return std::move(res);
}

template <typename Table, typename AccessMethod>
auto MakePushAccess(Table &table, AccessMethod access_method) -> std::unique_ptr<PushAccessAgent<Table, AccessMethod>> {
  std::unique_ptr<PushAccessAgent<Table, AccessMethod>> res(
      new PushAccessAgent<Table, AccessMethod>(table, std::move(access_method)));
  return std::move(res);
}

}  // namespace ps
}  // namespace tips
