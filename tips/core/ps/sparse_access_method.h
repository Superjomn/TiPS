#pragma once

#include "tips/core/common/common.h"
#include "tips/core/ps/access_method.h"
#include "tips/core/ps/sparse_table.h"

namespace tips {
namespace ps {

template <typename KEY, typename PARAM, typename VALUE>
class SparseTablePullAccess : public PullAccessMethod<KEY, PARAM, VALUE> {
 public:
  using key_t      = KEY;
  using param_t    = PARAM;
  using pull_val_t = VALUE;
  using table_t    = SparseTable<key_t, param_t>;

  explicit SparseTablePullAccess(table_t *table) : table_{table} {}

  /**
   * @brief assign an initial value to param
   */
  void InitParam(const key_t &key, param_t &param, Datatype dtype, int length) override {
    param_t x(dtype, length);  // move
    param = std::move(x);
    memset(param.buffer(), 0, param.num_bytes());
    LOG(INFO) << "Init param: " << key << ": " << param;
  }

  /**
   * @brief assign param to val
   * TODO(Superjomn) Consider zero-copy way.
   */
  void GetPullValue(const key_t &key, const param_t &param, pull_val_t &val) override { val.ShadowCopyFrom(param); }

  void ApplyPullValue(const key_t &key, param_t &param, const pull_val_t &val) override {
    param = val;
    LOG(INFO) << "get param: " << param;
  }

 private:
  table_t *table_{};
};

template <typename KEY, typename PARAM, typename GRAD>
class SparseTableSgdPushAccess : public PushAccessMethod<KEY, PARAM, GRAD> {
 public:
  using key_t   = KEY;
  using param_t = PARAM;
  using grad_t  = GRAD;
  using table_t = SparseTable<KEY, PARAM>;

  explicit SparseTableSgdPushAccess(table_t *table, float lr) : table_(table), lr_(lr) {}

  void ApplyPushValue(const key_t &key, param_t &param, const grad_t &grad) override {
    param_t temp(param.dtype(), param.size());
    param_t::Mul(grad.ShadowCopy(), lr_, temp.ShadowCopy());
    param = std::move(temp);
    LOG(INFO) << "Updated " << key << " " << param;
  }

 private:
  table_t *table_;
  float lr_{0.001};
};

}  // namespace ps
}  // namespace tips
