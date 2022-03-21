#pragma once

#include "tips/core/common/common.h"
#include "tips/core/ps/access_method.h"
#include "tips/core/ps/sparse_table.h"

namespace tips {
namespace ps {

template <typename PARAM, typename VALUE>
class SparseTablePullAccess : public PullAccessMethod<PARAM, VALUE> {
 public:
  using key_t         = uint64_t;
  using param_t       = PARAM;
  using pull_val_t    = VALUE;
  using table_t       = SparseTable;
  using initializer_t = std::function<void(const SparseTable &, param_t *)>;

  explicit SparseTablePullAccess(table_t *table, initializer_t initializer)
      : table_{table}, initializer_(initializer) {}

  /**
   * @brief assign an initial value to param
   */
  void InitParam(const key_t &key, param_t &param, Datatype dtype, int length) override {
    param_t x(dtype, length);  // move
    param = std::move(x);
    initializer_(*table_, param);
  }

  /**
   * @brief assign param to val
   * TODO(Superjomn) Consider zero-copy way.
   */
  void GetPullValue(const key_t &key, const param_t &param, pull_val_t &val) override { val.ShadowCopyFrom(param); }

  void ApplyPullValue(const key_t &key, param_t &param, const pull_val_t &val) override { param = val; }

 private:
  table_t *table_{};
  initializer_t initializer_;
};

template <typename PARAM, typename GRAD>
class SparseTableSgdPushAccess : public PushAccessMethod<PARAM, GRAD> {
 public:
  using key_t   = uint64_t;
  using param_t = PARAM;
  using grad_t  = GRAD;
  using table_t = SparseTable;

  explicit SparseTableSgdPushAccess(table_t *table, float lr = 0.001) : table_(table), lr_(lr) {}

  void ApplyPushValue(const key_t &key, param_t &param, const grad_t &grad) override {
    param_t temp(param.dtype(), param.size());
    param_t::Mul(grad.ShadowCopy(), lr_, temp.ShadowCopy());
    param = std::move(temp);
  }

 private:
  table_t *table_{};
  float lr_;
};

}  // namespace ps
}  // namespace tips
