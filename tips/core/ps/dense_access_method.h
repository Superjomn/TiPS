#pragma once
#include "tips/core/ps/access_method.h"
#include "tips/core/ps/dense_table.h"

//! \file This file defines some actual AccessMethods for DenseTable.

namespace tips {
namespace ps {

template <typename KEY, typename PARAM, typename VALUE>
class BasicDensePullAccessMethod : public PullAccessMethod<KEY, PARAM, VALUE> {
 public:
  using key_t      = KEY;
  using param_t    = PARAM;
  using value_t    = VALUE;
  using value_type = typename PARAM::value_type;
  using table_t    = DenseTable<value_type>;

  explicit BasicDensePullAccessMethod(table_t *table) : table_(table) { CHECK(table_); }

  void InitParam(const key_t &key, param_t &param) override {}

 private:
  table_t *table_{};
};

}  // namespace ps
}  // namespace tips
