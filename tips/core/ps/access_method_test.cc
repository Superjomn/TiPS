#include "tips/core/ps/access_method.h"

#include <gtest/gtest.h>

#include "tips/core/operations.h"
#include "tips/core/ps/sparse_access_method.h"

namespace tips {
namespace ps {

TEST(SparseAccess, begin) { tips_init(); }

TEST(SparseAccess, pull) {
  SparseTable<size_t /*key*/, float /*param*/> table;
  table.Assign(1, 1.f);

  SparseTablePullAccess<size_t, float, float> pull_access(&table);

  auto pull = MakePullAccess(table, pull_access);

  float x;
  pull->GetPullValue(1, x);
  ASSERT_EQ(x, 1.f);

  pull->ApplyPullValue(1, x, 2);
  ASSERT_EQ(x, 2);
}

TEST(SparseAccess, sgd_push) {
  SparseTable<size_t /*key*/, float /*param*/> table;
  table.Assign(1, 1.f);

  SparseTableSgdPushAccess<size_t, float, float> push_access(&table, 0.01f);
  SparseTablePullAccess<size_t, float, float> pull_access(&table);

  auto push = MakePushAccess(table, push_access);
  push->ApplyPushValue(1, 1.f);

  auto pull = MakePullAccess(table, pull_access);
  float x;
  pull->GetPullValue(1, x);
  LOG(INFO) << x;
  ASSERT_NEAR(x, 1.01f, 1e-5);
}

TEST(SparseAccess, end) { tips_shutdown(); }

}  // namespace ps
}  // namespace tips