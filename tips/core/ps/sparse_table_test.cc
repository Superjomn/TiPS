#include <gtest/gtest.h>

#include "tips/core/operations.h"
#include "tips/core/ps/access_method.h"

namespace tips {
namespace ps {

TEST(SparseTableShard, basic) {
  SparseTableShard<size_t, float> shard;
  shard.Assign(1, 1.2);
  shard.Assign(2, 2.2);

  float val;
  ASSERT_TRUE(shard.Find(1, val));
  ASSERT_NEAR(val, 1.2, 1e-5);
}

TEST(SparseTable, basic) {
  SparseTable<size_t /*key*/, float /*param*/> table(2, 4);
  float* x;
  ASSERT_FALSE(table.Find(0, x));

  table.Assign(2, 2.0);
  ASSERT_TRUE(table.Find(2, x));
  ASSERT_EQ(*x, 2);
}

}  // namespace ps
}  // namespace tips
