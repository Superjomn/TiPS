#include <gtest/gtest.h>

#include "tips/core/operations.h"
#include "tips/core/ps/access_method.h"

namespace tips {
namespace ps {

TEST(SparseTable, set_get) {
  SparseTable table(2, 4);
  float* x;
  ASSERT_FALSE(table.Find(1, x));
  table.Assign(1, 2.0f);
  ASSERT_TRUE(table.Find(1, x));
  ASSERT_EQ(*x, 2);
}

}  // namespace ps
}  // namespace tips
