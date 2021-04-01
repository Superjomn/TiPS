#include <gtest/gtest.h>
#include "tips/core/operations.h"
#include "tips/core/ps/access_method.h"

namespace tips {
namespace ps {

TEST(SparseAccess, basic) {
  tips_init();
  {
    SparseTable<size_t /*key*/, float /*param*/> table;
    float* x;
    ASSERT_FALSE(table.Find(1, x));

    table.Assign(1, 2.0);
    ASSERT_TRUE(table.Find(1, x));
    ASSERT_EQ(*x, 2);
  }
  tips_shutdown();
}

}  // namespace ps
}  // namespace tips
