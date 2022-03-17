#include "tips/core/ps/dense_table.h"

#include <gtest/gtest.h>

#include "tips/core/operations.h"

namespace tips {
namespace ps {

TEST(DenseTable, basic) {
  tips_init();

  {
    DenseTable<int32_t> table(4, 2);
    table.Initialize();

    table.Resize(100);
    ASSERT_EQ(table.shard_num(), 8);
    ASSERT_EQ(table.size(), 100);

    auto* rcd = table.Lookup(1);
    ASSERT_TRUE(rcd);
    LOG(INFO) << "data: " << *rcd;

    table.Finalize();
  }

  tips_shutdown();
}

}  // namespace ps
}  // namespace tips