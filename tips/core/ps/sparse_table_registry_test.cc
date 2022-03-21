#include "tips/core/ps/sparse_table_registry.h"
#include <gtest/gtest.h>
#include "tips/core/common/common.h"

namespace tips {
namespace ps {

TEST(SparseTableRegistry, basic) {
  SparseTableRegistry registry;
  auto table_id = registry.AddTable("test", 10, 4);
  ASSERT_EQ(registry.size(), 1UL);

  auto* table = registry.GetTable(table_id);
  ASSERT_EQ(table->GetTableName(), "test");

  auto* non_table = registry.GetTable("Not exist");
  ASSERT_FALSE(non_table);
}

}  // namespace ps
}  // namespace tips
