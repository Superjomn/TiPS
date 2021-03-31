#include <gtest/gtest.h>

#include "tips/core/common/thread_group.h"

namespace tips {

TEST(ThreadGroup, basic) {
  ThreadGroup group(4);
  ASSERT_EQ(group.num_threads(), 4);

  int x = 0;

  group.Run([&](int tid) { x += tid; });

  ASSERT_EQ(x, 0 + 1 + 2 + 3);
}

TEST(ThreadGroup, reset_thread_num) {
  ThreadGroup group(4);
  ASSERT_EQ(group.num_threads(), 4);

  int x = 0;
  group.Run([&](int tid) { x += tid; });

  ASSERT_EQ(x, 0 + 1 + 2 + 3);

  group.SetThreadNum(1);
  group.Run([&](int tid) { x += tid; });

  ASSERT_EQ(x, 0 + 1 + 2 + 3);
}

TEST(ThreadGroup, ParallelRunRange) {
  std::vector<int> nums;
  for (int i = 0; i < 10; i++) nums.push_back(i);

  ThreadGroup group(4);
  ParallelRunRange(
      nums.size(),
      [&](int tid, int begin, int end) {
        for (int i = begin; i != end; i++) {
          nums[i] += 1;
        }
      },
      group);

  ASSERT_EQ(nums.size(), 10);
  for (int i = 0; i < nums.size(); i++) {
    EXPECT_EQ(nums[i], i + 1);
  }
}

}  // namespace tips