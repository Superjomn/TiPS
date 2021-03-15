#include "swiftps/core/common/naive_buffer.h"

#include <gtest/gtest.h>

namespace swifts {

template <typename POD>
void TestPodValue(POD v) {
  NaiveBuffer buffer;
  buffer << v;

  NaiveBuffer read_buffer(buffer.data(), buffer.size());
  POD res;
  read_buffer >> res;
  ASSERT_EQ(res, v);
}

TEST(NaiveBuffer, POD) {
  TestPodValue(1.23);  // float
  TestPodValue(123);   // int32_t
  TestPodValue(true);   // bool
}

}  // namespace swifts