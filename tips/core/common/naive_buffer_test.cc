#include "tips/core/common/naive_buffer.h"

#include <gtest/gtest.h>

namespace tips {

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
  TestPodValue(true);  // bool
}

TEST(NaiveBuffer, string) {
  NaiveBuffer buffer;
  std::string v = "hello world!";
  buffer << v;
  buffer << 123;

  NaiveBuffer read_buffer(buffer.data(), buffer.size());
  std::string v0;
  int v1;
  read_buffer >> v0 >> v1;

  ASSERT_EQ(v0, v);
  ASSERT_EQ(v1, 123);
}

}  // namespace tips