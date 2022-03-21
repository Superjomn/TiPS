#include <gtest/gtest.h>

#include "tips/core/common/buffer.h"

namespace tips {

template <typename T>
void TestType() {
  Buffer vec(DatatypeTypetrait<T>(), 20);
  ASSERT_EQ(vec.size(), 20);
  ASSERT_EQ(vec.num_bytes(), 20 * sizeof(T));

  auto* data = vec.mutable_data<T>();
  for (int i = 0; i < vec.size(); i++) {
    data[i] = i;
  }

  Buffer out(vec.dtype(), vec.size());

  Vec<T> vec1 = vec.ToVec<T>();
  auto out1   = out.ToVec<T>();

  Vec<T>::Add(vec1, vec1, out1);

  auto* output_data = out.mutable_data<T>();
  for (int i = 0; i < out.size(); i++) {
    ASSERT_NEAR(output_data[i], data[i] * 2, 1e-5);
  }
}

TEST(AnyVec, basic) {
  TestType<float>();
  TestType<double>();
  TestType<int32_t>();
  TestType<int64_t>();
}

}  // namespace tips