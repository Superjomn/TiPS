#pragma once

#include <vector>
#include "tips/core/common/common.h"
#include "tips/core/common/vec.h"

namespace tips {

/**
 * A vector container that holds any data type.
 */
class AnyVec {
 public:
  AnyVec(Datatype dtype, int num_elements)
      : dtype_(dtype), data_(new uint8_t(num_elements * DatatypeNumBytes(dtype))), num_elements_(num_elements) {}

  template <typename T>
  ps::Vec<T> ToVec() {
    return ps::Vec<T>(absl::Span<T>(reinterpret_cast<T*>(data_), num_elements_));
  }

  Datatype dtype() const { return dtype_; }

  template <typename T>
  T* mutable_data() {
    return static_cast<T*>(data_);
  }

  template <typename T>
  const T* data() const {
    return static_cast<const T*>(data_);
  }

  size_t size() const { return num_elements_; }

 private:
  Datatype dtype_;
  void* data_{};
  int num_elements_{};
};

}  // namespace tips
