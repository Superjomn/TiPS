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
  using byte_t = uint8_t;
  AnyVec()     = default;

  //! Construct, own the memory.
  AnyVec(Datatype dtype, int num_elements)
      : dtype_(dtype),
        data_(new uint8_t[num_elements * DatatypeNumBytes(dtype)]),
        num_elements_(num_elements),
        own_data_(true) {}

  //! Construct, move the resource if owned by the other.
  AnyVec(AnyVec&& other)
      : dtype_(other.dtype_), num_elements_(other.num_elements_), own_data_(other.own_data_), data_(other.data_) {
    other.own_data_     = false;
    other.num_elements_ = 0;
    other.data_         = nullptr;
  }

  //! Construct from a buffer, without own the buffer.
  AnyVec(Datatype dtype, int num_elements, void* buf)
      : dtype_(dtype), num_elements_(num_elements), data_(static_cast<byte_t*>(buf)), own_data_{false} {}

  AnyVec(const AnyVec&) = delete;

  ~AnyVec() {
    if (data_ && own_data_) {
      delete data_;
    }
  }

  //! Copy the data, without allocating memory.
  AnyVec ShadowCopy() const {
    AnyVec res;
    res.data_         = data_;
    res.num_elements_ = num_elements_;
    res.dtype_        = dtype_;
    res.own_data_     = false;
    return res;
  }

  //! Copy the data, allocate memory.
  AnyVec Copy() const {
    AnyVec res(dtype_, num_elements_);  // Allocate memory
    memcpy(res.data_, data_, num_bytes());
    return res;
  }

  AnyVec& operator=(AnyVec&& other) {
    Clear();

    dtype_        = other.dtype_;
    own_data_     = other.own_data_;
    num_elements_ = other.num_elements_;
    data_         = other.data_;

    other.own_data_     = false;
    other.num_elements_ = 0;
    other.data_         = nullptr;
    return *this;
  }

  AnyVec& operator=(const AnyVec& other) {
    Clear();

    *this = std::move(other.ShadowCopy());
  }

  void CopyFrom(const AnyVec& other) {
    Clear();
    *this = std::move(other.Copy());
  }

  void ShadowCopyFrom(const AnyVec& other) {
    Clear();
    *this = std::move(other.ShadowCopy());
  }

  template <typename T>
  Vec<T> ToVec() {
    return Vec<T>(absl::Span<T>(reinterpret_cast<T*>(data_), num_elements_));
  }

  template <typename T>
  Vec<const T> ToVec() const {
    return Vec<const T>(absl::Span<T>(reinterpret_cast<T*>(data_), num_elements_));
  }

  void Clear() {
    if (own_data_ && data_) {
      delete data_;
    }

    own_data_     = false;
    data_         = nullptr;
    num_elements_ = 0;
  }

  Datatype dtype() const { return dtype_; }

  static void Mul(AnyVec a, AnyVec b, AnyVec out) {
    CHECK_EQ(a.dtype(), b.dtype());
    CHECK_EQ(a.dtype(), out.dtype());
    switch (a.dtype_) {
#define ___(dtype_repr, dtype)      \
  case Datatype::dtype_repr: {      \
    auto av   = a.ToVec<dtype>();   \
    auto bv   = b.ToVec<dtype>();   \
    auto outv = out.ToVec<dtype>(); \
    Vec<dtype>::Mul(av, bv, outv);  \
  } break;
      TIPS_DATATYPE_FOREACH(___)
#undef ___
      default:
        LOG(FATAL) << "Not supported dtype";
    }
  }

  static void Mul(AnyVec a, float b, AnyVec out) {
    CHECK_EQ(a.dtype(), out.dtype());
    switch (a.dtype_) {
#define ___(dtype_repr, dtype)                        \
  case Datatype::dtype_repr: {                        \
    auto av   = a.ToVec<dtype>();                     \
    auto outv = out.ToVec<dtype>();                   \
    Vec<dtype>::Mul(av, static_cast<dtype>(b), outv); \
  } break;
      TIPS_DATATYPE_FOREACH(___)
#undef ___
      default:
        LOG(FATAL) << "Not supported dtype";
    }
  }

  template <typename T>
  T* mutable_data() {
    CHECK_EQ(DatatypeTypetrait<T>(), dtype_) << "type mismatch";
    return reinterpret_cast<T*>(data_);
  }

  template <typename T>
  const T* data() const {
    CHECK_EQ(DatatypeTypetrait<T>(), dtype_) << "type mismatch";
    return reinterpret_cast<const T*>(data_);
  }

  byte_t* buffer() { return data_; }
  const byte_t* buffer() const { return data_; }

  size_t size() const { return num_elements_; }

  size_t num_bytes() const { return num_elements_ * DatatypeNumBytes(dtype_); }

  friend std::ostream& operator<<(std::ostream& os, const AnyVec& other) {
    switch (other.dtype()) {
#define ___(repr, dtype)                \
  case Datatype::repr:                  \
    os << other.ToVec<dtype>().ToStr(); \
    break;

      TIPS_DATATYPE_FOREACH(___)

#undef ___
    }
  }

 private:
  Datatype dtype_;
  byte_t* data_{};
  int num_elements_{};
  bool own_data_{};
};

}  // namespace tips
