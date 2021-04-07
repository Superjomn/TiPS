#pragma once

#include <absl/types/span.h>
#include <cmath>

#include "tips/core/common/common.h"

namespace tips {

/**
 * Vec simplifies the operation of vectors, it doesn't own the data, but just hold the address, similar to absl::Span.
 */
template <typename T>
class Vec {
 public:
  using value_type = T;
  using self_type  = Vec<T>;

  Vec(absl::Span<T> data) : data_(data.data()), size_(data.size()) {}

  Vec(const Vec &other) {
    data_ = other.data_;
    size_ = other.size_;
  }

  Vec &operator=(const Vec &other) {
    data_ = other.data_;
    size_ = other.size_;
    return *this;
  }

  friend std::vector<Vec> outer(const Vec &a, const Vec &b) {
    CHECK_GT(a.size(), 0);
    CHECK_GT(b.size(), 0);

    std::vector<Vec> vs;
    for (size_t i = 0; i < a.size(); i++) {
      Vec v(b.size());
      vs.push_back(std::move(b));
    }
    for (size_t i = 0; i < a.size(); i++) {
      auto &v = vs[i];
      for (size_t j = 0; j < b.size(); j++) {
        v[j] = a[i] * b[j];
      }
    }
    return std::move(vs);
  }

  void Init(bool random_init = false) {
    CHECK_GT(size_, 0);
    CHECK(!data_) << "data can be inited only once";
    for (size_t i = 0; i < size_; ++i) {
      data()[i] = 0.0;
    }
    if (random_init) RandInit(0.0);
  }

  void Clear() {
    for (int i = 0; i < size_; i++) {
      data_[i] = 0;
    }
  }

  void random() { RandInit(0.0); }

  value_type &operator[](size_t i) {
    CHECK_GE(i, 0);
    CHECK_LE(i, size());
    return data_[i];
  }
  const value_type &operator[](size_t i) const {
    CHECK(i >= 0 && i < size());
    return data_[i];
  }

  value_type dot(const Vec &other) const {
    CHECK_EQ(size(), other.size());
    value_type res = 0.0;
    for (size_t i = 0; i < size(); i++) {
      res += data()[i] * other[i];
    }
    return std::move(res);
  }

  friend std::ostream &operator<<(std::ostream &os, const Vec &other) {
    os << "Vec:\t";
    for (uint32_t i = 0; i < other.size(); ++i) {
      os << other[i] << " ";
    }
    return os;
  }

  std::string ToStr() const {
    std::stringstream ss;
    ss << "Vec:\t";
    for (uint32_t i = 0; i < size(); ++i) {
      ss << data()[i] << " ";
    }
    return std::move(ss.str());
  }

  size_t size() const { return size_; }

  value_type *data() { return data_; }
  const value_type *data() const { return data_; }

  friend Vec operator*(const Vec &vec, value_type b) {
    Vec v(vec);
    for (size_t i = 0; i < v.size(); i++) {
      v[i] *= b;
    }
    return std::move(v);
  }
  friend Vec operator*(value_type b, const Vec &vec) { return std::move(vec * b); }
  friend Vec operator*(const Vec &a, const Vec &b) {
    CHECK_EQ(a.size(), b.size());
    Vec tmp(a);
    for (size_t i = 0; i < a.size(); i++) {
      tmp[i] *= b[i];
    }
    return std::move(tmp);
  }
  friend Vec operator/(const Vec &vec, value_type b) {
    Vec v(vec);
    for (size_t i = 0; i < v.size(); i++) {
      v[i] /= b;
    }
    return std::move(v);
  }
  friend Vec operator/(value_type b, const Vec &vec) { return std::move(1.0 / b * vec); }
  friend Vec operator/(const Vec &a, const Vec &b) {
    Vec v(a);
    for (size_t i = 0; i < v.size(); i++) {
      v[i] /= b[i];
    }
    return std::move(v);
  }
  friend Vec operator+(const Vec &vec, value_type b) {
    Vec v(vec);
    for (size_t i = 0; i < v.size(); i++) {
      v[i] += b;
    }
    return std::move(v);
  }
  friend Vec operator+(value_type b, const Vec &vec) { return std::move(vec + b); }
  friend Vec operator-(const Vec &vec, value_type b) {
    Vec v(vec);
    for (size_t i = 0; i < v.size(); i++) {
      v[i] -= b;
    }
    return std::move(v);
  }
  friend Vec operator-(value_type b, const Vec &vec) { return std::move(-1.0 * vec + b); }
  friend Vec operator-(const Vec &a, const Vec &b) {
    Vec v(a);
    for (size_t i = 0; i < v.size(); i++) {
      v[i] -= b[i];
    }
    return std::move(v);
  }
  friend Vec operator+=(Vec &a, const Vec &b) {
    CHECK_EQ(a.size(), b.size());
    for (size_t i = 0; i < a.size(); ++i) {
      a[i] += b[i];
    }
    return a;
  }
  friend Vec operator+=(Vec &a, value_type b) {
    for (size_t i = 0; i < a.size(); ++i) {
      a[i] += b;
    }
    return a;
  }
  friend Vec &operator-=(Vec &a, const Vec &b) {
    for (size_t i = 0; i < a.size(); ++i) {
      a[i] -= b[i];
    }
    return a;
  }
  friend Vec &operator-=(Vec &a, value_type b) {
    for (size_t i = 0; i < a.size(); ++i) {
      a[i] -= b;
    }
    return a;
  }
  friend Vec &operator/=(Vec &a, value_type b) {
    for (size_t i = 0; i < a.size(); ++i) {
      a[i] /= b;
    }
    return a;
  }

  static void Mul(Vec<T> in0, Vec<T> in1, Vec<T> out) {
    auto *in0_data = in0.data();
    auto *in1_data = in1.data();
    auto *out_data = out.data();

    CHECK_EQ(in0.size(), in1.size());
    CHECK_EQ(in0.size(), out.size());

    for (int i = 0; i < in0.size(); i++) {
      out_data[i] = in0_data[i] * in1_data[i];
    }
  }

  static void Mul(Vec<T> in0, T scale, Vec<T> out) {
    auto *in0_data = in0.data();
    auto *out_data = out.data();

    CHECK_EQ(in0.size(), out.size());

    for (int i = 0; i < in0.size(); i++) {
      out_data[i] = in0_data[i] * scale;
    }
  }

  static void Add(Vec<T> in0, Vec<T> in1, Vec<T> out) {
    auto *in0_data = in0.data();
    auto *in1_data = in1.data();
    auto *out_data = out.data();

    CHECK_EQ(in0.size(), in1.size());
    CHECK_EQ(in0.size(), out.size());

    for (int i = 0; i < in0.size(); i++) {
      out_data[i] = in0_data[i] + in1_data[i];
    }
  }

  static void Add(Vec<T> in0, T bias, Vec<T> out) {
    auto *in0_data = in0.data();
    auto *out_data = out.data();

    CHECK_EQ(in0.size(), out.size());

    for (int i = 0; i < in0.size(); in0++) {
      out_data[i] = in0_data[i] + bias;
    }
  }

 protected:
  void RandInit(float offset = 0.5) {
    for (size_t i = 0; i < size(); i++) data_[i] = (rand() / (float)RAND_MAX - 0.5) / size_;
  }

 private:
  value_type *data_{};
  size_t size_{0};
};  // class Vec

template <typename T>
Vec<T> sqrt(const Vec<T> &vec) {
  Vec<T> tmp(vec);
  for (size_t i = 0; i < vec.size(); i++) {
    tmp[i] = std::sqrt(tmp[i]);
  }
  return std::move(tmp);
}

}  // namespace tips
