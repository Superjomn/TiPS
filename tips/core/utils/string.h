#pragma once
#include <absl/types/span.h>

#include <sstream>

namespace tips {

template <typename T>
std::string ToString(absl::Span<T> datas) {
  std::stringstream ss;
  for (int i = 0; i < datas.size() - 1; i++) {
    ss << datas[i] << " ";
  }
  if (!datas.empty()) {
    ss << datas.back();
  }
  return ss.str();
}

}  // namespace tips
