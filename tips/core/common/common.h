#pragma once

#include <errno.h>
#include <ifaddrs.h>
#include <netinet/in.h>
#include <string.h>
#include <sys/types.h>

#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "tips/core/common/datatype.h"
#include "tips/core/common/logging.h"
#include "tips/core/common/macro.h"

namespace tips {

// Call func(args...). If interrupted by signal, recall the function.
template <class FUNC, class... ARGS>
auto ignore_signal_call(FUNC&& func, ARGS&&... args) -> typename std::result_of<FUNC(ARGS...)>::type {
  for (;;) {
    auto err = func(args...);

    if (err < 0 && errno == EINTR) {
      LOG(INFO) << "Signal is caught. Ignored.";
      continue;
    }

    return err;
  }
}
std::string GetLocalIp();

std::string StringFormat(const std::string& fmt_str, ...);

#define ZCHECK(op) CHECK_EQ((op), 0)

template <typename Elem>
class OrderedMap {
  std::vector<Elem> list_;
  std::unordered_map<std::string, int> order_;

 public:
  void Set(const std::string& key, Elem&& e) {
    list_.emplace_back(std::move(e));
    CHECK(!order_.count(key)) << "duplicate key '" << key << "' found";
    order_[key] = list_.size() - 1;
  }

  const Elem& Get(const std::string& key) const {
    CHECK(order_.count(key)) << "No key " << key << " found";
    return list_[order_.at(key)];
  }

  Elem& GetMutable(const std::string& key) {
    CHECK(order_.count(key)) << "No key " << key << " found";
    return list_[order_[key]];
  }

  std::vector<Elem>& elements() { return list_; }
  const std::vector<Elem>& elements() const { return list_; }
};

}  // namespace tips
