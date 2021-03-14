#pragma once
#include <condition_variable>
#include <functional>
#include <mutex>

namespace swifts {

class alignas(64) ManagedThread {
 public:
  ManagedThread() {}

  ~ManagedThread() {}

 private:
  bool active_{};
  bool to_terminate_{};
  std::function<void()> task_;
};

}  // namespace swifts
