#pragma once
#include <pthread.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <atomic>

namespace tips {

class SpinLock {
 public:
  void lock() {
    while (lck_.test_and_set(std::memory_order_acquire)) {
    }
  }

  void unlock() { lck_.clear(std::memory_order_release); }

 private:
  std::atomic_flag lck_ = ATOMIC_FLAG_INIT;
};

}  // namespace tips
