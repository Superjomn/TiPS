#pragma once
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>

#include "tips/core/common/semaphore.h"

namespace tips {

class alignas(64) ManagedThread {
 public:
  ManagedThread() {
    thread_ = std::thread([this]() { RunThread(); });
  }

  ~ManagedThread() {
    CHECK(!active_);
    to_terminate_ = true;
    sem_start_.Post();
    thread_.join();
  }

  bool IsActive() const { return active_; }
  void Start(std::function<void()> task) {
    CHECK(!active_);
    active_ = true;
    task_   = std::move(task);
    sem_start_.Post();
  }

  void Join() {
    CHECK(active_);
    sem_finish_.Wait();
    active_ = false;
  }

 private:
  bool active_{};
  bool to_terminate_{};
  std::function<void()> task_;
  Semaphore sem_start_, sem_finish_;
  std::thread thread_;

  void RunThread() {
    while (!to_terminate_) {
      sem_start_.Wait();

      if (to_terminate_) {
        break;
      }

      task_();

      sem_finish_.Post();
    }
  }
};

}  // namespace tips
