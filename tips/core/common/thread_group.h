#pragma once

#include <absl/synchronization/barrier.h>
#include <functional>
#include <memory>
#include <thread>
#include <vector>
#include "tips/core/common/common.h"
#include "tips/core/common/managed_thread.h"

namespace tips {

class ThreadGroup {
 public:
  using task_t = std::function<void(int)>;

  ThreadGroup() = default;

  explicit ThreadGroup(int thread_num) {
    CHECK_GE(thread_num, 1) << "At least 1 thread is required for a ThreadGroup";
    SetThreadNum(thread_num);
  }

  /**
   * Resize the number of threads.
   */
  void SetThreadNum(int thread_num) {
    CHECK_GE(thread_num, 1) << "At least 1 thread is required for a ThreadGroup";
    CHECK(!func_);

    if (thread_num == threads_.size()) return;

    threads_ = std::vector<ManagedThread>(thread_num);
    barrier_.reset(new absl::Barrier(thread_num));
  }

  size_t num_threads() const { return threads_.size(); }

  void Start(const task_t& func) {
    CHECK(func);
    CHECK(!func_);

    func_ = func;

    for (int i = 0; i < threads_.size(); i++) {
      threads_[i].Start([this, i] {
        thread_id() = i;
        func_(i);
      });
    }
  }

  void Run(const task_t& func) {
    Start(func);
    Join();
  }

  void Join() {
    CHECK(func_);
    for (auto& t : threads_) {
      t.Join();
    }

    func_ = nullptr;
  }

  /**
   * Thread local singleton.
   */
  static ThreadGroup& GetThreadLocal() {
    thread_local ThreadGroup g;
    return g;
  }

  void Barrier() { barrier_->Block(); }

  static int& thread_id() {
    thread_local int x = 0;
    return x;
  }

  TIPS_DISALLOW_COPY_AND_ASSIGN(ThreadGroup)

 private:
  task_t func_;
  std::vector<ManagedThread> threads_;
  std::unique_ptr<absl::Barrier> barrier_;
};

template <typename Func>
void ParallelRunRange(size_t n, Func&& func, ThreadGroup& thread_group = ThreadGroup::GetThreadLocal()) {
  int num_threads = thread_group.num_threads();
  thread_group.Run([n, &func, num_threads](int tid) {
    if (n < num_threads) {
      if (tid < n) {
        func(tid, tid, tid + 1);
      }
    } else {
      size_t part = (n + num_threads) / num_threads;
      if (part * tid < n) {
        func(tid, part * tid, std::min(part * (tid + 1), n));
      }
    }
  });
}

}  // namespace tips
