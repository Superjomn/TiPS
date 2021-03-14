#include <atomic>
#include <functional>
#include <vector>

#include "swiftsnails/core/common/managed_thread.h"

namespace swifts {

class ThreadPool {
 public:
  using task_t = std::function<void(int)>;

  explicit ThreadPool(int n_threads = 0) { ResizeThreads(n_threads); }

  void ResizeThreads(int n_threads) {
    CHECK_GT(n_threads, 0);
    CHECK(!Joinable());

    if (n_threads == threads_.size()) return;

    threads_ = std::vector<ManagedThread>(n_threads);
  }

  void Run(task_t task) {
    Start(std::move(task));
    Join();
  }

 protected:
  bool Joinable() const { return static_cast<bool>(task_); }
  void Start(task_t task) {
    CHECK(task);
    CHECK(!Joinable());

    if (threads_.empty()) {
      task(0);
      return;
    }

    task_ = task;

    for (int i = 0; i < parallel_num_; i++) {
      threads_[i].Start([this, i] { task_(i); });
    }
  }

  void Join() {
    CHECK(Joinable());

    for (int i = 0; i < threads_.size(); i++) {
      threads_[i].Join();
    }

    task_ = nullptr;
  }

 private:
  std::vector<ManagedThread> threads_;
  task_t task_;
  int parallel_num_{};
};

}  // namespace swifts
