#pragma once

#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <vector>

#include "tips/core/common/logging.h"

namespace tips {

/**
 * Channel is a thread-safe producer-consumer queue.
 */
template <typename T>
class Channel {
 public:
  Channel() = default;

  explicit Channel(size_t capacity) : capacity_(std::min(max_capacity(), capacity)) {}

  size_t capacity() const { return capacity_; }

  bool closed() const { return closed_; }

  void Open() {
    std::lock_guard<std::mutex> lock(mu_);
    closed_ = false;
    Notify();
  }

  void Close() {
    std::lock_guard<std::mutex> lock(mu_);
    closed_ = true;
    full_cond_.notify_all();
    empty_cond_.notify_all();
  }

  size_t size() const {
    std::lock_guard<std::mutex> lock(mu_);
    return data_.size();
  }

  bool empty() const {
    std::lock_guard<std::mutex> lock(mu_);
    return data_.empty();
  }

  //! Read one record.
  bool Read(T* p) {
    std::unique_lock<std::mutex> lock(mu_);
    bool res = Read(p, &lock);
    Notify();
    return res;
  }

  bool Write(const T& p) {
    std::unique_lock<std::mutex> lock(mu_);
    bool res = Write(p, &lock);
    Notify();
    return res;
  }

  size_t WriteMove(T&& p) {
    std::unique_lock<std::mutex> lock(mu_);
    bool res = WriteMove(std::move(p), &lock);
    Notify();
    return res;
  }

 private:
  bool WaitForRead(std::unique_lock<std::mutex>* lock) {
    while (empty_unlocked() && !closed_) {
      if (full_waiters_count_ != 0) {
        full_cond_.notify_one();
      }

      empty_waiters_count_++;
      empty_cond_.wait(*lock);
      empty_waiters_count_--;
    }

    return !empty_unlocked();
  }

  bool WaitForWrite(std::unique_lock<std::mutex>* lock) {
    while (full_unlocked() && !closed_) {
      if (empty_waiters_count_ != 0) {
        empty_cond_.notify_one();
      }

      full_waiters_count_++;
      full_cond_.wait(*lock);
      full_waiters_count_--;
    }

    return !closed_;
  }

  bool Read(T* p, std::unique_lock<std::mutex>* lock) {
    if (WaitForRead(lock)) {
      *p = std::move(data_.front());
      data_.pop_front();
      return true;
    }
    return false;
  }

  bool Write(const T& p, std::unique_lock<std::mutex>* lock) {
    if (WaitForWrite(lock)) {
      data_.push_back(p);
      return true;
    }
    return false;
  }

  bool WriteMove(T&& p, std::unique_lock<std::mutex>* lock) {
    if (WaitForWrite(lock)) {
      data_.template emplace_back(std::move(p));
      return true;
    }
    return false;
  }

  void Notify() {
    if (empty_waiters_count_ > 0 && (!empty_unlocked() || closed_)) {
      empty_cond_.notify_one();
    }

    if (full_waiters_count_ > 0 && (!full_unlocked() || closed_)) {
      full_cond_.notify_one();
    }
  }

  inline bool empty_unlocked() const { return data_.empty(); }

  bool full_unlocked() const { return data_.size() >= capacity_ + reading_count_; }

  static constexpr size_t max_capacity() { return std::numeric_limits<size_t>::max() / 2; }

 private:
  size_t capacity_{max_capacity()};
  size_t block_size_{1024};
  bool closed_{};

  mutable std::mutex mu_;
  std::deque<T> data_;
  size_t reading_count_{};
  int empty_waiters_count_{};
  int full_waiters_count_{};

  std::condition_variable empty_cond_;
  std::condition_variable full_cond_;
};

/**
 * Create a Channel that is sharable.
 */
template <typename T>
std::shared_ptr<Channel<T>> MakeChannel(size_t capacity = std::numeric_limits<size_t>::max()) {
  return std::make_shared<Channel<T>>(capacity);
}

}  // namespace tips
