#pragma once
#include <pthread.h>

#include "tips/core/common/common.h"

namespace tips {

class RWLock {
 public:
  RWLock() { CHECK((pthread_rwlock_init(&_lock, NULL) == 0)); }
  ~RWLock() { CHECK((pthread_rwlock_destroy(&_lock) == 0)); }

  void rdlock() { CHECK((pthread_rwlock_rdlock(&_lock) == 0)); }

  void wrlock() { CHECK((pthread_rwlock_wrlock(&_lock) == 0)); }

  void unlock() { CHECK((pthread_rwlock_unlock(&_lock) == 0)); }

 private:
  pthread_rwlock_t _lock;
};

// lock guard
class RwLockReadGuard {
 public:
  RwLockReadGuard(RWLock &lock) : _lock(&lock) { _lock->rdlock(); }

  ~RwLockReadGuard() { _lock->unlock(); }

 private:
  RWLock *_lock{};
};

// lock guard
class RwLockWriteGuard {
 public:
  RwLockWriteGuard(RWLock &lock) : _lock(&lock) { _lock->wrlock(); }

  ~RwLockWriteGuard() { _lock->unlock(); }

 private:
  RWLock *_lock{};
};

}  // namespace tips
