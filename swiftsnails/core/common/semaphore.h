#pragma once
#include <glog/logging.h>
#include <semaphore.h>

#include <condition_variable>
#include <mutex>

#include "swiftsnails/core/common/common.h"

namespace swifts {

class Semaphore {
 public:
  Semaphore() { CHECK_EQ(sem_init(&sem_, 0, 0), 0); }

  ~Semaphore() { CHECK_EQ(sem_destroy(&sem_), 0); }

  void Post() { CHECK_EQ(sem_post(&sem_), 0); }

  void Wait() { CHECK_EQ(ignore_signal_call(sem_wait, &sem_), 0); }

  bool TryWait() {
    int err = 0;
    CHECK((err = ignore_signal_call(sem_trywait, &sem_), err == 0 || errno == EAGAIN));
    return err == 0;
  }

 private:
  sem_t sem_;
};

}  // namespace swifts
