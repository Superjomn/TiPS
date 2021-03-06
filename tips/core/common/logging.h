
// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*
 * This file implements an lightweight alternative for glog, which is more
 * friendly for mobile.
 */
#pragma once

#ifndef _LOGGING_H_
#define _LOGGING_H_

#include <assert.h>
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>

#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>

#define LITE_WITH_LOG

#define LOG(status) LOG_##status.stream()
#define LOG_INFO ::tips::LogMessage(__FILE__, __FUNCTION__, __LINE__, "I")
#define LOG_ERROR LOG_INFO
#define LOG_WARNING ::tips::LogMessage(__FILE__, __FUNCTION__, __LINE__, "W")
#define LOG_FATAL ::tips::LogMessageFatal(__FILE__, __FUNCTION__, __LINE__)

// VLOG()
#define VLOG(level) ::tips::VLogMessage(__FILE__, __FUNCTION__, __LINE__, level).stream()

// CHECK()
// clang-format off
#define CHECK(x) if (!(x)) ::tips::LogMessageFatal(__FILE__, __FUNCTION__, __LINE__).stream() << "Check failed: " #x << ": " // NOLINT(*)
#define _CHECK_BINARY(x, cmp, y) CHECK((x cmp y)) << (x) << "!" #cmp << (y) << " " // NOLINT(*)

// clang-format on
#define CHECK_EQ(x, y) _CHECK_BINARY(x, ==, y)
#define CHECK_NE(x, y) _CHECK_BINARY(x, !=, y)
#define CHECK_LT(x, y) _CHECK_BINARY(x, <, y)
#define CHECK_LE(x, y) _CHECK_BINARY(x, <=, y)
#define CHECK_GT(x, y) _CHECK_BINARY(x, >, y)
#define CHECK_GE(x, y) _CHECK_BINARY(x, >=, y)
#define CHECK_NEAR(x, y, thre) \
  CHECK(std::abs((x) - (y)) <= thre) << x << " and " << y << " diff is larger than " << thre;

namespace tips {

#define LITE_WITH_EXCEPTION

#ifdef LITE_WITH_EXCEPTION
struct PaddleLiteException : public std::exception {
  const std::string exception_prefix = "Paddle-Lite C++ Exception: \n";
  std::string message;
  explicit PaddleLiteException(const char* detail) { message = exception_prefix + std::string(detail); }
  const char* what() const noexcept { return message.c_str(); }
};
#endif

void gen_log(std::ostream& log_stream_,
             const char* file,
             const char* func,
             int lineno,
             const char* level,
             const int kMaxLen = 20);

// LogMessage
class LogMessage {
 public:
  LogMessage(const char* file, const char* func, int lineno, const char* level = "I") {
    level_ = level;
    gen_log(log_stream_, file, func, lineno, level);
  }

  ~LogMessage() {
    log_stream_ << '\n';
    fprintf(stderr, "%s", log_stream_.str().c_str());
  }

  std::ostream& stream() { return log_stream_; }

 protected:
  std::stringstream log_stream_;
  std::string level_;

  LogMessage(const LogMessage&) = delete;
  void operator=(const LogMessage&) = delete;
};

// LogMessageFatal
class LogMessageFatal : public LogMessage {
 public:
  LogMessageFatal(const char* file, const char* func, int lineno, const char* level = "F")
      : LogMessage(file, func, lineno, level) {}

  ~LogMessageFatal()
#ifdef LITE_WITH_EXCEPTION
      noexcept(false)
#endif
  {
    log_stream_ << '\n';
    fprintf(stderr, "%s", log_stream_.str().c_str());

#ifdef LITE_WITH_EXCEPTION
    throw PaddleLiteException(log_stream_.str().c_str());
#else
#ifndef LITE_ON_TINY_PUBLISH
    abort();
#else
    // If we decide whether the process exits according to the NDEBUG macro
    // definition, assert() can be used here.
    abort();
#endif
#endif
  }
};

// VLOG
class VLogMessage {
 public:
  VLogMessage(const char* file, const char* func, int lineno, const int32_t level_int = 0) {
    const char* GLOG_v = getenv("GLOG_v");
    GLOG_v_int         = (GLOG_v && atoi(GLOG_v) > 0) ? atoi(GLOG_v) : 0;
    this->level_int    = level_int;
    if (GLOG_v_int < level_int) {
      return;
    }
    const char* level = std::to_string(level_int).c_str();
    ::tips::gen_log(log_stream_, file, func, lineno, level);
  }

  ~VLogMessage() {
    if (GLOG_v_int < this->level_int) {
      return;
    }
    log_stream_ << '\n';
    fprintf(stderr, "%s", log_stream_.str().c_str());
  }

  std::ostream& stream() { return log_stream_; }

 protected:
  std::stringstream log_stream_;
  int32_t GLOG_v_int;
  int32_t level_int;

  VLogMessage(const VLogMessage&) = delete;
  void operator=(const VLogMessage&) = delete;
};
#else
class Voidify {
 public:
  Voidify() {}
  ~Voidify() {}

  template <typename T>
  Voidify& operator<<(const T& obj) {
    return *this;
  }
};

class VoidifyFatal : public Voidify {
 public:
#ifdef LITE_WITH_EXCEPTION
  ~VoidifyFatal() noexcept(false) { throw std::exception(); }
#else
  ~VoidifyFatal() {
    // If we decide whether the process exits according to the NDEBUG macro
    // definition, assert() can be used here.
    abort();
  }
#endif
};

#endif

}  // namespace tips
