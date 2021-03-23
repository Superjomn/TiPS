#include "tips/core/common/logging.h"
#include <iomanip>

namespace tips {

void gen_log(
    std::ostream& log_stream_, const char* file, const char* func, int lineno, const char* level, const int kMaxLen) {
  const int len = strlen(file);

  struct tm tm_time;  // Time of creation of LogMessage
  time_t timestamp = time(NULL);
#if defined(_WIN32)
  localtime_s(&tm_time, &timestamp);
#else
  localtime_r(&timestamp, &tm_time);
#endif
  struct timeval tv;
  gettimeofday(&tv, NULL);

  // print date / time
  log_stream_ << '[' << level << ' ' << std::setw(2) << 1 + tm_time.tm_mon << '/' << std::setw(2) << tm_time.tm_mday
              << ' ' << std::setw(2) << tm_time.tm_hour << ':' << std::setw(2) << tm_time.tm_min << ':' << std::setw(2)
              << tm_time.tm_sec << '.' << std::setw(3) << tv.tv_usec / 1000 << " ";

  if (len > kMaxLen) {
    log_stream_ << "..." << file + len - kMaxLen << ":" << lineno << " " << func << "] ";
  } else {
    log_stream_ << file << " " << func << ":" << lineno << "] ";
  }
}

}  // namespace tips