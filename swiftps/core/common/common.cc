#include "swiftps/core/common/common.h"

#include <arpa/inet.h>
#include <net/if.h>
#include <stdarg.h>
#include <sys/ioctl.h>

#include <cstring>
#include <memory>

namespace swifts {

std::string GetLocalIp() {
  int sockfd = -1;
  char buf[512];
  struct ifconf ifconf;
  struct ifreq* ifreq;

  ifconf.ifc_len = 512;
  ifconf.ifc_buf = buf;
  PCHECK((sockfd = socket(AF_INET, SOCK_DGRAM, 0)) >= 0);
  PCHECK(ioctl(sockfd, SIOCGIFCONF, &ifconf) >= 0);
  PCHECK(0 == close(sockfd));

  ifreq = (struct ifreq*)buf;

  for (int i = 0; i < int(ifconf.ifc_len / sizeof(struct ifreq)); i++) {
    std::string ip;
    ip = inet_ntoa(((struct sockaddr_in*)&ifreq->ifr_addr)->sin_addr);

    if (ip != "127.0.0.1") {
      return ip;
    }

    ifreq++;
  }

  LOG(FATAL) << "IP not found";
  return "";
}

std::string StringFormat(const std::string& fmt_str, ...) {
  /* Reserve two times as much as the length of the fmt_str */
  int final_n, n = (static_cast<int>(fmt_str.size())) * 2;
  std::unique_ptr<char[]> formatted;
  va_list ap;
  while (1) {
    formatted.reset(new char[n]);                 /* Wrap the plain char array into the unique_ptr */
    std::strcpy(&formatted[0], fmt_str.c_str());  // NOLINT
    va_start(ap, fmt_str);
    final_n = vsnprintf(&formatted[0], n, fmt_str.c_str(), ap);
    va_end(ap);
    if (final_n < 0 || final_n >= n)
      n += abs(final_n - n + 1);
    else
      break;
  }
  return std::string(formatted.get());
}
}  // namespace swifts