#pragma once
#include <absl/strings/string_view.h>
#include "tips/core/common/common.h"

namespace tips {
namespace ps {

/**
 * Server: the server service of PS.
 */
class Server {
 public:
  void AddTableService(absl::string_view table_name, int num_nodes, int num_local_shards);

 private:
};

}  // namespace ps
}  // namespace tips
