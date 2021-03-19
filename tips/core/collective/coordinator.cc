#include "tips/core/collective/coordinator.h"

namespace tips {

bool IncreTensorCount(MessageTable& table, RequestMessage&& msg, int mpi_size) {
  auto name       = msg.msg().tensor_name()->str();
  auto table_iter = table.find(name);
  if (table_iter == table.end()) {
    table.emplace(name, std::vector<RequestMessage>({std::move(msg)}));
    table_iter = table.find(name);
  } else {
    table_iter->second.push_back(std::move(msg));
  }

  return table_iter->second.size() == mpi_size;
}

}  // namespace tips