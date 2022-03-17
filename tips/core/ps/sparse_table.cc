#include "tips/core/ps/sparse_table.h"

namespace tips {
namespace ps {

size_t SparseTable::size() const {
  size_t res = 0;
  for (int i = 0; i < shard_num(); i++) {
    auto &shard = local_shards_[i];
    res += shard.size();
  }
  return res;
}

}  // namespace ps
}  // namespace tips