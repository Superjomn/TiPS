#include "tips/core/ps/sparse_table.h"
#include <absl/strings/str_format.h>

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

SparseTable::shard_t &SparseTable::local_shard(int shard_id) {
  CHECK_LT(shard_id, local_shard_num());
  return local_shards_[shard_id];
}

const SparseTable::shard_t &SparseTable::local_shard(int shard_id) const {
  CHECK_LT(shard_id, local_shard_num());
  return local_shards_[shard_id];
}

std::string SparseTable::Summary() const {
  std::stringstream ss;

  ss << absl::StrFormat("SparseTable[%s]:\n", std::string(GetTableName()));
  ss << "shard_id\tlocal_shard_id\n";

  for (int i = 0; i < shard_num(); i++) {
    ss << shard_info(i).shard_id << "\t" << shard_info(i).local_shard_id << "\n";
  }
  ss << "---------";

  return ss.str();
}

}  // namespace ps
}  // namespace tips
