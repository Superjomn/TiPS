#include "tips/core/ps/sparse_table_registry.h"

namespace tips {
namespace ps {

uint64_t SparseTableRegistry::AddTable(absl::string_view table_name, int num_nodes, int num_local_shards) {
  uint64_t table_id = ToHashValue(table_name);
  CHECK(!tables_.count(table_id)) << "Duplicate add table [" << table_id << "]";
  auto* raw_table = new SparseTable(num_nodes, num_local_shards);
  raw_table->SetTableName(table_name);
  tables_.emplace(table_id, raw_table);
  return table_id;
}

SparseTable* SparseTableRegistry::GetTable(uint64_t table_id) const {
  auto it = tables_.find(table_id);
  if (it == tables_.end()) return nullptr;
  return it->second.get();
}

SparseTable* SparseTableRegistry::GetTable(absl::string_view table_name) const {
  return GetTable(ToHashValue(table_name));
}

SparseTableRegistry& SparseTableRegistry::Instance() {
  static SparseTableRegistry x;
  return x;
}

absl::string_view SparseTableRegistry::ToTableName(uint64_t table_id) const {
  auto it = tables_.find(table_id);
  return it == tables_.end() ? absl::string_view() : it->second->GetTableName();
}

}  // namespace ps
}  // namespace tips