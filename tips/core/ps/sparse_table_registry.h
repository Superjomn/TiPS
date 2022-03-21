#pragma once

#include <absl/container/flat_hash_map.h>
#include <absl/hash/hash.h>
#include <absl/strings/string_view.h>
#include "tips/core/ps/sparse_table.h"

namespace tips {
namespace ps {

class SparseTableRegistry {
 public:
  //! Add a table to SparseTableRegistry
  //! We leave the `table_name` string argument for easier debugging.
  uint64_t AddTable(absl::string_view table_name, int num_nodes, int num_local_shards);

  static uint64_t ToTableId(absl::string_view table_name) { return absl::Hash<absl::string_view>()(table_name); }

  //! Get a table, return nullptr if not exist.
  SparseTable* GetTable(uint64_t table_id) const;

  SparseTable* GetTable(absl::string_view table_name) const;

  size_t size() const { return tables_.size(); }

  // The process-wide singleton.
  static SparseTableRegistry& Instance();

  //! Get the table name
  absl::string_view ToTableName(uint64_t table_id) const;

 private:
  absl::flat_hash_map<uint64_t, std::unique_ptr<SparseTable>> tables_;
};

}  // namespace ps
}  // namespace tips
