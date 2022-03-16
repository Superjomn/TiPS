#pragma once

#include <absl/container/flat_hash_map.h>
#include <absl/container/inlined_vector.h>
#include <absl/hash/hash.h>
#include <absl/types/any.h>

#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <unordered_map>

#include "tips/core/common/common.h"
#include "tips/core/common/rwlock.h"
#include "tips/core/ps/table.h"

namespace tips {
namespace ps {

template <typename T>
size_t ToHashValue(const T &x) {
  return absl::Hash<T>{}(x);
}

using sparse_value_t = absl::variant<int32_t, float>;

/**
 * @brief shard of SparseTable
 *
 * a SparseTable contains several shards and the key-values will be
 * splitted to the shards.
 *
 * SparseTable use shards to improve the efficiency of Read-Write Lock.
 *
 * @param Key key
 * @param Value param can be pair of Param and Grad if AdaGrad is used
 *
 * Value's operation should be defined in AccessMethod
 */
struct alignas(64) SparseTableShard {
 public:
  using key_t   = uint64_t;
  using value_t = sparse_value_t;
  using map_t   = absl::flat_hash_map<key_t, value_t>;

  SparseTableShard() = default;

  template <typename Value>
  bool Find(const key_t &key, Value *&val) {
    RwLockReadGuard lock(rwlock_);
    auto it = data().find(key);
    if (it == data().end()) return false;
    val = &(absl::get<Value>(it->second));
    return true;
  }

  template <typename Value>
  bool Find(const key_t &key, Value &val) {
    RwLockReadGuard lock(rwlock_);
    auto it = data().find(key);
    if (it == data().end()) return false;
    val = absl::get<Value>(it->second);
    return true;
  }

  template <typename Value>
  void Assign(const key_t &key, const Value &val) {
    RwLockWriteGuard lock(rwlock_);
    data()[key] = val;
  }

  size_t size() const {
    RwLockReadGuard lock(rwlock_);
    return data().size();
  }

  void SetShardId(int x) {
    CHECK_GE(x, 0);
    shard_id_ = x;
  }

  int shard_id() const { return shard_id_; }

  /**
   * @brief output parameters to ostream
   * @warning should define value's output method first
   */
  template <typename Value>
  std::string __str__() const {
    std::stringstream ss;
    for (auto &item : data()) {
      ss << item.first << "\t";
      ss << absl::get<Value>(item.second) << "\n";
    }
    return ss.str();
  }

 protected:
  // not thread safe!
  map_t &data() { return data_; }
  const map_t &data() const { return data_; }

 private:
  map_t data_;
  int shard_id_ = -1;
  mutable RWLock rwlock_;
};  // struct SparseTableShard

/**
 * @brief Container of sparse parameters
 *
 * a SparseTable has several shards to split the storage and operation of
 * parameters.
 */
class SparseTable : public Table {
 public:
  using key_t   = uint64_t;
  using value_t = sparse_value_t;
  using param_t = value_t;
  using shard_t = SparseTableShard;

  SparseTable(int num_nodes, int num_local_shards) : Table(num_nodes, num_local_shards) {
    local_shards_.resize(local_shard_num());
  }

  //! Get a local shard.
  shard_t &local_shard(int shard_id) {
    CHECK_LT(shard_id, local_shard_num());
    return local_shards_[shard_id];
  }

  const shard_t &local_shard(int shard_id) const {
    CHECK_LT(shard_id, local_shard_num());
    return local_shards_[shard_id];
  }

  template <typename Value>
  bool Find(const key_t &key, Value *&val) {
    int local_shard_id = ToHashValue(key) % local_shard_num();
    return local_shard(local_shard_id).Find(key, val);
  }

  template <typename Value>
  bool Find(const key_t &key, Value &val) {
    int local_shard_id = ToHashValue(key) % local_shard_num();
    return local_shard(local_shard_id).Find(key, val);
  }

  void Assign(const key_t &key, const value_t &val) {
    int local_shard_id = ToShardId(key) % local_shard_num();
    local_shard(local_shard_id).Assign(key, val);
  }

  /**
   * output parameters to ostream
   */
  template <typename Value>
  std::string __str__() const {
    std::stringstream ss;
    for (int i = 0; i < shard_num(); i++) {
      ss << local_shard(i).template __str__<Value>();
    }
    return ss.str();
  }

  /**
   * output to a local file
   */
  template <typename Value>
  void WriteToFile(const std::string &path) {
    std::ofstream file(path.c_str(), std::ios::out);
    for (int i = 0; i < shard_num(); i++) {
      file << local_shard(i).__str__<Value>();
    }
  }

  //! Get number of elements in this table.
  size_t size() const {
    size_t res = 0;
    for (int i = 0; i < shard_num(); i++) {
      auto &shard = local_shards_[i];
      res += shard.size();
    }
    return res;
  }

  // TODO assign protected
  int ToShardId(const key_t &key) { return ToHashValue(key) % shard_num(); }

 private:
  absl::InlinedVector<shard_t, 4> local_shards_;
};  // class SparseTable

}  // namespace ps
}  // namespace tips
