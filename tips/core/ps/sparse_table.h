#pragma once

#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <unordered_map>

#include "tips/core/common/common.h"
#include "tips/core/common/rwlock.h"

namespace tips {
namespace ps {

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
template <typename KEY, typename VALUE>
struct alignas(64) SparseTableShard {
 public:
  using key_t   = KEY;
  using value_t = VALUE;
  using map_t   = std::unordered_map<key_t, value_t>;

  SparseTableShard() { data().set_empty_key(std::numeric_limits<key_t>::max()); }

  bool Find(const key_t &key, value_t *&val) const {
    RwLockReadGuard lock(rwlock_);
    auto it = data().find(key);
    if (it == data().end()) return false;
    val = &(it->second);
    return true;
  }

  bool Find(const key_t &key, value_t &val) {
    RwLockReadGuard lock(rwlock_);
    auto it = data().find(key);
    if (it == data().end()) return false;
    val = it->second;
    return true;
  }

  void Assign(const key_t &key, const value_t &val) {
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
  friend std::ostream &operator<<(std::ostream &os, SparseTableShard &shard) {
    RwLockReadGuard lk(shard.rwlock_);
    for (auto &item : shard.data()) {
      os << item.first << "\t";
      os << item.second << std::endl;
    }
    return os;
  }

 protected:
  // not thread safe!
  map_t &data() { return data_; }

 private:
  map_t data_;
  int shard_id_ = -1;
  mutable RWLock rwlock_;
};  // struct SparseTableShard

/**
 * @brief container of sparse parameters
 *
 * a SparseTable has several shards to split the storage and operation of
 * parameters.
 */
template <typename Key, typename Value>
class SparseTable {
 public:
  typedef Key key_t;
  typedef Value value_t;
  typedef SparseTableShard<key_t, value_t> shard_t;

  SparseTable(int shard_num) {
    num_shards_ = shard_num;
    shards_.reset(new shard_t[num_shards()]);
  }

  shard_t &shard(int shard_id) { return shards_[shard_id]; }

  bool Find(const key_t &key, value_t *&val) {
    int shard_id = ToShardId(key);
    return shard(shard_id).Find(key, val);
  }

  bool Find(const key_t &key, value_t &val) {
    int shard_id = ToShardId(key);
    return shard(shard_id).Find(key, val);
  }

  void Assign(const key_t &key, const value_t &val) {
    int shard_id = ToShardId(key);
    shard(shard_id).Assign(key, val);
  }
  /**
   * output parameters to ostream
   */
  void Output() {
    for (int i = 0; i < num_shards(); i++) {
      std::cout << shard(i);
    }
  }
  /**
   * output to a local file
   */
  void Output(const std::string &path) {
    std::ofstream file(path.c_str(), std::ios::out);
    for (int i = 0; i < num_shards(); i++) {
      file << shard(i);
    }
  }

  size_t size() const {
    size_t res = 0;
    for (int i = 0; i < num_shards(); i++) {
      auto &shard = shards_[i];
      res += shard.size();
    }
    return res;
  }
  // TODO assign protected
  int ToShardId(const key_t &key) { return get_hash_code(key) % num_shards(); }
  int num_shards() const { return num_shards_; }

 private:
  std::unique_ptr<shard_t[]> shards_;
  int num_shards_ = 1;
};  // class SparseTable

}  // namespace ps
}  // namespace tips
