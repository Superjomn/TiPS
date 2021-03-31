#pragma once

#include <absl/types/span.h>
#include "tips/core/common/common.h"
#include "tips/core/ps/table.h"

namespace tips {
namespace ps {

template <typename T>
struct alignas(64) DenseTableShard {
  using record_type = T;

  absl::Span<T> data() { return absl::Span<T>(data_.data(), data_.size()); }
  absl::Span<const T> data() const { return absl::Span<T>(data_.data(), data_.size()); }
  void SetData(absl::Span<const T> d) {
    data_.resize(d.size());
    for (int i = 0; i < d.size(); i++) {
      data_[i] = d[i];
    }
  }
  void SetData(std::vector<T>&& d) { data_ = std::move(d); }

  record_type* Lookup(size_t offset) {
    if (offset < data_.size()) {
      return &data_[offset];
    }
    return nullptr;
  }

  const record_type* Lookup(size_t offset) const {
    if (offset < data_.size()) {
      return &data_[offset];
    }
    return nullptr;
  }

 private:
  std::vector<record_type> data_;
};

/**
 * Dense map of parameters.
 * @tparam T the record type.
 *
 * The DenseTable partition the whole key space to multiple boundaries, each boundary range is bound to a dense shard.
 */
template <typename T>
class DenseTable : public Table {
 public:
  using key_type    = size_t;
  using record_type = T;
  using shard_type  = DenseTableShard<T>;

  DenseTable() {
    local_shards_.reset(new shard_type[local_shard_num()]);
    mpi_barrier();
  }
  ~DenseTable() { mpi_barrier(); }

  shard_type& local_shard(int i) { return local_shards_[i]; }
  const shard_type& local_shard(int i) const { return local_shards_[i]; }

  shard_type& shard(int i) { return local_shard(shard_info(i).local_shard_id); }
  const shard_type& shard(int i) const { return local_shard(shard_info(i).local_shard_id); }

  size_t shard_boundary(int shard_id) const { return boundaries_[shard_id]; }

  size_t size() const { return size_; }

  const record_type* Lookup(key_t key) const {
    int part     = size() / shard_num();
    int shard_id = key / part;
    return shard(shard_id).Lookup(key - shard_boundary(shard_id));
  }

  record_type* Lookup(key_t key) {
    int part     = size() / shard_num();
    int shard_id = key / part;
    return shard(shard_id).Lookup(key - shard_boundary(shard_id));
  }

  void Resize(size_t size) {
    mpi_barrier();
    size_ = size;
    boundaries_.resize(shard_num() + 1);

    CHECK_GT(shard_num(), 0);
    for (int i = 0; i <= shard_num(); i++) {
      boundaries_[i] = i * size / shard_num();
    }

    for (int i = 0; i < local_shard_num(); i++) {
      int shard_id = local_shard_info(i).shard_id;
      std::vector<T> new_data(boundaries_[shard_id + 1] - boundaries_[shard_id]);
      local_shards_[i].SetData(new_data);
    }

    mpi_barrier();
  }

  template <typename Func>
  void ForEach(Func&& func) {
    mpi_barrier();

    for (int i = 0; i < local_shard_num(); i++) {
      record_type* data = &local_shard(i).data()[0];
      int shard_id      = local_shard(i).shard_id;
      size_t first      = boundaries_[shard_id];
      size_t last       = boundaries_[shard_id + 1];
      CHECK_EQ(local_shard(i).data.size(), last - first);

      ParallelRunRange(last - first /*n*/, [data, first, &func](int tid, size_t start, size_t end) {
        for (size_t j = start; j != end; j++) {
          func(tid, first + j, data[j]);
        }
      });
    }

    mpi_barrier();
  }

  void Load(const std::string& path);
  void Save(const std::string& path) const;

 private:
  std::unique_ptr<shard_type[]> local_shards_;
  std::vector<size_t> boundaries_;
  size_t size_{};
};

}  // namespace ps
}  // namespace tips
