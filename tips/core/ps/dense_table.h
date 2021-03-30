#pragma once

#include <absl/types/span.h>
#include "tips/core/common/common.h"
#include "tips/core/ps/table.h"

namespace tips {
namespace ps {

template <typename T>
struct alignas(64) DenseTableShard {
  using value_type = T;

  absl::Span<T> data() { return absl::Span<T>(data_.data(), data_.size()); }
  absl::Span<const T> data() const { return absl::Span<T>(data_.data(), data_.size()); }
  void SetData(absl::Span<const T> d) { data_.assign(d.data(), d.size()); }
  void SetData(std::vector<T>&& d) { data_ = std::move(d); }

 private:
  std::vector<T> data_;
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
  using key_type   = size_t;
  using value_type = T;
  using shard_type = DenseTableShard<T>;

  DenseTable() { local_shards_.reset(new shard_type[local_shard_num()]); }

  shard_type& local_shard(int i) { return local_shards_[i]; }
  shard_type& shard(int i) { return local_shard(Table::Global().shard(i).local_shard_id); }
  size_t shard_boundary(int shard_id) const { return boundaries_[shard_id]; }
  size_t size() const { return size_; }

  void Resize(size_t size) {
    mpi_barrier();
    size_ = size;
    boundaries_.resize(shard_num() + 1);

    for (int i = 0; i <= shard_num(); i++) {
      boundaries_[i] = i * size / shard_num();
    }

    for (int i = 0; i < local_shard_num(); i++) {
      int shard_id = Table::Global().local_shard(i).shard_id;
      std::vector<T> new_data(boundaries_[shard_id + 1] - boundaries_[shard_id]);
      local_shards_[i].ResetData(new_data);
    }

    mpi_barrier();
  }

  template <typename Func>
  void ForEach(Func&& func) {
    mpi_barrier();

    for (int i = 0; i < local_shard_num(); i++) {
      value_type* data = &local_shard(i).data()[0];
      int shard_id     = Table::Global().local_shard(i).shard_id;
      size_t first     = boundaries_[shard_id];
      size_t last      = boundaries_[shard_id + 1];
      CHECK_EQ(local_shard(i).data.size(), last - first);

      client_channel().Write([data, first, &func] { LOG(FATAL) << "Not implemented"; });
    }
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
