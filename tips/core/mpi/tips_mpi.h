#pragma once
#include <mpi.h>
#include <stdint.h>
#include <unistd.h>

#include "tips/core/common/logging.h"
#include "tips/core/common/naive_buffer.h"

namespace tips {

// MPI data types
// @{
template <class T>
struct mpi_type_trait {};

template <>
struct mpi_type_trait<double> {
  static MPI_Datatype type() { return MPI_DOUBLE; }
};

template <>
struct mpi_type_trait<float> {
  static MPI_Datatype type() { return MPI_FLOAT; }
};

template <>
struct mpi_type_trait<int32_t> {
  static MPI_Datatype type() { return MPI_INT; }
};

template <>
struct mpi_type_trait<uint32_t> {
  static MPI_Datatype type() { return MPI_UNSIGNED; }
};

template <>
struct mpi_type_trait<int64_t> {
  static MPI_Datatype type() { return MPI_LONG_LONG; }
};

template <>
struct mpi_type_trait<uint64_t> {
  static MPI_Datatype type() { return MPI_UNSIGNED_LONG_LONG; }
};

template <>
struct mpi_type_trait<long long> {
  static MPI_Datatype type() { return MPI_LONG_LONG; }
};

template <>
struct mpi_type_trait<unsigned long long> {
  static MPI_Datatype type() { return MPI_UNSIGNED_LONG_LONG; }
};
// @}

inline MPI_Comm mpi_comm() { return MPI_COMM_WORLD; }
int mpi_rank();

class MpiContext {
 public:
  MpiContext();

  // Get the ip address of this node.
  const std::string& ip() const { return ip_table_[mpi_rank()]; }
  // Get the ip address of a specific rank.
  const std::string& ip(int rank) const { return ip_table_[rank]; }

  bool IsInitialized();

  bool IsFinalized();

  static void Initialize(int* argc = nullptr, char*** argv = nullptr);
  static void Initialize(int argc, char** argv) { ZCHECK(MPI_Init(&argc, &argv)); }
  static void Finalize() { ZCHECK(MPI_Finalize()); }

  int get_rank() const {
    int rank{-1};
    ZCHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    return rank;
  }

  int get_size() const {
    int size{-1};
    ZCHECK(MPI_Comm_size(MPI_COMM_WORLD, &size));
    return size;
  }

  inline int rank() const {
    CHECK_NE(rank_, -1);
    return rank_;
  }

  inline int size() const {
    CHECK_GT(size_, 0);
    return size_;
  }

  static MpiContext& Global();

 private:
  int rank_{-1};
  int size_{-1};
  std::vector<std::string> ip_table_;
  bool initialized_{};
};

template <typename T>
T mpi_allreduce(T x, MPI_Op op) {
  T ret;
  CHECK_EQ(MPI_Allreduce(&x, &ret, 1, mpi_type_trait<T>::type(), op, mpi_comm()), 0);
  return ret;
}

template <typename T>
void mpi_broadcast(T* p, int count, int root) {
  NaiveBuffer wbuffer;
  int len = 0;

  if (mpi_rank() == root) {
    for (int i = 0; i < count; i++) {
      wbuffer << p[i];
    }
    len = wbuffer.size();
  }

  // broadcast from root to all other nodes
  MPI_Bcast(&len, 1, mpi_type_trait<int32_t>::type(), root, mpi_comm());

  if (len > 0 && wbuffer.size() == 0) {
    wbuffer.Require(len);
  }
  MPI_Bcast(reinterpret_cast<void*>(wbuffer.data()), len, MPI_BYTE, root, mpi_comm());

  // read the data
  NaiveBuffer rbuffer(static_cast<char*>(wbuffer.data()), wbuffer.size());
  for (int i = 0; i < count; i++) {
    rbuffer >> p[i];
  }
}

template <typename T>
void mpi_broadcast(T* p, int count, int root, int rank) {
  NaiveBuffer wbuffer;
  int len = 0;

  if (rank == root) {
    for (int i = 0; i < count; i++) {
      wbuffer << p[i];
    }
    len = wbuffer.size();
  }

  // broadcast from root to all other nodes
  MPI_Bcast(&len, 1, mpi_type_trait<int32_t>::type(), root, mpi_comm());

  if (len > 0 && wbuffer.size() == 0) {
    wbuffer.Require(len);
  }
  MPI_Bcast(reinterpret_cast<void*>(wbuffer.data()), len, MPI_BYTE, root, mpi_comm());

  // read the data
  NaiveBuffer rbuffer(static_cast<char*>(wbuffer.data()), wbuffer.size());
  for (int i = 0; i < count; i++) {
    rbuffer >> p[i];
  }
}

// inline void mpi_barrier() { MPI_Barrier(mpi_comm()); }

inline int mpi_size() { return MpiContext::Global().size(); }

std::string mpi_rank_repr();

void mpi_barrier(MPI_Comm comm = mpi_comm());
void mpi_barrier(MPI_Comm comm, int size);

}  // namespace tips

#define MPI_LOG LOG(INFO) << ::tips::mpi_rank_repr()
#define MPI_WARN LOG(INFO) << ::tips::mpi_rank_repr()
