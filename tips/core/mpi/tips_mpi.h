#pragma once
#include <mpi.h>
#include <stdint.h>

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

  static void Initialize(int argc, char** argv) { ZCHECK(MPI_Init(&argc, &argv)); }
  static void Finalize() { ZCHECK(MPI_Finalize()); }

  static MpiContext& Global();

 private:
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

inline void mpi_barrier() { MPI_Barrier(mpi_comm()); }

inline int mpi_size() {
  int size;
  ZCHECK(MPI_Comm_size(mpi_comm(), &size));
  return size;
}

}  // namespace tips
