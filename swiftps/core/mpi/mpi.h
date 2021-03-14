#pragma once
#include <glog/logging.h>
#include <mpi.h>
#include <stdint.h>

namespace swifts {

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

class MpiCtx {
 public:
  MpiCtx();

  MpiCtx& Global();

 private:
  std::vector<std::string> ip_table_;
};

template <typename T>
T mpi_allreduce(T x, MPI_Op op) {
  T ret;
  CHECK_EQ(MPI_Allreduce(&x, &ret, 1, mpi_type_trait<T>::type(), op, mpi_comm()), 0);
  return ret;
}

template <typename T>
void mpi_broadcast(T* p, int count, int root) {
  // TBD
}

int mpi_size() {
  int size;
  CHECK_EQ(MPI_Comm_size(mpi_comm(), &size), 0);
  return size;
}

}  // namespace swifts
