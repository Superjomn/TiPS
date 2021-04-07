#include "tips/core/common/datatype.h"
#include "tips/core/common/common.h"

namespace tips {

const char* DatatypeToStr(Datatype dtype) {
  switch (dtype) {
#define ___(repr, type)  \
  case (Datatype::repr): \
    return #repr;

    TIPS_DATATYPE_FOREACH(___)

#undef ___

    default:
      LOG(FATAL) << "Not supported type";
  }
}

int DatatypeNumBytes(Datatype dtype) {
  switch (dtype) {
    case Datatype::kFp32:
      return sizeof(float);
    case Datatype::kFp64:
      return sizeof(double);
    case Datatype::kInt32:
      return sizeof(int32_t);
    case Datatype::kInt64:
      return sizeof(int64_t);
    default:
      LOG(FATAL) << "Not supported type";
  }

  return -1;
}

template <>
Datatype DatatypeTypetrait<int32_t>() {
  return Datatype::kInt32;
}
template <>
Datatype DatatypeTypetrait<int64_t>() {
  return Datatype::kInt64;
}
template <>
Datatype DatatypeTypetrait<float>() {
  return Datatype::kFp32;
}
template <>
Datatype DatatypeTypetrait<double>() {
  return Datatype::kFp64;
}
template <>
Datatype DatatypeTypetrait<int8_t>() {
  return Datatype::kInt8;
}
template <>
Datatype DatatypeTypetrait<uint8_t>() {
  return Datatype::kUInt8;
}

std::ostream& operator<<(std::ostream& os, Datatype dtype) {
  os << DatatypeToStr(dtype);
  return os;
}

}  // namespace tips