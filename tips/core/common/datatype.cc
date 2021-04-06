#include "tips/core/common/datatype.h"
#include "tips/core/common/common.h"

namespace tips {

const char* DatatypeToStr(Datatype dtype) {
  switch (dtype) {
#define ___(x)        \
  case (Datatype::x): \
    return #x;

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

}  // namespace tips