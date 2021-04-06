#pragma once
#include <cstdint>

namespace tips {

#define TIPS_DATATYPE_FOREACH(__) __(kFp32) __(kFp64) __(kInt32) __(kInt64)

enum class Datatype {
#define ___(x) x,
  TIPS_DATATYPE_FOREACH(___)
#undef ___
};

const char* DatatypeToStr(Datatype dtype);

int DatatypeNumBytes(Datatype dtype);

}  // namespace tips
