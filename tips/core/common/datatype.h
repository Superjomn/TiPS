#pragma once
#include <cstdint>
#include <iostream>

namespace tips {

#define TIPS_DATATYPE_FOREACH(__) \
  __(kFp32, float) __(kFp64, double) __(kInt32, int32_t) __(kInt64, int64_t) __(kInt8, int8_t) __(kUInt8, uint8_t)
#define TIPS_DATATYPE_FOREACH_T(__)

enum class Datatype {
#define ___(repr, type) repr,
  TIPS_DATATYPE_FOREACH(___)
#undef ___
};

const char* DatatypeToStr(Datatype dtype);

int DatatypeNumBytes(Datatype dtype);

template <typename T>
Datatype DatatypeTypetrait();

std::ostream& operator<<(std::ostream& os, Datatype dtype);

}  // namespace tips
