#pragma once

#ifndef LIKELY
#define likely(x) __builtin_expect(!!(x), 1)
#endif

#ifndef UNLIKELY
#define unlikely(x) __builtin_expect(!!(x), 0)
#endif

// A macro to disallow the copy constructor and operator= functions
// This is usually placed in the private: declarations for a class.
#define SWIFTS_DISALLOW_COPY_AND_ASSIGN(TypeName) \
  TypeName(const TypeName&) = delete;             \
  void operator=(const TypeName&) = delete;

// Compiler attributes
#if (defined(__GNUC__) || defined(__APPLE__)) && !defined(SWIG)
// Compiler supports GCC-style attributes
#define SWIFTS_ATTRIBUTE_NORETURN __attribute__((noreturn))
#define SWIFTS_ATTRIBUTE_ALWAYS_INLINE __attribute__((always_inline))
#define SWIFTS_ATTRIBUTE_NOINLINE __attribute__((noinline))
#define SWIFTS_ATTRIBUTE_UNUSED __attribute__((unused))
#define SWIFTS_ATTRIBUTE_COLD __attribute__((cold))
#define SWIFTS_ATTRIBUTE_WEAK __attribute__((weak))
#define SWIFTS_PACKED __attribute__((packed))
#define SWIFTS_MUST_USE_RESULT __attribute__((warn_unused_result))
#define SWIFTS_PRINTF_ATTRIBUTE(string_index, first_to_check) \
  __attribute__((__format__(__printf__, string_index, first_to_check)))
#define SWIFTS_SCANF_ATTRIBUTE(string_index, first_to_check) \
  __attribute__((__format__(__scanf__, string_index, first_to_check)))
#elif defined(_MSC_VER)
// Non-GCC equivalents
#define SWIFTS_ATTRIBUTE_NORETURN __declspec(noreturn)
#define SWIFTS_ATTRIBUTE_ALWAYS_INLINE __forceinline
#define SWIFTS_ATTRIBUTE_NOINLINE
#define SWIFTS_ATTRIBUTE_UNUSED
#define SWIFTS_ATTRIBUTE_COLD
#define SWIFTS_ATTRIBUTE_WEAK
#define SWIFTS_MUST_USE_RESULT
#define SWIFTS_PACKED
#define SWIFTS_PRINTF_ATTRIBUTE(string_index, first_to_check)
#define SWIFTS_SCANF_ATTRIBUTE(string_index, first_to_check)
#else
// Non-GCC equivalents
#define SWIFTS_ATTRIBUTE_NORETURN
#define SWIFTS_ATTRIBUTE_ALWAYS_INLINE
#define SWIFTS_ATTRIBUTE_NOINLINE
#define SWIFTS_ATTRIBUTE_UNUSED
#define SWIFTS_ATTRIBUTE_COLD
#define SWIFTS_ATTRIBUTE_WEAK
#define SWIFTS_MUST_USE_RESULT
#define SWIFTS_PACKED
#define SWIFTS_PRINTF_ATTRIBUTE(string_index, first_to_check)
#define SWIFTS_SCANF_ATTRIBUTE(string_index, first_to_check)
#endif
