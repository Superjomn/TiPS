#pragma once
#include <flatbuffers/flatbuffers.h>

#include "tips/core/common/common.h"

namespace tips {

/**
 * A wrapper to copy the RequestMessage flatbuffers object. It allocate and copy the data itself.
 */
template <typename FBS_T>
struct FBS_TypeBufferOwned {
  using self_t = FBS_TypeBufferOwned<FBS_T>;

  FBS_TypeBufferOwned() {}
  FBS_TypeBufferOwned(const FBS_TypeBufferOwned& other) = delete;
  explicit FBS_TypeBufferOwned(flatbuffers::DetachedBuffer&& buffer)
      : detached_buffer_(new flatbuffers::DetachedBuffer(std::move(buffer))) {}

  FBS_TypeBufferOwned& operator=(FBS_TypeBufferOwned&& other) {
    MoveFrom(std::move(other));
    return *this;
  }
  FBS_TypeBufferOwned& operator=(const FBS_TypeBufferOwned& other) = delete;

  self_t& CopyFrom(const self_t& other) {
    CHECK(!HasData()) << "The destination of CopyFrom should be empty";
    if (other.HasData()) {
      buffer_ = new uint8_t[other.GetSize()];
      std::memcpy(buffer_, other.GetBuffer(), other.GetSize());
    }
    return *this;
  }

  self_t& MoveFrom(self_t&& other) {
    CHECK(!HasData()) << "The destination of MoveFrom should be empty";
    if (other.HasData()) {
      if (other.buffer_) {
        buffer_       = other.buffer_;
        len_          = other.len_;
        other.buffer_ = nullptr;
        other.len_    = 0;
      } else if (other.detached_buffer_) {
        detached_buffer_ = std::move(other.detached_buffer_);
      }
    }

    return *this;
  }

  bool HasData() const { return buffer_ || detached_buffer_; }

  /**
   * construct.
   * @param buffer address of the buffer.
   * @param len length of the buffer.
   * @param need_copy whether need to allocate memory and copy the data, false will just copy the pointer.
   */
  FBS_TypeBufferOwned(uint8_t* buffer, size_t len) {
    VLOG(5) << "Copy a FBS_TypeBufferOwned";
    Copy(buffer, len);
  }

  FBS_TypeBufferOwned(FBS_TypeBufferOwned&& other) { MoveFrom(std::move(other)); }

  const FBS_T& msg() const {
    CHECK(HasData());
    if (buffer_) {
      return *flatbuffers::GetRoot<FBS_T>(buffer_);
    } else {
      return *flatbuffers::GetRoot<FBS_T>(detached_buffer_->data());
    }
  }

  const uint8_t* GetBuffer() const { return detached_buffer_ ? detached_buffer_->data() : buffer_; }

  size_t GetSize() const { return detached_buffer_ ? detached_buffer_->size() : len_; }

  ~FBS_TypeBufferOwned() {
    if (buffer_) {
      free(buffer_);
    }
  }

 private:
  void Copy(const uint8_t* buffer, size_t len);

  uint8_t* buffer_{};
  size_t len_{};

  std::unique_ptr<flatbuffers::DetachedBuffer> detached_buffer_;
};

template <typename FBS_T>
void FBS_TypeBufferOwned<FBS_T>::Copy(const uint8_t* buffer, size_t len) {
  CHECK(!buffer_) << "FBS_TypeBufferOwned duplicate assign found";
  if (len == 0) return;
  len_    = len;
  buffer_ = reinterpret_cast<uint8_t*>(std::malloc(len));
  CHECK(buffer_) << "allocate memory failed";
  std::memcpy(buffer_, buffer, len);
}

}  // namespace tips
