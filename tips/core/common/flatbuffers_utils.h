#include <flatbuffers/flatbuffers.h>

#include "tips/core/common/common.h"

namespace tips {

/**
 * A wrapper to copy the RequestMessage flatbuffers object. It allocate and copy the data itself.
 */
template <typename FBS_T>
struct FBS_TypeBufferOwned {
  FBS_TypeBufferOwned() = default;
  FBS_TypeBufferOwned(const FBS_TypeBufferOwned& other) { Copy(other.buffer_, other.len_); }
  FBS_TypeBufferOwned(flatbuffers::DetachedBuffer&& buffer)
      : detached_buffer_(new flatbuffers::DetachedBuffer(std::move(buffer))) {}
  FBS_TypeBufferOwned& operator=(FBS_TypeBufferOwned&& other);

  /**
   * construct.
   * @param buffer address of the buffer.
   * @param len length of the buffer.
   * @param need_copy whether need to allocate memory and copy the data, false will just copy the pointer.
   */
  FBS_TypeBufferOwned(uint8_t* buffer, size_t len, bool need_copy = true) {
    if (need_copy) {
      Copy(buffer, len);
    } else {
      buffer_ = buffer;
      len_    = len;
    }
  }

  FBS_TypeBufferOwned(FBS_TypeBufferOwned&& other)
      : buffer_(other.buffer_), len_(other.len_), detached_buffer_(std::move(other.detached_buffer_)) {
    other.buffer_ = nullptr;
    other.len_    = 0;
  }

  const FBS_T& msg() const {
    CHECK(buffer_ || detached_buffer_);
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

template <typename FBS_T>
FBS_TypeBufferOwned<FBS_T>& FBS_TypeBufferOwned<FBS_T>::operator=(FBS_TypeBufferOwned&& other) {
  // Clear this.
  if (buffer_) {
    delete buffer_;
  }
  detached_buffer_.reset(nullptr);

  // Set the new data
  if (other.buffer_) {
    buffer_       = other.buffer_;
    len_          = other.len_;
    other.buffer_ = nullptr;
    other.len_    = 0;
  }

  if (other.detached_buffer_) {
    detached_buffer_ = std::move(other.detached_buffer_);
  }

  return *this;
}

}  // namespace tips
