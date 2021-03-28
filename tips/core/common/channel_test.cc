#include "tips/core/common/channel.h"

#include <gtest/gtest.h>

#include <chrono>

#include "tips/core/common/managed_thread.h"

namespace tips {
using namespace std::chrono_literals;

TEST(Channel, one_writer_one_reader) {
  auto channel = MakeChannel<int>(3);

  ManagedThread reader, writer;

  writer.Start([channel] {
    for (int i = 0; i < 10; i++) {
      channel->Write(i);
    }
  });

  reader.Start([channel] {
    for (int i = 0; i < 10; i++) {
      int x;
      channel->Read(&x);
      LOG(INFO) << "read " << x;
    }
  });

  writer.Join();
  reader.Join();
}

TEST(Channel, multi_writer_reader) {
  auto channel = MakeChannel<std::pair<int /*tid*/, int /*val*/>>(5);

  const int num_writers = 5;
  const int num_readers = 3;

  std::vector<ManagedThread> readers(num_readers);
  std::vector<ManagedThread> writers(num_writers);

  for (int tid = 0; tid < num_readers; tid++) {
    readers[tid].Start([tid, channel] {
      std::pair<int, int> buf;
      for (int i = 0; i < 10; i++) {
        if (channel->Read(&buf)) {
          LOG(INFO) << "reader " << tid << " read from " << buf.first << " - " << buf.second;
        }
      }
    });
  }

  for (int tid = 0; tid < num_writers; tid++) {
    writers[tid].Start([tid, channel] {
      const int write_size = 3;
      std::pair<int, int> buf;
      buf.first  = tid;
      buf.second = write_size;
      for (int i = 0; i < 10; i++) {
        channel->Write(buf);
      }
    });
  }

  std::this_thread::sleep_for(500ms);
  channel->Close();

  for (auto& writer : writers) writer.Join();

  for (auto& reader : readers) reader.Join();
}

}  // namespace tips