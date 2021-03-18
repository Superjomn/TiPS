#include "tips/core/common/naive_rpc.h"
#include "tips/core/message/test0_generated.h"

#include <mpi.h>

#include <chrono>
#include <thread>

namespace tips {

using namespace std::chrono_literals;  // NOLINT
using namespace test::test_message0;   // NOLINT

void UseFlats() {
  flatbuffers::FlatBufferBuilder builder;
  auto hello = builder.CreateString("hello world");

  MessageRequestBuilder bb(builder);
  bb.add_greet(hello);
  bb.add_v(13);
  auto bb_out = bb.Finish();
  builder.Finish(bb_out);

  LOG(INFO) << "buffer pointer: " << builder.GetBufferPointer();
  int size = builder.GetSize();
  LOG(INFO) << "size: " << size;

  auto read = flatbuffers::GetRoot<MessageRequest>(builder.GetBufferPointer());
  LOG(INFO) << "greet: " << read->greet()->str();
  LOG(INFO) << "v: " << read->v();
}

void TestRpc() {
  RpcServer server;
  RpcCallback2 callback = [&server](const RpcMsgHead& head, uint8_t* buffer) {
    CHECK(buffer);
    std::this_thread::sleep_for(500ms);
    if (head.message_type == RpcMsgType::REQUEST) {
      LOG(INFO) << "server " << mpi_rank() << " get a request";

      RpcMsgHead response_head;
      response_head.server_id = mpi_rank();
      response_head.client_id = head.client_id;
      response_head.service   = head.service;
      response_head.request   = head.request;

      auto message = flatbuffers::GetRoot<MessageRequest>(buffer);

      int v    = message->v();
      auto msg = message->greet()->str();

      CHECK_EQ(v, mpi_rank());
      CHECK_EQ(msg, "hello node" + std::to_string(mpi_rank()));
      LOG(INFO) << mpi_rank() << " get message from master: " << msg;

      flatbuffers::FlatBufferBuilder write_builder;
      auto write_message_builder = MessageResponseBuilder(write_builder);
      write_message_builder.add_from_rank(mpi_rank());
      auto end = write_message_builder.Finish();
      write_builder.Finish(end);

      server.SendResponse(response_head, write_builder);
    }

    if (head.message_type == RpcMsgType::RESPONSE) {
      LOG(INFO) << "server " << mpi_rank() << " get a response";
    }
  };

  CHECK_EQ(mpi_size(), 3);
  auto* service = server.AddService(callback);

  LOG(INFO) << "to Intialize server";
  server.Initialize();
  mpi_barrier();

  if (mpi_rank() == 0) {
    RpcCallback2 callback = [&server](RpcMsgHead head, uint8_t* buf) {};

    LOG(INFO) << "master send request...";
    {
      flatbuffers::FlatBufferBuilder write_builder;
      auto hello = write_builder.CreateString("hello node1");

      auto request_message = MessageRequestBuilder(write_builder);
      request_message.add_v(1);
      request_message.add_greet(hello);
      auto end = request_message.Finish();
      write_builder.Finish(end);

      server.SendRequest(1, service, write_builder, callback);
    }
    {
      flatbuffers::FlatBufferBuilder write_builder;
      auto hello = write_builder.CreateString("hello node2");

      auto request_message = MessageRequestBuilder(write_builder);
      request_message.add_v(1);
      request_message.add_greet(hello);
      auto end = request_message.Finish();
      write_builder.Finish(end);

      server.SendRequest(1, service, write_builder, callback);
    }
  }

  mpi_barrier();
  std::this_thread::sleep_for(2000ms);
  server.Finalize();
}

}  // namespace tips

int main(int argc, char** argv) {
  tips::UseFlats();

  MPI_Init(&argc, &argv);

  tips::TestRpc();

  MPI_Finalize();
  return 0;
}
