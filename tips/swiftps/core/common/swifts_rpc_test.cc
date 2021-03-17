#include "swiftps/core/common/naive_rpc.h"

#include <mpi.h>

#include <chrono>
#include <thread>

namespace swifts {

using namespace std::chrono_literals;

void TestRpc() {
  RpcServer server;
  RpcCallback callback = [&server](const RpcMsgHead& head, NaiveBuffer& buffer) {
    std::this_thread::sleep_for(500ms);
    if (head.message_type == RpcMsgType::REQUEST) {
      LOG(INFO) << "server " << mpi_rank() << " get a request";

      RpcMsgHead response_head;
      response_head.server_id = mpi_rank();
      response_head.client_id = head.client_id;
      response_head.service   = head.service;
      response_head.request   = head.request;

      int v;
      std::string msg;
      buffer >> v >> msg;

      CHECK_EQ(v, mpi_rank());
      CHECK_EQ(msg, "hello node" + std::to_string(mpi_rank()));
      LOG(INFO) << mpi_rank() << " get message from master: " << msg;

      NaiveBuffer write_buf;
      write_buf << mpi_rank();
      server.SendResponse(response_head, write_buf);
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
    RpcCallback callback = [&server](RpcMsgHead head, NaiveBuffer& buf) {};

    LOG(INFO) << "master send request...";
    {
      NaiveBuffer writebuf;
      writebuf << 1;
      writebuf << std::string("hello node1");
      server.SendRequest(1, service, writebuf, callback);
    }
    {
      NaiveBuffer writebuf;
      writebuf << 2;
      writebuf << std::string("hello node2");
      server.SendRequest(2, service, writebuf, callback);
    }
  }

  mpi_barrier();
  std::this_thread::sleep_for(2000ms);
  server.Finalize();
}

}  // namespace swifts

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  swifts::TestRpc();

  MPI_Finalize();
  return 0;
}
