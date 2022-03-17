# PS (the Parameter Server)
This directory contains most facilities of TiPS's distributed parameter server module.

## Design

We carefully design this module to make it works in large distributed tasks.
And much ideas are borrowed from SwiftSnais(one of my previous projects).

### Concepts

- SparseTable: the distributed vector map,
- PsServer: the server of the PS, which process the client's Push and Pull requests,
- PsClient: the client of the PS, which send the Push and Pull requests to PsServers,
- access method: the APIs for operation on a SparseTable,
- Route: the map of server and client nodes,
  - in the overall landscape of the cluster, we mark some of the nodes as PsServers, and some as PsWorkers, so
  we need a Route to help transfer the messages between these two kinds of nodes.

## APIs


### PULL(keys)

**PsClient::Pull**
```c++
struct PullRequest {
  struct Record {
    uint64_t key;
    Datatype dtype;
    int length{};
  };
  std::vector<Record> rcds;
};

struct PullResponse {
  
};

struct PullCache {
  struct Record {
      uint64_t key;
      AnyVec data;
  };
  std::unordered_map<uint64_t, Record> data;
};

// Pull data to cache.
bool PsClient::Global().Pull(const PullRequest& req, PullCache* cache);
bool PsClient::Global().PullAsync(const PullRequest& req, PullCache* cache, Callback done);
```

### PUSH(key, value)

```c++
struct PushRequest {
  struct Record {
    uint64_t key;
    Datatype dtype;
    int length{};
  };
  std::vector<Record> data;
};

struct PushResponse {
  // ...
};


// Pull data to cache.
bool PsClient::Global().Push(const PushRequest& req);
bool PsClient::Global().PushAsnyc(const PushRequest& req, Callback done);
```
