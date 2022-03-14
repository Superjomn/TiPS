#pragma once

#include <unordered_map>
#include <unordered_set>

#include "tips/core/common/common.h"
#include "tips/core/common/rwlock.h"

namespace tips {
namespace ps {

/**
 * This is a parameter cache in worker nodes.
 * @tparam KEY Type of the keys.
 * @tparam PARAM Type of the parameters.
 * @tparam GRAD Type of the gradients.
 */
template <typename KEY, typename PARAM, typename GRAD>
class WorkerParamCache {
 public:
  using key_t   = KEY;
  using param_t = PARAM;
  using grad_t  = GRAD;

  WorkerParamCache() {}

  void InitKeys(const std::unordered_set<key_t>& keys) {
    RwLockWriteGuard lk(rwlock_);
    for (auto& key : keys) {
      params_[key] = param_t();
      grads_[key]  = grad_t();
    }
  }

  void Clear() {
    RwLockWriteGuard lk(rwlock_);
    params_.clear();
    grads_.clear();
  }

  size_t size() const {
    RwLockReadGuard lk(rwlock_);
    return params_.size();
  }

 private:
  mutable RWLock rwlock_;
  std::unordered_map<key_t, param_t> params_;
  std::unordered_map<key_t, grad_t> grads_;
};

}  // namespace ps
}  // namespace tips
