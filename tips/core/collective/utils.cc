#include "tips/core/collective/utils.h"

namespace tips {
namespace collective {

MPI_Op CollectiveOpKindToMpiOp(CollectiveOpKind op) {
  switch (op) {
    case CollectiveOpKind::SUM:
      return MPI_SUM;
    case CollectiveOpKind::MAX:
      return MPI_MAX;
    case CollectiveOpKind::MIN:
      return MPI_MIN;
  }
}

}  // namespace collective
}  // namespace tips