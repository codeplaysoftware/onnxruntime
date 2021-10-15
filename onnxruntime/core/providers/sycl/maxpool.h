// Codeplay Software Ltd.

#include "core/providers/sycl/sycl_kernel.h"
#include "core/framework/data_types_internal.h"
#include "core/framework/op_kernel.h"
#include "core/providers/sycl/sycl_fwd.h"
#include "core/providers/cpu/nn/pool_attributes.h"

namespace onnxruntime {
namespace sycl {
// Matmul OpKernel : C = A*B (matrix product)
template <typename T>
class MaxPool final : public SyclKernel {
 public:
  MaxPool(const OpKernelInfo& info) : SyclKernel(info), pool_attrs_(info, "MaxPool", GetStartVersion(info)) {
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  PoolAttributes pool_attrs_;

  static int GetStartVersion(const OpKernelInfo& info) {
    return info.node().SinceVersion();
  }
};
}  // namespace sycl
}  // namespace onnxruntime
