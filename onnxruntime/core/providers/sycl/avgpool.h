// Codeplay Software Ltd.

#include "core/providers/sycl/sycl_kernel.h"
#include "core/framework/data_types_internal.h"
#include "core/framework/op_kernel.h"
#include "core/providers/sycl/sycl_fwd.h"
#include "core/providers/cpu/nn/pool_attributes.h"

namespace onnxruntime {
namespace sycl {
template <typename T>
class AveragePool final : public SyclKernel {
 public:
  AveragePool(const OpKernelInfo& info) : SyclKernel(info), pool_attrs_(info, "AveragePool", GetStartVersion(info)) {
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
