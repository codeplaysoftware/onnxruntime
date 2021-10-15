// Codeplay Software Ltd.

#include "core/providers/sycl/sycl_kernel.h"
#include "core/framework/data_types_internal.h"
#include "core/framework/op_kernel.h"
#include "core/providers/sycl/sycl_fwd.h"

namespace onnxruntime {
namespace sycl {
// Add OpKernel C = A+B
template <typename T>
class Add final : public SyclKernel {
 public:
  Add(const OpKernelInfo& info) : SyclKernel(info) {
  }

  Status ComputeInternal(OpKernelContext* context) const override;
};
}  // namespace sycl
}  // namespace onnxruntime
