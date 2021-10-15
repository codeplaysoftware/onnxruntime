// Codeplay Software Ltd.

#include "core/providers/sycl/sycl_kernel.h"
#include "core/framework/data_types_internal.h"
#include "core/framework/op_kernel.h"
#include "core/providers/sycl/sycl_fwd.h"

namespace onnxruntime {
namespace sycl {
// Matmul OpKernel : C = A*B (matrix product)
template <typename T>
class MatMul final : public SyclKernel {
 public:
  MatMul(const OpKernelInfo& info) : SyclKernel(info) {
  }

  Status ComputeInternal(OpKernelContext* context) const override;
};
}  // namespace sycl
}  // namespace onnxruntime
