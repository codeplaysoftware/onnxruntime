// Codeplay Software Ltd.

#include "core/providers/sycl/sycl_kernel.h"
#include "core/framework/data_types_internal.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/nn/conv_attributes.h"
#include "core/providers/sycl/sycl_fwd.h"

namespace onnxruntime {
namespace sycl {

template <typename T>
class Conv final : public SyclKernel {
 public:
  Conv(const OpKernelInfo& info) : SyclKernel(info), conv_attrs_(info) {
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  ConvAttributes conv_attrs_;
};
}  // namespace sycl
}  // namespace onnxruntime
