// Codeplay Software Ltd.

#include "core/providers/sycl/sycl_kernel.h"
#include "core/framework/data_types_internal.h"
#include "core/framework/op_kernel.h"
#include "core/providers/sycl/sycl_fwd.h"
#include "core/providers/cpu/nn/pool_base.h"

namespace onnxruntime {
namespace sycl {

template <typename T, typename PoolType>
class Pool final : public SyclKernel, public PoolBase {
 public:
  Pool(const OpKernelInfo& info) : SyclKernel(info), PoolBase(info) {}

  Status ComputeInternal(OpKernelContext* context) const override;
};

}  // namespace sycl
}  // namespace onnxruntime
