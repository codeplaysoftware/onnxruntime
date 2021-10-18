// Codeplay Software Ltd.

#include "core/providers/sycl/sycl_kernel.h"
#include "core/framework/data_types_internal.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/nn/conv_attributes.h"
#include "core/providers/sycl/sycl_fwd.h"

namespace onnxruntime {
namespace sycl {

template <typename T>
class BatchNorm final : public SyclKernel {
 public:
  BatchNorm(const OpKernelInfo& info) : SyclKernel(info) {
    float temp;
    ORT_ENFORCE(info.GetAttr<float>("epsilon", &temp).IsOK());

    epsilon_ = static_cast<double>(temp);

    //TODO: need to add support for training mode, spatial_ and momentum_
    // leaving these as they are not used for now
    is_training_mode_ = (info.GetAttrOrDefault<int64_t>("training_mode", 0) == 1);
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  double epsilon_;
  int64_t spatial_ = 1;        // default as per spec
                               //   double momentum_;
  bool is_training_mode_ = 0;  //default as per spec
};

}  // namespace sycl
}  // namespace onnxruntime