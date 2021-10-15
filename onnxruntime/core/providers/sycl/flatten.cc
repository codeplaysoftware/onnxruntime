// Codeplay Software Ltd.

#include "core/providers/sycl/flatten.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace sycl {

#define REGISTER_Flatten_KERNEL_TYPED(T)                          \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                        \
      Flatten,                                                    \
      kOnnxDomain,                                                \
      1,                                                          \
      12,                                                         \
      T,                                                          \
      kSyclExecutionProvider,                                     \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Flatten);

Status Flatten::ComputeInternal(OpKernelContext* context) const {
  const Tensor* X = context->Input<Tensor>(0);
  const TensorShape& X_shape = X->Shape();

  auto axis = axis_;
  // Valid axis range is [-rank, rank] instead of [-rank, rank-1], add additional check to only handle neg axis case.
  if (axis < 0) {
    axis = HandleNegativeAxis(axis, X_shape.NumDimensions());  // handle negative and enforce axis is valid
  }

  ORT_ENFORCE(gsl::narrow_cast<int64_t>(X_shape.NumDimensions()) >= axis, "The rank of input tensor must be >= axis");

  Tensor* Y = context->Output(0, {X_shape.SizeToDimension(axis), X_shape.SizeFromDimension(axis)});
  //If source and target pointers are not equal (non-inplace operation), we need to copy the data.
  const void* source = X->DataRaw();
  void* target = Y->MutableDataRaw();

  if (target != source) {
    auto data_transfer = this->GetDataTransfer();
    data_transfer->CopyTensor(*X, *Y);
  }

  return Status::OK();
}

REGISTER_Flatten_KERNEL_TYPED(float);

}  // namespace sycl
}  // namespace onnxruntime
