// Codeplay Software Ltd.

#include "core/framework/data_types_internal.h"
#include "core/providers/sycl/dropout.h"

namespace onnxruntime {
namespace sycl {

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Dropout,
    kOnnxDomain,
    1,
    12,
    kSyclExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::AllIEEEFloatTensorTypes())
        .TypeConstraint("T1", DataTypeImpl::AllIEEEFloatTensorTypes())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<bool>()),
    Dropout);

Status Dropout::ComputeInternal(OpKernelContext* context) const {
  //Get X_data
  const Tensor* X = context->Input<Tensor>(0);
  auto X_span = X->DataAsSpan<float>();

  if (X == nullptr)
    return Status(common::ONNXRUNTIME, common::FAIL, "X Input is not available.");

  const TensorShape& X_shape = X->Shape();
  const int64_t N = X_shape.Size();

  //Get Y_data
  auto Y = context->Output(0, X_shape);
  auto Y_span = Y->MutableDataAsSpan<float>();

  //Get mask_data
  auto mask = context->Output(1, X_shape);

  ORT_ENFORCE(Y != nullptr);

  ORT_ENFORCE(!mask || mask->Shape().Size() == N);

  //Get the ratio_data
  float ratio_data = default_ratio_;
  auto ratio = context->Input<Tensor>(1);
  if (ratio) {
    utils::MLTypeCallDispatcher<float> t_disp(ratio->GetElementType());
    t_disp.Invoke<GetRatioDataImpl>(ratio, ratio_data);
  }

  const Tensor* training_mode = context->Input<Tensor>(2);
  //Check for inference mode.
  if ((0 == ratio_data) || (training_mode == nullptr || *(training_mode->Data<bool>()) == false)) {
    const void* source = X->DataRaw();
    void* target = Y->MutableDataRaw();

    if (target != source) {
      auto data_transfer = this->GetDataTransfer();
      data_transfer->CopyTensor(*X, *Y);
    }

    // If mask is requested, return all 1s.
    if (mask != nullptr) {
      //   call to SYCL EP memset
      //   memset is supported from ComputeCpp 2.7 onwards
    }
  }

  return Status::OK();
}

}  // namespace sycl
}  // namespace onnxruntime
