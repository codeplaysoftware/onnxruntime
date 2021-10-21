/*
 * Copyright Codeplay Software Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "core/framework/data_types_internal.h"
#include "core/providers/sycl/dropout.h"

namespace onnxruntime {
namespace sycl {

#define REGISTER_VERSIONED_DROPOUT_KERNEL_TYPED(T, start, end)           \
  ONNX_OPERATOR_VERSIONED_KERNEL_EX(                                     \
      Dropout,                                                           \
      kOnnxDomain,                                                       \
      start,                                                             \
      end,                                                               \
      kSyclExecutionProvider,                                            \
      KernelDefBuilder()                                                 \
          .TypeConstraint("T", DataTypeImpl::AllIEEEFloatTensorTypes())  \
          .TypeConstraint("T1", DataTypeImpl::AllIEEEFloatTensorTypes()) \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<bool>()),    \
      Dropout);

#define REGISTER_DROPOUT_KERNEL_TYPED(T, start)                          \
  ONNX_OPERATOR_KERNEL_EX(                                               \
      Dropout,                                                           \
      kOnnxDomain,                                                       \
      start,                                                             \
      kSyclExecutionProvider,                                            \
      KernelDefBuilder()                                                 \
          .TypeConstraint("T", DataTypeImpl::AllIEEEFloatTensorTypes())  \
          .TypeConstraint("T1", DataTypeImpl::AllIEEEFloatTensorTypes()) \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<bool>()),    \
      Dropout);

Status Dropout::ComputeInternal(OpKernelContext* context) const {
  //Get X_data
  const Tensor* X = context->Input<Tensor>(0);

  if (X == nullptr)
    return Status(common::ONNXRUNTIME, common::FAIL, "X Input is not available.");

  const TensorShape& X_shape = X->Shape();
  const int64_t N = X_shape.Size();

  //Get Y_data
  auto Y = context->Output(0, X_shape);

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

    //Copy input data to output if required
    if (target != source) {
      auto data_transfer = this->GetDataTransfer();
      data_transfer->CopyTensor(*X, *Y);
    }

    // If mask is requested, return all 1s.
    if (mask != nullptr) {
      //   TODO: add call to SYCL EP memset
      //   memset is supported from ComputeCpp 2.7 onwards
    }
    return Status::OK();
  } else {
    return Status(onnxruntime::common::ONNXRUNTIME, onnxruntime::common::NOT_IMPLEMENTED, "Dropout not implemented for training");
  }
}

REGISTER_VERSIONED_DROPOUT_KERNEL_TYPED(float, 12, 12)
REGISTER_DROPOUT_KERNEL_TYPED(float, 13)

}  // namespace sycl
}  // namespace onnxruntime
