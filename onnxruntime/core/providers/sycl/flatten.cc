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

#include "core/providers/sycl/flatten.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace sycl {

#define REGISTER_VERSIONED_FLATTEN_KERNEL_TYPED(start, end)              \
  ONNX_OPERATOR_VERSIONED_KERNEL_EX(                                     \
      Flatten,                                                           \
      kOnnxDomain,                                                       \
      start,                                                             \
      end,                                                               \
      kSyclExecutionProvider,                                            \
      KernelDefBuilder()                                                 \
          .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()), \
      Flatten);

#define REGISTER_FLATTEN_KERNEL_TYPED(start)                             \
  ONNX_OPERATOR_KERNEL_EX(                                               \
      Flatten,                                                           \
      kOnnxDomain,                                                       \
      start,                                                             \
      kSyclExecutionProvider,                                            \
      KernelDefBuilder()                                                 \
          .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()), \
      Flatten);

Status Flatten::ComputeInternal(OpKernelContext* context) const {
  const Tensor* X = context->Input<Tensor>(0);
  const TensorShape& x_shape = X->Shape();

  auto axis = axis_;
  // Valid axis range is [-rank, rank] instead of [-rank, rank-1], add additional check to only handle neg axis case.
  if (axis < 0) {
    axis = HandleNegativeAxis(axis, x_shape.NumDimensions());  // handle negative and enforce axis is valid
  }

  ORT_ENFORCE(gsl::narrow_cast<int64_t>(x_shape.NumDimensions()) >= axis, "The rank of input tensor must be >= axis");

  Tensor* Y = context->Output(0, {x_shape.SizeToDimension(axis), x_shape.SizeFromDimension(axis)});
  //If source and target pointers are not equal (non-inplace operation), we need to copy the data.
  const void* source = X->DataRaw();
  void* target = Y->MutableDataRaw();

  //Copy input data to output if required
  if (target != source) {
    auto data_transfer = this->GetDataTransfer();
    data_transfer->CopyTensor(*X, *Y);
  }

  return Status::OK();
}

REGISTER_VERSIONED_FLATTEN_KERNEL_TYPED(1, 8)
REGISTER_VERSIONED_FLATTEN_KERNEL_TYPED(9, 10)
REGISTER_VERSIONED_FLATTEN_KERNEL_TYPED(11, 12)
REGISTER_FLATTEN_KERNEL_TYPED(13)

}  // namespace sycl
}  // namespace onnxruntime
