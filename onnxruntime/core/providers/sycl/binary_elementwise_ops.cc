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

#include "core/providers/sycl/binary_elementwise_ops.h"

#include <CL/sycl.hpp>

#include "sycldnn/backend/snn_backend.h"
#include "sycldnn/binaryop/launch.h"
#include "sycldnn/binaryop/operators.h"
#include "sycldnn/status.h"

namespace snn = sycldnn;
using Backend = snn::backend::SNNBackend;

namespace onnxruntime {

namespace sycl {

// Registering Kernels
#define REGISTER_VERSIONED_ADD_KERNELS_TYPED(T, op, start, end)   \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                        \
      op,                                                         \
      kOnnxDomain,                                                \
      start,                                                      \
      end,                                                        \
      T,                                                          \
      kSyclExecutionProvider,                                     \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Add<T>);

#define REGISTER_ADD_KERNELS_TYPED(T, op, start)                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      op,                                                         \
      kOnnxDomain,                                                \
      start,                                                      \
      T,                                                          \
      kSyclExecutionProvider,                                     \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Add<T>);

template <typename T>
Status Add<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* A = context->Input<Tensor>(0);
  const Tensor* B = context->Input<Tensor>(1);

  Tensor* Y = context->Output(0, A->Shape());

  size_t count1 = A->SizeInBytes() / sizeof(T);
  size_t count2 = B->SizeInBytes() / sizeof(T);

  // SYCL BUFFERS
  const cl::sycl::buffer<T, 1> A_buffer = *A->template Ptr<cl::sycl::buffer<T, 1>>();
  const cl::sycl::buffer<T, 1> B_buffer = *B->template Ptr<cl::sycl::buffer<T, 1>>();
  cl::sycl::buffer<T, 1> Y_buffer = *Y->template MutablePtr<cl::sycl::buffer<T, 1>>();

  // SYCL DNN Backend
  Backend backend(*Queue());

  using DeviceMem = Backend::internal_pointer_type<T>;

  // Creating Device Pointers to Buffers
  auto a_data = DeviceMem(A_buffer, static_cast<size_t>(A->ByteOffset() / sizeof(T)));
  auto b_data = DeviceMem(B_buffer, static_cast<size_t>(B->ByteOffset() / sizeof(T)));
  auto y_data = DeviceMem(Y_buffer, static_cast<size_t>(Y->ByteOffset() / sizeof(T)));

  snn::binaryop::BinaryParams params;
  params.lhs_items = static_cast<int>(count1);
  params.rhs_items = static_cast<int>(count2);

  snn::binaryop::launch<T, snn::binaryop::Add>(a_data, b_data, y_data, params, backend);

  return Status::OK();
}

REGISTER_VERSIONED_ADD_KERNELS_TYPED(float, Add, 7, 12)
REGISTER_VERSIONED_ADD_KERNELS_TYPED(float, Add, 13, 13)
REGISTER_ADD_KERNELS_TYPED(float, Add, 14)

}  // namespace sycl
}  // namespace onnxruntime
