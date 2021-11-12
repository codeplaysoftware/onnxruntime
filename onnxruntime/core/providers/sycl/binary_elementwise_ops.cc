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
#include "sycldnn/pointwise/launch.h"
#include "sycldnn/status.h"

namespace snn = sycldnn;
using Backend = snn::backend::SNNBackend;

namespace onnxruntime {

namespace sycl {

// Registering Kernels
#define REGISTER_VERSIONED_ADD_KERNELS_TYPED(T, start, end)       \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                        \
      Add,                                                        \
      kOnnxDomain,                                                \
      start,                                                      \
      end,                                                        \
      T,                                                          \
      kSyclExecutionProvider,                                     \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Add<T>);

#define REGISTER_ADD_KERNELS_TYPED(T, start)                      \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      Add,                                                        \
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

  // SYCL BUFFERS
  const cl::sycl::buffer<T, 1> A_buffer = *A->template Ptr<cl::sycl::buffer<T, 1>>();
  cl::sycl::buffer<T, 1>* B_buffer = const_cast<cl::sycl::buffer<float, 1>*>(B->template Ptr<cl::sycl::buffer<T, 1>>());
  cl::sycl::buffer<T, 1> Y_buffer = *Y->template MutablePtr<cl::sycl::buffer<T, 1>>();

  size_t count = A_buffer.size();  // Had to use .size() because of SYCL2020;

  // SYCL DNN Backend
  auto queue = *Queue();
  Backend backend(queue);

  using DeviceMem = Backend::internal_pointer_type<T>;

  // Creating Device Pointers to Buffers
  auto a_data = DeviceMem(A_buffer, static_cast<size_t>(A->ByteOffset() / sizeof(T)));

  // Copying contents of 2nd input to output memory
  // TODO : Copy operation to be removed for performance considerations
  // This is a temporary work-around until a proper SYCL DNN pointwise op is implemented
  // which takes both A and B const inputs and writes directly to the output Y
  queue.submit([&](cl::sycl::handler& cgh) {
    auto in_acc = B_buffer->template get_access<cl::sycl::access::mode::read>(cgh);
    auto out_acc = Y_buffer.template get_access<cl::sycl::access::mode::discard_write>(cgh);
    cgh.copy(in_acc, out_acc);
  });

  auto y_data = DeviceMem(Y_buffer, static_cast<size_t>(Y->ByteOffset() / sizeof(T)));

  // Launching add kernel
  // TODO: Add binary operation in SYCL-DNN to manage the ResidualAdd operation
  snn::pointwise::launch<T, snn::pointwise::ResidualAdd, snn::pointwise::Forward, Backend>(a_data, y_data, count, backend);

  return Status::OK();
}

REGISTER_VERSIONED_ADD_KERNELS_TYPED(float, 7, 12)
REGISTER_VERSIONED_ADD_KERNELS_TYPED(float, 13, 13)
REGISTER_ADD_KERNELS_TYPED(float, 14)

}  // namespace sycl
}  // namespace onnxruntime
