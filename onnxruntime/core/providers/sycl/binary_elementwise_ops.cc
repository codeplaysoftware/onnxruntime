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
  const Tensor* X1 = context->Input<Tensor>(0);
  const Tensor* X2 = context->Input<Tensor>(1);

  Tensor* Y = context->Output(0, X1->Shape());

  const T* X1_data = X1->template Data<T>();
  const T* X2_data = X2->template Data<T>();
  T* Y_data = Y->template MutableData<T>();

  size_t dataSize = Y->SizeInBytes() / sizeof(T);

  // Buffer USM Interop
  cl::sycl::buffer<T, 1> X1_buffer{X1_data,
                                   cl::sycl::range<1>{dataSize},
                                   {cl::sycl::property::buffer::context_bound{Queue()->get_context()},
                                    cl::sycl::property::buffer::use_host_ptr{}}};

  // SYCL DNN Backend
  auto queue = *Queue();
  Backend backend(queue);

  using DeviceMem = Backend::internal_pointer_type<T>;

  auto X1_ = DeviceMem(X1_buffer, 0);

  DeviceMem input = backend.template allocate<T>(dataSize);

  auto event = queue.submit([&](cl::sycl::handler& cgh) {
    auto buf = input.get_buffer();
    auto acc = buf.template get_access<cl::sycl::access::mode::discard_write>(cgh);
    cgh.copy(X2_data, acc);
  });
  event.wait_and_throw();

  snn::pointwise::launch<T, snn::pointwise::ResidualAdd, snn::pointwise::Forward, Backend>(X1_, input, dataSize, backend);

  event = queue.submit([&](cl::sycl::handler& cgh) {
    auto buf = input.get_buffer();
    auto acc = buf.template get_access<cl::sycl::access::mode::read>(cgh);
    cgh.copy(acc, Y_data);
  });
  event.wait_and_throw();

  backend.template deallocate(X1_);
  backend.template deallocate(input);

  return Status::OK();
}

REGISTER_VERSIONED_ADD_KERNELS_TYPED(float, 7, 12)
REGISTER_VERSIONED_ADD_KERNELS_TYPED(float, 13, 13)
REGISTER_ADD_KERNELS_TYPED(float, 14)

}  // namespace sycl
}  // namespace onnxruntime
