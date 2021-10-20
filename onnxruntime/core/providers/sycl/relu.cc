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

#include "core/providers/sycl/relu.h"

#include <CL/sycl.hpp>

#include "sycldnn/backend/snn_backend.h"
#include "sycldnn/pointwise/launch.h"
#include "sycldnn/status.h"

namespace snn = sycldnn;
using Backend = snn::backend::SNNBackend;

namespace onnxruntime {
namespace sycl {

// Registering VERSIONNED TYPED Kernels
#define REGISTER_VERSIONED_RELU_KERNEL_TYPED(T, start, end)       \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                        \
      Relu,                                                       \
      kOnnxDomain,                                                \
      start,                                                      \
      end,                                                        \
      T,                                                          \
      kSyclExecutionProvider,                                     \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Relu<T>);

#define REGISTER_RELU_KERNEL_TYPED(T, start)                      \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      Relu,                                                       \
      kOnnxDomain,                                                \
      start,                                                      \
      T,                                                          \
      kSyclExecutionProvider,                                     \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Relu<T>);

template <typename T>
Status Relu<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* X1 = context->Input<Tensor>(0);
  Tensor* Y = context->Output(0, X1->Shape());

  size_t dataSize = Y->SizeInBytes() / sizeof(T);

  if (Y->Shape().Size() == 0)
    return Status::OK();

  const T* X1_data = X1->template Data<T>();
  T* Y_data = Y->template MutableData<T>();

  // Buffer USM Interop
  cl::sycl::buffer<T, 1> X1_buffer{X1_data,
                                   cl::sycl::range<1>{dataSize},
                                   {cl::sycl::property::buffer::context_bound{Queue()->get_context()},
                                    cl::sycl::property::buffer::use_host_ptr{}}};

  cl::sycl::buffer<T, 1> Y_buffer{Y_data,
                                  cl::sycl::range<1>{dataSize},
                                  {cl::sycl::property::buffer::context_bound{Queue()->get_context()},
                                   cl::sycl::property::buffer::use_host_ptr{}}};

  // SYCL DNN Backend
  Backend backend{*Queue()};
  using DeviceMem = Backend::internal_pointer_type<T>;

  auto X1_ = DeviceMem(X1_buffer, 0);  //Offset = 0
  auto Y_ = DeviceMem(Y_buffer, 0);

  // Launch kernel
  snn::pointwise::launch<float, snn::pointwise::Relu, snn::pointwise::Forward>(X1_, Y_, dataSize, backend);

  backend.template deallocate(X1_);
  backend.template deallocate(Y_);

  return Status::OK();
}

// REGISTER KERNEL
REGISTER_VERSIONED_RELU_KERNEL_TYPED(float, 6, 12)
REGISTER_VERSIONED_RELU_KERNEL_TYPED(float, 13, 13)
REGISTER_RELU_KERNEL_TYPED(float, 14)

}  // namespace sycl
}  // namespace onnxruntime
