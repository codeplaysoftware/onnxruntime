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

#include "core/providers/sycl/softmax.h"

#include <CL/sycl.hpp>

#include "sycldnn/backend/sycl_blas_backend.h"
#include "sycldnn/softmax/launch.h"
#include "sycldnn/status.h"

namespace snn = sycldnn;
using Backend = snn::backend::SyclBLASBackend;

namespace onnxruntime {
namespace sycl {

// Registering VERSIONNED TYPED Kernels
#define REGISTER_VERSIONED_SOFTMAX_KERNEL_TYPE(T, start, end)     \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                        \
      Softmax,                                                    \
      kOnnxDomain,                                                \
      start,                                                      \
      end,                                                        \
      T,                                                          \
      kSyclExecutionProvider,                                     \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Softmax<T>);

#define REGISTER_SOFTMAX_KERNEL_TYPE(T, start)                    \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      Softmax,                                                    \
      kOnnxDomain,                                                \
      start,                                                      \
      T,                                                          \
      kSyclExecutionProvider,                                     \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Softmax<T>);

template <typename T>
Status Softmax<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* input_tensor = context->Input<Tensor>(0);
  Tensor* output_tensor = context->Output(0, input_tensor->Shape());

  if (output_tensor->Shape().Size() == 0)
    return Status::OK();

  const int64_t N = input_tensor->Shape()[0];
  const int64_t H = input_tensor->Shape()[1];
  const int64_t W = input_tensor->Shape()[2];
  const int64_t C = input_tensor->Shape()[3];

  const T* input_data = input_tensor->template Data<T>();
  T* output_data = output_tensor->template MutableData<T>();

  // Buffer USM Interop
  cl::sycl::buffer<T, 1> input_buffer{input_data,
                                      cl::sycl::range<1>{static_cast<size_t>(N * H * W * C)},
                                      {cl::sycl::property::buffer::context_bound{Queue()->get_context()},
                                       cl::sycl::property::buffer::use_host_ptr{}}};

  cl::sycl::buffer<T, 1> output_buffer{output_data,
                                       cl::sycl::range<1>{static_cast<size_t>(N * H * W * C)},
                                       {cl::sycl::property::buffer::context_bound{Queue()->get_context()},
                                        cl::sycl::property::buffer::use_host_ptr{}}};

  Backend backend{*Queue()};

  using DeviceMem = typename Backend::template internal_pointer_type<T>;

  auto input_ = DeviceMem(input_buffer, 0);
  auto output_ = DeviceMem(output_buffer, 0);

  DeviceMem workspace = backend.template allocate<T>(static_cast<size_t>(N * H * W));

  snn::softmax::SoftmaxParams params;
  params.channels = static_cast<int>(C);
  params.batch = static_cast<int>(N);
  params.rows = static_cast<int>(H);
  params.cols = static_cast<int>(W);

  snn::softmax::launch_forward<T, snn::softmax::Softmax>(input_, workspace, output_, params, backend);

  backend.template deallocate(workspace);
  backend.template deallocate(input_);
  backend.template deallocate(output_);

  return Status::OK();
}

// REGISTER KERNEL
REGISTER_VERSIONED_SOFTMAX_KERNEL_TYPE(float, 1, 10)
REGISTER_VERSIONED_SOFTMAX_KERNEL_TYPE(float, 11, 12)
REGISTER_SOFTMAX_KERNEL_TYPE(float, 13)

}  // namespace sycl
}  // namespace onnxruntime
