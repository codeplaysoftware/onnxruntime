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
  const Tensor* X = context->Input<Tensor>(0);
  Tensor* Y = context->Output(0, X->Shape());

  if (Y->Shape().Size() == 0) {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "Invalid Output Dimensions");
  }

  const int64_t N = X->Shape()[0];
  const int64_t C = X->Shape().NumDimensions() > 1 ? X->Shape()[1] : 1;
  const int64_t H = X->Shape().NumDimensions() > 2 ? X->Shape()[2] : 1;
  const int64_t W = X->Shape().NumDimensions() > 3 ? X->Shape()[3] : 1;

  bool is_transpose_required = H * W > 1;

  // SYCL BUFFERS
  const cl::sycl::buffer<T, 1> X_buffer = *X->template Ptr<cl::sycl::buffer<T, 1>>();
  cl::sycl::buffer<T, 1> Y_buffer = *Y->template MutablePtr<cl::sycl::buffer<T, 1>>();

  // SYCL DNN Backend
  Backend backend{*Queue()};

  using DeviceMem = typename Backend::template internal_pointer_type<T>;

  // Creating Device Pointers to Buffers
  auto x_data = DeviceMem(X_buffer, static_cast<size_t>(X->ByteOffset() / sizeof(T)));
  auto y_data = DeviceMem(Y_buffer, static_cast<size_t>(Y->ByteOffset() / sizeof(T)));

  DeviceMem input, output;
  if (is_transpose_required) {
    input = backend.template allocate<T>(static_cast<size_t>(N * H * W * C));
    output = backend.template allocate<T>(static_cast<size_t>(N * H * W * C));

    // Performing input conversion from NCHW to NHWC
    const std::vector<int> input_sizes = {(int)N, (int)C, (int)H, (int)W};
    snn::transpose::convert_nchw_to_nhwc<T, Backend>(x_data, input, input_sizes, backend);
  } else {
    input = x_data;
    output = y_data;
  }

  // Allocating Workspace Memory
  DeviceMem workspace = backend.template allocate<T>(static_cast<size_t>(N * H * W));

  // Setting Softmax parameters
  snn::softmax::SoftmaxParams params;
  params.channels = static_cast<int>(C);
  params.batch = static_cast<int>(N);
  params.rows = static_cast<int>(H);
  params.cols = static_cast<int>(W);

  // Launching softmax kernel
  snn::softmax::launch_forward<T, snn::softmax::Softmax>(input, workspace, output, params, backend);

  if (is_transpose_required) {
    // Reverting the output back to NCHW layout
    const std::vector<int> output_sizes = {(int)N, (int)H, (int)W, (int)C};
    snn::transpose::convert_nhwc_to_nchw<T, Backend>(output, y_data, output_sizes, backend);
  }

  return Status::OK();
}

// REGISTER KERNEL
REGISTER_VERSIONED_SOFTMAX_KERNEL_TYPE(float, 1, 10)
REGISTER_VERSIONED_SOFTMAX_KERNEL_TYPE(float, 11, 12)
REGISTER_SOFTMAX_KERNEL_TYPE(float, 13)

}  // namespace sycl
}  // namespace onnxruntime
