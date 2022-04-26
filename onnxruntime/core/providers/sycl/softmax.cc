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
#include "sycldnn/transpose/launch.h"
#include "sycldnn/status.h"

namespace snn = sycldnn;
using Backend = snn::backend::SyclBLASBackend;

namespace onnxruntime {
namespace sycl {

// Registering VERSIONNED TYPED Kernels
#define REGISTER_VERSIONED_SOFTMAX_KERNEL_TYPE(T, start, end)              \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                 \
      Softmax, kOnnxDomain, start, end, T, kSyclExecutionProvider,         \
      KernelDefBuilder().TypeConstraint("T",                               \
                                        DataTypeImpl::GetTensorType<T>()), \
      Softmax<T>);

#define REGISTER_SOFTMAX_KERNEL_TYPE(T, start)                              \
  ONNX_OPERATOR_TYPED_KERNEL_EX(Softmax, kOnnxDomain, start, T,             \
                                kSyclExecutionProvider,                     \
                                KernelDefBuilder().TypeConstraint(          \
                                    "T", DataTypeImpl::GetTensorType<T>()), \
                                Softmax<T>);

template <typename T>
Status Softmax<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* X = context->Input<Tensor>(0);
  Tensor* Y = context->Output(0, X->Shape());

  auto x_shape = X->Shape();

  // One or more dim values = 0, nothing to do
  if (x_shape.Size() == 0) {
    return Status::OK();
  }

  // Dimensionality & Axis of computation
  size_t rank = x_shape.NumDimensions();
  const size_t axis = static_cast<size_t>(HandleNegativeAxis(
      axis_, rank));  // Count a negative axis backward (ex : -1 -> (rank -1))

  if (axis >= rank) {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                  "Invalid Index (out of bounds [-rank,rank-1] )");
  }

  // Coerced dimensions of N Dimensional Input into 2D input
  // [d(0)*d(1)*..*d(axis-1) , d(axis)*d(axis+1)*..*d(rank-1)]
  const size_t NN = x_shape.SizeToDimension(axis);
  const size_t DD = x_shape.SizeFromDimension(axis);

  // Dimension variables for sycldnn
  int64_t N, C, H, W;
  N = C = H = W = 1;

  if (opset_ < 13 || (opset_ == 13 && axis == 0)) {
    // Input dimensions to be coerced into 2D [N,C] (C including reduction axis
    // axis)
    N = NN;
    C = DD;
  } else {
    // Input dimensions other than axis to be coerced contiguously into N & H,
    // and axis assigned to C
    N = NN;
    C = x_shape[axis];
    H = DD / C;
  }

  // SYCL BUFFERS
  const cl::sycl::buffer<T, 1> X_buffer =
      *X->template Ptr<cl::sycl::buffer<T, 1>>();
  cl::sycl::buffer<T, 1> Y_buffer =
      *Y->template MutablePtr<cl::sycl::buffer<T, 1>>();

  // SYCL DNN Backend
  Backend backend{*Queue()};

  using DeviceMem = typename Backend::template internal_pointer_type<T>;

  // Creating Device Pointers to Buffers
  auto x_data =
      DeviceMem(X_buffer, static_cast<size_t>(X->ByteOffset() / sizeof(T)));
  auto y_data =
      DeviceMem(Y_buffer, static_cast<size_t>(Y->ByteOffset() / sizeof(T)));

  // Transpose is only needed when axis is not the innermost when having
  // OpSet=13
  bool is_transpose_required = (opset_ == 13 && axis != (rank - 1));

  DeviceMem input, output;
  if (is_transpose_required) {
    input = backend.template allocate<T>(static_cast<size_t>(N * H * W * C));
    output = backend.template allocate<T>(static_cast<size_t>(N * H * W * C));

    // Performing input conversion from NCHW to NHWC (Porting C (a.k.a axis
    // dimension) to innermost position)
    const std::vector<int> input_sizes = {(int)N, (int)C, (int)H, (int)W};
    snn::transpose::convert_nchw_to_nhwc<T, Backend>(x_data, input, input_sizes,
                                                     backend);
  } else {
    input = x_data;
    output = y_data;
  }

  // Allocating Workspace Memory
  DeviceMem workspace =
      backend.template allocate<T>(static_cast<size_t>(N * H));

  // Setting Softmax parameters
  snn::softmax::SoftmaxParams params;
  params.channels = static_cast<int>(C);
  params.batch = static_cast<int>(N);
  params.rows = static_cast<int>(H);
  params.cols = static_cast<int>(W);

  // Launching softmax kernel
  snn::softmax::launch<T, snn::softmax::Forward>(input, workspace, output,
                                                 params, backend);

  if (is_transpose_required) {
    // Reverting the output back to NCHW layout
    const std::vector<int> output_sizes = {(int)N, (int)H, (int)W, (int)C};
    snn::transpose::convert_nhwc_to_nchw<T, Backend>(output, y_data,
                                                     output_sizes, backend);

    // Deallocating the input and output memory elements
    backend.template deallocate(input);
    backend.template deallocate(output);
  }

  // Deallocating the workspace memory
  backend.template deallocate(workspace);

  return Status::OK();
}

// REGISTER KERNEL
REGISTER_VERSIONED_SOFTMAX_KERNEL_TYPE(float, 1, 10)
REGISTER_VERSIONED_SOFTMAX_KERNEL_TYPE(float, 11, 12)
REGISTER_SOFTMAX_KERNEL_TYPE(float, 13)

}  // namespace sycl
}  // namespace onnxruntime
