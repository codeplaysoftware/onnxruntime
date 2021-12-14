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

#include "core/providers/sycl/pool.h"

#include <CL/sycl.hpp>

#include "sycldnn/backend/snn_backend.h"
#include "sycldnn/pooling/launch.h"
#include "sycldnn/pooling/operators.h"
#include "sycldnn/transpose/launch.h"
#include "sycldnn/helpers/padding.h"
#include "sycldnn/status.h"

namespace snn = sycldnn;
using Backend = snn::backend::SNNBackend;

namespace onnxruntime {
namespace sycl {

// Registering VERSIONNED TYPED Kernels
#define REGISTER_POOLING_VERSIONED_KERNEL_TYPED(T, op, pool_type, start, end) \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                    \
      op,                                                                     \
      kOnnxDomain,                                                            \
      start,                                                                  \
      end,                                                                    \
      T,                                                                      \
      kSyclExecutionProvider,                                                 \
      KernelDefBuilder()                                                      \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),             \
      Pool<T, pool_type>);

#define REGISTER_POOLING_KERNEL_TYPED(T, op, pool_type, start)    \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      op,                                                         \
      kOnnxDomain,                                                \
      start,                                                      \
      T,                                                          \
      kSyclExecutionProvider,                                     \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Pool<T, pool_type>);

template <typename T, typename PoolType>
Status Pool<T, PoolType>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* X = context->Input<Tensor>(0);
  const TensorShape& x_shape = X->Shape();
  const int64_t N = x_shape[0];

  size_t input_dims = x_shape.NumDimensions();
  ORT_RETURN_IF_NOT(input_dims >= 3, "Input dimension cannot be less than 3.");

  const int64_t C = input_dims > 1 ? x_shape[1] : 1;
  const int64_t H_in = input_dims > 2 ? x_shape[2] : 1;
  const int64_t W_in = input_dims > 3 ? x_shape[3] : 1;

  size_t pooling_dims = input_dims - 2;
  if (pooling_dims > 3) {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "Unsupported pooling size.");
  }
  if (!pool_attrs_.global_pooling) {
    ORT_RETURN_IF_NOT(pooling_dims == pool_attrs_.kernel_shape.size(),
                      "kernel_shape num_dims is not compatible with X num_dims.");
  }

  std::vector<int64_t> pads = pool_attrs_.pads;
  std::vector<int64_t> output_dims = pool_attrs_.SetOutputSize(x_shape, x_shape[1], &pads);
  TensorShape output_shape(output_dims);
  Tensor* Y = context->Output(0, output_shape);

  // Edge case: one or more dims with value of 0
  if (output_shape.Size() == 0)
    return Status::OK();

  const int64_t H_out = output_dims.size() > 2 ? output_shape[2] : 1;
  const int64_t W_out = output_dims.size() > 3 ? output_shape[3] : 1;

  snn::pooling::PoolingParams params;
  params.in_rows = static_cast<int>(H_in);
  params.in_cols = static_cast<int>(W_in);
  params.window_rows = pool_attrs_.global_pooling ? static_cast<int>(H_in) : static_cast<int>(pool_attrs_.kernel_shape[0]);
  params.window_cols = pool_attrs_.global_pooling ? static_cast<int>(W_in) : static_cast<int>(pool_attrs_.kernel_shape[1]);
  params.stride_rows = pool_attrs_.global_pooling ? 1 : static_cast<int>(pool_attrs_.strides[0]);
  params.stride_cols = pool_attrs_.global_pooling ? 1 : static_cast<int>(pool_attrs_.strides[1]);
  params.batch = static_cast<int>(N);
  params.channels = static_cast<int>(C);

  if (pool_attrs_.kernel_shape.size() == 1) {
    params.window_cols = 1;
  }
  if (pool_attrs_.strides.size() == 1) {
    params.stride_cols = 1;
  }

  if (pool_attrs_.auto_pad == AutoPadType::VALID) {
    params = snn::helpers::add_padding_to(params, snn::PaddingMode::VALID);
  } else if (pool_attrs_.auto_pad == AutoPadType::SAME_LOWER || pool_attrs_.auto_pad == AutoPadType::SAME_UPPER) {
    params = snn::helpers::add_padding_to(params, snn::PaddingMode::SAME);
  } else {
    params.pad_rows = pool_attrs_.global_pooling ? 0 : static_cast<int>(pads[0] + pads[pooling_dims - 1]);
    params.pad_cols = pool_attrs_.global_pooling ? 0 : static_cast<int>(pads[1] + pads[pooling_dims - 1]);

    if (pool_attrs_.ceil_mode) {
      params.out_rows = std::ceil((params.in_rows - params.window_rows + params.pad_rows) / (T)params.stride_rows) + 1;
      params.out_cols = std::ceil((params.in_cols - params.window_cols + params.pad_cols) / (T)params.stride_cols) + 1;
    } else {
      params.out_rows = std::floor((params.in_rows - params.window_rows + params.pad_rows) / (T)params.stride_rows + 1);
      params.out_cols = std::floor((params.in_cols - params.window_cols + params.pad_cols) / (T)params.stride_cols + 1);
    }
  }

  ORT_RETURN_IF_NOT(H_out == static_cast<int64_t>(params.out_rows) && W_out == static_cast<int64_t>(params.out_cols), "Output size mismatch detected.");

  // SYCL BUFFERS
  const cl::sycl::buffer<T, 1> X_buffer = *X->template Ptr<cl::sycl::buffer<T, 1>>();
  cl::sycl::buffer<T, 1> Y_buffer = *Y->template MutablePtr<cl::sycl::buffer<T, 1>>();

  // SYCL DNN Backend
  Backend backend{*Queue()};

  using DeviceMem = Backend::internal_pointer_type<T>;

  // Creating Device Pointers to Buffers
  auto x_data = DeviceMem(X_buffer, static_cast<size_t>(X->ByteOffset() / sizeof(T)));
  auto y_data = DeviceMem(Y_buffer, static_cast<size_t>(Y->ByteOffset() / sizeof(T)));

  // Allocating Intermediate Memory to perform computations in NHWC format through SYCL-DNN
  DeviceMem input, output;
  input = backend.template allocate<T>(static_cast<size_t>(N * C * H_in * W_in));
  output = backend.template allocate<T>(static_cast<size_t>(N * C * H_out * W_out));

  // Performing input conversion from NCHW to NHWC
  const std::vector<int> input_sizes = {(int)N, (int)C, (int)H_in, (int)W_in};
  snn::transpose::convert_nchw_to_nhwc<T, Backend>(x_data, input, input_sizes, backend);

  // Launch Pooling kernel
  if constexpr (PoolType::type == onnxruntime::PoolType::kAveragePool) {
    snn::pooling::launch<T, snn::pooling::Average, snn::pooling::Forward>(
        input, output, params, backend);
  } else if constexpr (PoolType::type == onnxruntime::PoolType::kMaxPool) {
    snn::pooling::launch<T, snn::pooling::Max, snn::pooling::Forward>(
        input, output, params, backend);
  }

  // Reverting the output back to NCHW layout
  const std::vector<int> output_sizes = {(int)N, (int)H_out, (int)W_out, (int)C};
  snn::transpose::convert_nhwc_to_nchw<T, Backend>(output, y_data, output_sizes, backend);

  //Deallocating all the memory elements used
  backend.template deallocate(input);
  backend.template deallocate(output);

  return Status::OK();
}

// REGISTER KERNEL
REGISTER_POOLING_VERSIONED_KERNEL_TYPED(float, MaxPool, MaxPool<1>, 1, 7)
REGISTER_POOLING_VERSIONED_KERNEL_TYPED(float, MaxPool, MaxPool<8>, 8, 9)
REGISTER_POOLING_KERNEL_TYPED(float, MaxPool, MaxPool<8>, 10)
REGISTER_POOLING_KERNEL_TYPED(float, MaxPool, MaxPool<8>, 11)
REGISTER_POOLING_KERNEL_TYPED(float, MaxPool, MaxPool<8>, 12)
REGISTER_POOLING_KERNEL_TYPED(float, GlobalMaxPool, MaxPool<1>, 1)

REGISTER_POOLING_VERSIONED_KERNEL_TYPED(float, AveragePool, AveragePool, 7, 9)
REGISTER_POOLING_VERSIONED_KERNEL_TYPED(float, AveragePool, AveragePool, 10, 10)
REGISTER_POOLING_KERNEL_TYPED(float, AveragePool, AveragePool, 11)
REGISTER_POOLING_KERNEL_TYPED(float, GlobalAveragePool, AveragePool, 1)

}  // namespace sycl
}  // namespace onnxruntime
