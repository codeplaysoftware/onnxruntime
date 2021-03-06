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

#include "core/providers/sycl/conv.h"

#include <CL/sycl.hpp>

#include "sycldnn/backend/snn_backend.h"
#include "sycldnn/conv2d/launch.h"
#include "sycldnn/conv2d/selector/default_selector.h"
#include "sycldnn/conv2d/workspace_size.h"
#include "sycldnn/binaryop/launch.h"
#include "sycldnn/binaryop/operators.h"
#include "sycldnn/transpose/launch.h"
#include "sycldnn/helpers/padding.h"
#include "sycldnn/status.h"

namespace snn = sycldnn;
using Backend = snn::backend::SNNBackend;

namespace onnxruntime {
namespace sycl {

// Registering Kernel
#define REGISTER_VERSIONED_CONV_KERNEL_TYPED(T, start, end)                \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                 \
      Conv, kOnnxDomain, start, end, T, kSyclExecutionProvider,            \
      KernelDefBuilder().TypeConstraint("T",                               \
                                        DataTypeImpl::GetTensorType<T>()), \
      Conv<T>);

#define REGISTER_CONV_KERNEL_TYPED(T, start)                                \
  ONNX_OPERATOR_TYPED_KERNEL_EX(Conv, kOnnxDomain, start, T,                \
                                kSyclExecutionProvider,                     \
                                KernelDefBuilder().TypeConstraint(          \
                                    "T", DataTypeImpl::GetTensorType<T>()), \
                                Conv<T>);

template <typename T>
Status Conv<T>::ComputeInternal(OpKernelContext* context) const {
  const auto* X = context->Input<Tensor>(0);
  const auto* W = context->Input<Tensor>(1);
  const auto* B =
      context->Input<Tensor>(2);  // optional. nullptr if not provided

  size_t x_dims = X->Shape().NumDimensions();
  size_t w_dims = W->Shape().NumDimensions();

  int64_t N, C, H_in, W_in, H_out, W_out;
  int64_t M, C_w, R, S;

  N = X->Shape()[0];

#ifndef USE_SYCL_NHWC
  ORT_RETURN_IF_ERROR(conv_attrs_.ValidateInputShape(X, W));
  C = x_dims > 1 ? X->Shape()[1] : 1;
  H_in = x_dims > 2 ? X->Shape()[2] : 1;
  W_in = x_dims > 3 ? X->Shape()[3] : 1;

  M = W->Shape()[0];
  C_w = w_dims > 1 ? W->Shape()[1] : 1;
  R = w_dims > 2 ? W->Shape()[2] : 1;
  S = w_dims > 3 ? W->Shape()[3] : 1;
#else
  // TODO: Implement ValidateInputs for SYCL EP for the NHWC layout
  C = x_dims > 3 ? X->Shape()[3] : 1;
  H_in = x_dims > 1 ? X->Shape()[1] : 1;
  W_in = x_dims > 2 ? X->Shape()[2] : 1;

  R = W->Shape()[0];
  S = w_dims > 1 ? W->Shape()[1] : 1;
  C_w = w_dims > 2 ? W->Shape()[2] : 1;
  M = w_dims > 3 ? W->Shape()[3] : 1;
#endif

  if (conv_attrs_.group != 1) {
    // This check has to come before checking Channel Dimensions
    return Status(common::ONNXRUNTIME, common::NOT_IMPLEMENTED,
                  "Convolution groups input not supported with SYCL EP");
  } else if (C != C_w) {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                  "Invalid Channel Dimensions");
  } else if (std::any_of(conv_attrs_.dilations.begin(),
                         conv_attrs_.dilations.end(),
                         [](int i) { return i != 1; })) {
    return Status(common::ONNXRUNTIME, common::NOT_IMPLEMENTED,
                  "Convolution dilations not supported with SYCL EP");
  } else if (x_dims > 4 && X->Shape().SizeFromDimension(4) != 1) {
    // We don't support 3D input unless the prod(D_3,...,D_N) == 1
    return Status(common::ONNXRUNTIME, common::NOT_IMPLEMENTED,
                  "Convolution 3D input not supported with SYCL EP");
  }
  for (size_t index = 2; index < conv_attrs_.pads.size(); index++) {
    if (conv_attrs_.pads[index - 2] != conv_attrs_.pads[index]) {
      return Status(
          common::ONNXRUNTIME, common::NOT_IMPLEMENTED,
          "Convolution does not support asymmetrical padding with SYCL EP");
    }
  }

  std::vector<int64_t> kernel_shape = {R, S};

#ifndef USE_SYCL_NHWC
  ORT_RETURN_IF_ERROR(conv_attrs_.ComputeKernelShape(W->Shape(), kernel_shape));
#endif

  std::vector<int64_t> pads(conv_attrs_.pads);
  if (pads.size() < 2 * kernel_shape.size()) {
    pads.resize(kernel_shape.size() * 2, 0);
  }
  std::vector<int64_t> dilations(conv_attrs_.dilations);
  if (dilations.size() < kernel_shape.size()) {
    dilations.resize(kernel_shape.size(), 1);
  }
  std::vector<int64_t> strides(conv_attrs_.strides);
  if (strides.size() < kernel_shape.size()) {
    strides.resize(kernel_shape.size(), 1);
  }

  std::vector<int64_t> Y_dims({N});
  std::vector<int64_t> input_shape({H_in});
#ifndef USE_SYCL_NHWC
  Y_dims.push_back(M);
  if (x_dims > 3) input_shape.push_back(W_in);
  ORT_RETURN_IF_ERROR(conv_attrs_.InferOutputShape((TensorShape)input_shape, kernel_shape, strides, dilations, pads, Y_dims));
#else
  if (x_dims > 2) input_shape.push_back(W_in);
  ORT_RETURN_IF_ERROR(conv_attrs_.InferOutputShape((TensorShape)input_shape, kernel_shape, strides, dilations, pads, Y_dims));
  Y_dims.push_back(M);
#endif

  Tensor* Y = context->Output(0, Y_dims);

  // Bail out early if one of the dimensions is zero.
  if (Y->Shape().Size() == 0) {
    return Status::OK();
  }

  size_t y_dims = Y->Shape().NumDimensions();

#ifndef USE_SYCL_NHWC
  H_out = y_dims > 2 ? Y->Shape()[2] : 1;
  W_out = y_dims > 3 ? Y->Shape()[3] : 1;
#else
  H_out = y_dims > 1 ? Y->Shape()[1] : 1;
  W_out = y_dims > 2 ? Y->Shape()[2] : 1;
#endif

  // SYCL BUFFERS
  const cl::sycl::buffer<T, 1> X_buffer =
      *X->template Ptr<cl::sycl::buffer<T, 1>>();
  const cl::sycl::buffer<T, 1> W_buffer =
      *W->template Ptr<cl::sycl::buffer<T, 1>>();
  cl::sycl::buffer<T, 1> Y_buffer =
      *Y->template MutablePtr<cl::sycl::buffer<T, 1>>();

  // SYCL DNN Backend
  Backend backend{*Queue()};

  using DeviceMem = Backend::internal_pointer_type<T>;

  // Creating a Conv selector instance
  auto selector = snn::conv2d::get_default_selector(Queue()->get_device());

  // Creating Device Pointers to Buffers
  auto x_data =
      DeviceMem(X_buffer, static_cast<size_t>(X->ByteOffset() / sizeof(T)));
  auto w_data =
      DeviceMem(W_buffer, static_cast<size_t>(W->ByteOffset() / sizeof(T)));
  auto y_data =
      DeviceMem(Y_buffer, static_cast<size_t>(Y->ByteOffset() / sizeof(T)));

  // Setting Conv parameters
  snn::conv2d::Conv2DParams params;
  params.channels = static_cast<int>(C);
  params.features = static_cast<int>(M);
  params.batch = static_cast<int>(N);
  params.in_rows = static_cast<int>(H_in);
  params.in_cols = static_cast<int>(W_in);
  params.window_rows = static_cast<int>(R);
  params.window_cols = static_cast<int>(S);
  params.stride_rows = static_cast<int>(strides[0]);
  params.stride_cols = static_cast<int>(strides[strides.size() - 1]);
  params.out_rows = static_cast<int>(H_out);
  params.out_cols = static_cast<int>(W_out);
  params.pad_rows = static_cast<int>(pads[0]);
  params.pad_cols = static_cast<int>(pads[pads.size() - 1]);

  // Declaring the workspace Memory
  DeviceMem workspace;

  // Querying the required workspace size
  auto new_size =
      snn::conv2d::query_workspace_size<snn::conv2d::conv_type::Forward>(
          params, *selector)
          .recommended_size;

  // Allocating workspace if required
  if (new_size > 0) {
    workspace = backend.template allocate<T>(new_size);
  }

#ifndef USE_SYCL_NHWC
  // First transpose the input feature map and filter weights to
  // the desired data layout

  // Allocating Intermediate Memory to perform computations in NHWC
  // format through SYCL-DNN
  DeviceMem input, weights, output;
  input = backend.template allocate<T>(static_cast<size_t>(N * C * H_in * W_in));
  weights = backend.template allocate<T>(static_cast<size_t>(M * C * R * S));

  const std::vector<int> input_sizes = {(int)N, (int)C, (int)H_in, (int)W_in};
  const std::vector<int> weight_sizes = {(int)M, (int)C, (int)R, (int)S};
  const std::vector<int> weight_permutations = {2, 3, 1, 0};

  // Performing input conversion from NCHW to NHWC for feature map
  snn::transpose::convert_nchw_to_nhwc<T, Backend>(x_data, input, input_sizes, backend);

  // Performing conversion from MCHW to HWCM for weights
  snn::transpose::launch<T, Backend>(w_data, weights, weight_sizes, weight_permutations, backend);

  output =
      backend.template allocate<T>(static_cast<size_t>(N * M * H_out * W_out));

  // Launching Conv kernel
  snn::conv2d::launch<T, snn::conv2d::conv_type::Forward>(
      input, weights, output, params, *selector, backend, workspace, new_size);

#else
  // Launching Conv kernel
  snn::conv2d::launch<T, snn::conv2d::conv_type::Forward>(
      x_data, w_data, y_data, params, *selector, backend, workspace, new_size);

#endif

  // Check if Bias Addition is required
  if (nullptr != B) {
    const cl::sycl::buffer<T, 1> B_buffer =
        *B->template Ptr<cl::sycl::buffer<T, 1>>();
    auto b_data =
        DeviceMem(B_buffer, static_cast<size_t>(B->ByteOffset() / sizeof(T)));

    // Setting Bias parameters
    snn::binaryop::BinaryParams bias_params;
    bias_params.lhs_items = static_cast<int>(H_out * W_out * N * M);
    bias_params.rhs_items = static_cast<int>(M);

#ifndef USE_SYCL_NHWC
    // Launching Bias addition kernel
    snn::binaryop::launch<T, snn::binaryop::Add>(output, b_data, output,
                                                 bias_params, backend);
  }

  // Reverting the output back to NCHW layout
  const std::vector<int> output_sizes = {(int)N, (int)H_out, (int)W_out,
                                         (int)M};
  snn::transpose::convert_nhwc_to_nchw<T, Backend>(output, y_data, output_sizes,
                                                   backend);

  // Deallocating all the memory elements used
  backend.template deallocate(input);
  backend.template deallocate(weights);
  backend.template deallocate(output);
#else
    // Launching Bias addition kernel
    snn::binaryop::launch<T, snn::binaryop::Add>(y_data, b_data, y_data, bias_params, backend);
  }
#endif

  backend.template deallocate(workspace);
  return Status::OK();
}

// REGISTER KERNEL
REGISTER_VERSIONED_CONV_KERNEL_TYPED(float, 1, 10)
REGISTER_CONV_KERNEL_TYPED(float, 11)

}  // namespace sycl
}  // namespace onnxruntime
