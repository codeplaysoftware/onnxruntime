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

#include "core/providers/sycl/batchnorm.h"
#include "core/providers/cpu/nn/batch_norm_helper.h"

#include <CL/sycl.hpp>

#include "sycldnn/backend/sycl_blas_backend.h"
#include "sycldnn/batchnorm/launch.h"
#include "sycldnn/transpose/launch.h"
#include "sycldnn/status.h"

namespace snn = sycldnn;
using Backend = snn::backend::SyclBLASBackend;

namespace onnxruntime {
namespace sycl {

// Registering Kernel
#define REGISTER_VERSIONED_BATCHNORM_KERNEL_TYPED(T, start, end)              \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                    \
      BatchNormalization, kOnnxDomain, start, end, T, kSyclExecutionProvider, \
      KernelDefBuilder().TypeConstraint("T",                                  \
                                        DataTypeImpl::GetTensorType<T>()),    \
      BatchNorm<T>);

#define REGISTER_BATCHNORM_KERNEL_TYPED(T, start)                           \
  ONNX_OPERATOR_TYPED_KERNEL_EX(BatchNormalization, kOnnxDomain, start, T,  \
                                kSyclExecutionProvider,                     \
                                KernelDefBuilder().TypeConstraint(          \
                                    "T", DataTypeImpl::GetTensorType<T>()), \
                                BatchNorm<T>);

template <typename T>
Status BatchNorm<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* X = context->Input<Tensor>(0);
  const Tensor* scale = context->Input<Tensor>(1);
  const Tensor* B = context->Input<Tensor>(2);
  const Tensor* mean = context->Input<Tensor>(3);
  const Tensor* var = context->Input<Tensor>(4);

  const TensorShape& x_shape = X->Shape();
  Tensor* Y = context->Output(0, x_shape);

  // Training mode not supported
  if (is_training_mode_ == 1) {
    return Status(
        common::ONNXRUNTIME, common::NOT_IMPLEMENTED,
        "BatchNormalization Training mode not supported with SYCL EP");
  } else if (spatial_ != 1) {
    return Status(
        common::ONNXRUNTIME, common::NOT_IMPLEMENTED,
        "BatchNormalization non-spatial input not supported with SYCL EP");
  } else if (x_shape.NumDimensions() > 4 && x_shape.SizeFromDimension(4) != 1) {
    // We don't support 3D input unless the prod(D_3,...,D_N) == 1
    return Status(common::ONNXRUNTIME, common::NOT_IMPLEMENTED,
                  "BatchNormalization 3D input not supported with SYCL EP");
  }

  size_t input_dims = x_shape.NumDimensions();
  int64_t N, C, H, W;
  N = x_shape[0];

#ifndef USE_SYCL_NHWC
  ORT_RETURN_IF_ERROR(
      BatchNormHelper::ValidateInputs(X, scale, B, mean, var, spatial_ == 1));
  C = input_dims > 1 ? x_shape[1] : 1;
  H = input_dims > 2 ? x_shape[2] : 1;
  W = input_dims > 3 ? x_shape[3] : 1;
#else
  // TODO: Implement ValidateInputs for SYCL EP for the NHWC layout
  C = input_dims > 3 ? x_shape[3] : 1;
  H = input_dims > 1 ? x_shape[1] : 1;
  W = input_dims > 2 ? x_shape[2] : 1;
#endif

  // SYCL BUFFERS
  const cl::sycl::buffer<T, 1> X_buffer =
      *X->template Ptr<cl::sycl::buffer<T, 1>>();
  const cl::sycl::buffer<T, 1> scale_buffer =
      *scale->template Ptr<cl::sycl::buffer<T, 1>>();
  const cl::sycl::buffer<T, 1> B_buffer =
      *B->template Ptr<cl::sycl::buffer<T, 1>>();
  const cl::sycl::buffer<T, 1> mean_buffer =
      *mean->template Ptr<cl::sycl::buffer<T, 1>>();
  const cl::sycl::buffer<T, 1> var_buffer =
      *var->template Ptr<cl::sycl::buffer<T, 1>>();
  cl::sycl::buffer<T, 1> Y_buffer =
      *Y->template MutablePtr<cl::sycl::buffer<T, 1>>();

  // SYCL DNN Backend
  Backend backend{*Queue()};

  using DeviceMem = Backend::internal_pointer_type<T>;

  // Creating Device Pointers to Buffers
  auto x_data =
      DeviceMem(X_buffer, static_cast<size_t>(X->ByteOffset() / sizeof(T)));
  auto scale_data = DeviceMem(
      scale_buffer, static_cast<size_t>(scale->ByteOffset() / sizeof(T)));
  auto b_data =
      DeviceMem(B_buffer, static_cast<size_t>(B->ByteOffset() / sizeof(T)));
  auto mean_data = DeviceMem(
      mean_buffer, static_cast<size_t>(mean->ByteOffset() / sizeof(T)));
  auto var_data =
      DeviceMem(var_buffer, static_cast<size_t>(var->ByteOffset() / sizeof(T)));
  auto y_data =
      DeviceMem(Y_buffer, static_cast<size_t>(Y->ByteOffset() / sizeof(T)));

  // Setting Batchnorm parameters
  snn::batchnorm::BatchNormParams params;
  params.batch = static_cast<int>(N);
  params.rows = static_cast<int>(H);
  params.cols = static_cast<int>(W);
  params.channels = static_cast<int>(C);
  params.epsilon = epsilon_;

#ifndef USE_SYCL_NHWC
  // Allocating Intermediate Memory to perform computations in NHWC format 
  // through SYCL-DNN
  DeviceMem input, output;
  input = backend.template allocate<T>(static_cast<size_t>(N * C * H * W));
  output = backend.template allocate<T>(static_cast<size_t>(N * C * H * W));

  // Performing input conversion from NCHW to NHWC
  const std::vector<int> input_sizes = {(int)N, (int)C, (int)H, (int)W};
  snn::transpose::convert_nchw_to_nhwc<T, Backend>(x_data, input, input_sizes,
                                                   backend);

  // Launching Batchnorm kernel
  snn::batchnorm::launch<T, Backend, snn::batchnorm::Inference>(
      input, b_data, scale_data, mean_data, var_data, output, params, backend);

  // Reverting the output back to NCHW layout
  const std::vector<int> output_sizes = {(int)N, (int)H, (int)W, (int)C};
  snn::transpose::convert_nhwc_to_nchw<T, Backend>(output, y_data, output_sizes,
                                                   backend);

  //Deallocate the memory elements used
  backend.template deallocate(input);
  backend.template deallocate(output);

#else
  // Launching Batchnorm kernel
  snn::batchnorm::launch<T, Backend, snn::batchnorm::Inference>(x_data, b_data, scale_data, mean_data, var_data, y_data, params, backend);
#endif

  return Status::OK();
}

// REGISTER KERNEL
REGISTER_VERSIONED_BATCHNORM_KERNEL_TYPED(float, 7, 8)
REGISTER_VERSIONED_BATCHNORM_KERNEL_TYPED(float, 9, 13)
REGISTER_VERSIONED_BATCHNORM_KERNEL_TYPED(float, 14, 14)
REGISTER_BATCHNORM_KERNEL_TYPED(float, 15)

}  // namespace sycl
}  // namespace onnxruntime
