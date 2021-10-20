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
#define REGISTER_VERSIONED_BATCHNORM_KERNEL_TYPED(T, start, end)  \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                        \
      BatchNormalization,                                         \
      kOnnxDomain,                                                \
      start,                                                      \
      end,                                                        \
      T,                                                          \
      kSyclExecutionProvider,                                     \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      BatchNorm<T>);

#define REGISTER_BATCHNORM_KERNEL_TYPED(T, start)                 \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      BatchNormalization,                                         \
      kOnnxDomain,                                                \
      start,                                                      \
      T,                                                          \
      kSyclExecutionProvider,                                     \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
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

  ORT_RETURN_IF_ERROR(BatchNormHelper::ValidateInputs(X, scale, B, mean, var, spatial_ == 1));
  //Training mode not supported
  if (is_training_mode_ == 1) {
    return Status(onnxruntime::common::ONNXRUNTIME, onnxruntime::common::NOT_IMPLEMENTED, "BatchNormalization Training mode not supported with SYCL EP");
  }

  const int64_t N = x_shape[0];
  const int64_t C = x_shape[1];
  const int64_t H = x_shape[2];
  const int64_t W = x_shape[3];

  // RAW DATA PTRs
  const T* Xdata = X->template Data<T>();
  const T* scaledata = scale->template Data<T>();
  const T* Bdata = B->template Data<T>();
  const T* meandata = mean->template Data<T>();
  const T* vardata = var->template Data<T>();
  T* Ydata = Y->template MutableData<T>();

  // Buffer USM Interop
  cl::sycl::buffer<T, 1> X_buffer{Xdata,
                                  cl::sycl::range<1>{static_cast<size_t>(N * C * H * W)},
                                  {cl::sycl::property::buffer::context_bound{Queue()->get_context()},
                                   cl::sycl::property::buffer::use_host_ptr{}}};

  cl::sycl::buffer<T, 1> scale_buffer{scaledata,
                                      cl::sycl::range<1>{static_cast<size_t>(C)},
                                      {cl::sycl::property::buffer::context_bound{Queue()->get_context()},
                                       cl::sycl::property::buffer::use_host_ptr{}}};

  cl::sycl::buffer<T, 1> B_buffer{Bdata,
                                  cl::sycl::range<1>{static_cast<size_t>(C)},
                                  {cl::sycl::property::buffer::context_bound{Queue()->get_context()},
                                   cl::sycl::property::buffer::use_host_ptr{}}};

  cl::sycl::buffer<T, 1> mean_buffer{meandata,
                                     cl::sycl::range<1>{static_cast<size_t>(C)},
                                     {cl::sycl::property::buffer::context_bound{Queue()->get_context()},
                                      cl::sycl::property::buffer::use_host_ptr{}}};

  cl::sycl::buffer<T, 1> var_buffer{vardata,
                                    cl::sycl::range<1>{static_cast<size_t>(C)},
                                    {cl::sycl::property::buffer::context_bound{Queue()->get_context()},
                                     cl::sycl::property::buffer::use_host_ptr{}}};

  cl::sycl::buffer<T, 1> Y_buffer{Ydata,
                                  cl::sycl::range<1>{static_cast<size_t>(N * C * H * W)},
                                  {cl::sycl::property::buffer::context_bound{Queue()->get_context()},
                                   cl::sycl::property::buffer::use_host_ptr{}}};

  // SYCL DNN Backend
  auto queue = *Queue();
  Backend backend{queue};

  using DeviceMem = Backend::internal_pointer_type<T>;

  auto X_ = DeviceMem(X_buffer, 0);
  auto scale_ = DeviceMem(scale_buffer, 0);
  auto B_ = DeviceMem(B_buffer, 0);
  auto mean_ = DeviceMem(mean_buffer, 0);
  auto var_ = DeviceMem(var_buffer, 0);
  auto Y_ = DeviceMem(Y_buffer, 0);

  DeviceMem input, output;
  input = backend.template allocate<T>(static_cast<size_t>(N * C * H * W));
  output = backend.template allocate<T>(static_cast<size_t>(N * C * H * W));

  const std::vector<int> input_sizes = {(int)N, (int)C, (int)H, (int)W};
  snn::transpose::convert_nchw_to_nhwc<T, Backend>(X_, input, input_sizes, backend);

  snn::batchnorm::BatchNormParams params;
  params.batch = static_cast<int>(N);
  params.rows = static_cast<int>(H);
  params.cols = static_cast<int>(W);
  params.channels = static_cast<int>(C);
  params.epsilon = static_cast<float>(epsilon_);

  snn::batchnorm::launch<T, Backend, snn::batchnorm::Inference>(input, B_, scale_, mean_, var_, output, params, backend);

  const std::vector<int> output_sizes = {(int)N, (int)H, (int)W, (int)C};
  snn::transpose::convert_nhwc_to_nchw<T, Backend>(output, Y_, output_sizes, backend);

  backend.template deallocate(X_);
  backend.template deallocate(scale_);
  backend.template deallocate(B_);
  backend.template deallocate(mean_);
  backend.template deallocate(var_);
  backend.template deallocate(input);
  backend.template deallocate(output);
  backend.template deallocate(Y_);

  return Status::OK();
}

// REGISTER KERNEL
REGISTER_VERSIONED_BATCHNORM_KERNEL_TYPED(float, 7, 8)
REGISTER_VERSIONED_BATCHNORM_KERNEL_TYPED(float, 9, 13)
REGISTER_VERSIONED_BATCHNORM_KERNEL_TYPED(float, 14, 14)
REGISTER_BATCHNORM_KERNEL_TYPED(float, 15)

}  //namespace sycl
}  // namespace onnxruntime
