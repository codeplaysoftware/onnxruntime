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

#include "core/providers/sycl/matmul.h"
#include "core/providers/cpu/math/matmul_helper.h"

#include <CL/sycl.hpp>

#include "sycldnn/backend/sycl_blas_backend.h"
#include "sycldnn/status.h"

namespace snn = sycldnn;
using Backend = snn::backend::SyclBLASBackend;

namespace onnxruntime {
namespace sycl {

// Registering VERSIONNED TYPED Kernels
#define REGISTER_VERSIONED_MATMUL_KERNEL_TYPED(T, start, end)     \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                        \
      MatMul,                                                     \
      kOnnxDomain,                                                \
      start,                                                      \
      end,                                                        \
      T,                                                          \
      kSyclExecutionProvider,                                     \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      MatMul<T>);

#define REGISTER_MATMUL_KERNEL_TYPED(T, start)                    \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      MatMul,                                                     \
      kOnnxDomain,                                                \
      start,                                                      \
      T,                                                          \
      kSyclExecutionProvider,                                     \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      MatMul<T>);

template <typename T>
Status MatMul<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* A = context->Input<Tensor>(0);
  const Tensor* B = context->Input<Tensor>(1);

  MatMulComputeHelper helper;
  ORT_RETURN_IF_ERROR(helper.Compute(A->Shape(), B->Shape()));

  Tensor* Y = context->Output(0, helper.OutputShape());

  if (Y->Shape().Size() == 0)
    return Status::OK();

  const size_t M = static_cast<size_t>(helper.M());
  const size_t N = static_cast<size_t>(helper.N());
  const size_t K = static_cast<size_t>(helper.K());

  // No support for broadcasting or padding
  if (A->Shape().NumDimensions() != B->Shape().NumDimensions()) {
    return Status(common::ONNXRUNTIME, common::NOT_IMPLEMENTED, "Matmul padding/broadcasting not supported with SYCL EP");
  }

  // SYCL BUFFERS
  const cl::sycl::buffer<T, 1> A_buffer = *A->template Ptr<cl::sycl::buffer<T, 1>>();
  const cl::sycl::buffer<T, 1> B_buffer = *B->template Ptr<cl::sycl::buffer<T, 1>>();
  cl::sycl::buffer<T, 1> Y_buffer = *Y->template MutablePtr<cl::sycl::buffer<T, 1>>();

  // SYCL DNN Backend
  Backend backend{*Queue()};

  using DeviceMem = Backend::internal_pointer_type<T>;

  // Creating Device Pointers to Buffers
  auto a_data = DeviceMem(A_buffer, static_cast<size_t>(A->ByteOffset() / sizeof(T)));
  auto b_data = DeviceMem(B_buffer, static_cast<size_t>(B->ByteOffset() / sizeof(T)));
  auto y_data = DeviceMem(Y_buffer, static_cast<size_t>(Y->ByteOffset() / sizeof(T)));

  // Launching Matmul kernel
  backend.template matmul<false, false, T, int>(a_data, b_data, y_data, 0.f, M, K, N);

  return Status::OK();
}

// REGISTER KERNEL
REGISTER_VERSIONED_MATMUL_KERNEL_TYPED(float, 1, 8)
REGISTER_VERSIONED_MATMUL_KERNEL_TYPED(float, 9, 12)
REGISTER_MATMUL_KERNEL_TYPED(float, 13)

}  // namespace sycl
}  // namespace onnxruntime
