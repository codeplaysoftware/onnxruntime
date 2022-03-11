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

#include "core/providers/sycl/gemm.h"
#include "core/providers/cpu/math/gemm_helper.h"

#include <CL/sycl.hpp>

#include "sycldnn/backend/sycl_blas_backend.h"
#include "sycldnn/status.h"

namespace snn = sycldnn;
using Backend = snn::backend::SyclBLASBackend;

namespace onnxruntime {
namespace sycl {

// Registering VERSIONNED TYPED Kernels
#define REGISTER_VERSIONED_GEMM_KERNEL_TYPED(T, start, end)       \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                        \
      Gemm,                                                       \
      kOnnxDomain,                                                \
      start,                                                      \
      end,                                                        \
      T,                                                          \
      kSyclExecutionProvider,                                     \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Gemm<T>);

#define REGISTER_GEMM_KERNEL_TYPED(T, start)                      \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      Gemm,                                                       \
      kOnnxDomain,                                                \
      start,                                                      \
      T,                                                          \
      kSyclExecutionProvider,                                     \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Gemm<T>);

template <typename T>
Status Gemm<T>::ComputeInternal(OpKernelContext* context) const {
  // INPUT
  const Tensor* X = context->Input<Tensor>(0);
  const Tensor* W = context->Input<Tensor>(1);
  const Tensor* B = context->Input<Tensor>(2);

  // Gemm helper for dimensions verification / computation
  GemmHelper helper(X->Shape(), trans_A_, W->Shape(), trans_B_, B != nullptr ? B->Shape() : TensorShape({}));
  if (!helper.State().IsOK())
    return helper.State();

  // Extracting dimensions
  int M = static_cast<int>(helper.M());
  int N = static_cast<int>(helper.N());
  int K = static_cast<int>(helper.K());

  // OUTPUT
  Tensor* Y = context->Output(0, {M, N});

  // if input is empty tensor, return as nothing need to be calculated and we've set the shape for the output
  if (M == 0 || N == 0)
    return Status::OK();

  // SYCL BUFFERS
  const cl::sycl::buffer<T, 1> X_buffer = *X->template Ptr<cl::sycl::buffer<T, 1>>();
  const cl::sycl::buffer<T, 1> W_buffer = *W->template Ptr<cl::sycl::buffer<T, 1>>();
  cl::sycl::buffer<T, 1> Y_buffer = *Y->template MutablePtr<cl::sycl::buffer<T, 1>>();

  // SYCL DNN Backend
  auto queue = *Queue();
  Backend backend{queue};

  using DeviceMem = Backend::internal_pointer_type<T>;

  // Creating Device Pointers to Buffers
  auto x_data = DeviceMem(X_buffer, static_cast<size_t>(X->ByteOffset() / sizeof(T)));
  auto w_data = DeviceMem(W_buffer, static_cast<size_t>(W->ByteOffset() / sizeof(T)));
  auto y_data = DeviceMem(Y_buffer, static_cast<size_t>(Y->ByteOffset() / sizeof(T)));

  auto executor = backend.get_executor();

  //Check if Bias Addition is required
  if (beta_ != 0 && B != nullptr) {
    cl::sycl::buffer<T, 1>* B_buffer = const_cast<cl::sycl::buffer<T, 1>*>(B->template Ptr<cl::sycl::buffer<T, 1>>());
    const TensorShape& b_shape = B->Shape();
    auto b_data = DeviceMem(*B_buffer, static_cast<size_t>(B->ByteOffset() / sizeof(T)));

    if (b_shape.Size() == 1) {
      // do a cgh.fill()
      auto event = queue.submit([&](cl::sycl::handler& cgh) {
        auto in_acc = B_buffer->template get_access<cl::sycl::access::mode::read>(cgh);
        auto out_acc = Y_buffer.template get_access<cl::sycl::access::mode::discard_write>(cgh);
        cgh.fill(out_acc, in_acc[0]);
      });

    } else if (b_shape.NumDimensions() == 2 && b_shape[1] == 1) {
      // call SYCL-BLAS gemm
      // B(M,1)*ones(1,N)
      // TODO: We need to add Broadcast in SYCL-DNN to remove this slow Matmul
      auto ones = backend.template allocate<T>(static_cast<size_t>(N));
      auto event = queue.submit([&](cl::sycl::handler& cgh) {
        auto buf = ones.get_buffer();
        auto out_acc = buf.template get_access<cl::sycl::access::mode::discard_write>(cgh);
        cgh.fill(out_acc, (T)1);
      });

      backend.template matmul<false, false, T, int>(b_data, ones, y_data, 0.f, M, 1, N);
      backend.deallocate(ones);

    } else if (b_shape.NumDimensions() == 1 || b_shape[0] == 1) {
      // call SYCL-BLAS gemm
      // B(1,N)*ones(M,1)
      // TODO: We need to add Broadcast in SYCL-DNN to remove this slow Matmul
      auto ones = backend.template allocate<T>(static_cast<size_t>(M));
      auto event = queue.submit([&](cl::sycl::handler& cgh) {
        auto buf = ones.get_buffer();
        auto out_acc = buf.template get_access<cl::sycl::access::mode::discard_write>(cgh);
        cgh.fill(out_acc, (T)1);
      });

      backend.template matmul<false, false, T, int>(ones, b_data, y_data, 0.f, M, 1, N);
      backend.deallocate(ones);

    } else {
      auto data_transfer = this->GetDataTransfer();
      ORT_THROW_IF_ERROR(data_transfer->CopyTensor(*B, *Y));
    }
  }

  //Switch M and N to meet SYCL-BLAS requirements
  auto trans_m = N;
  auto trans_n = M;

  //Launching SYCL-BLAS Gemm
  if (M == 1) {
    // The LHS matrix is actually a vector
    auto gemv_m = trans_B_ ? K : trans_m;
    auto gemv_n = trans_B_ ? trans_m : K;
    auto gemv_lda = gemv_m;
    constexpr int increment = 1;
    blas::_gemv(executor, trans_B_ ? 't' : 'n', gemv_m, gemv_n,
                alpha_, w_data, gemv_lda, x_data, increment, beta_,
                y_data, increment);
  } else if (N == 1) {
    // The RHS matrix is actually a vector
    auto gemv_m = trans_A_ ? trans_n : K;
    auto gemv_n = trans_A_ ? K : trans_n;
    auto gemv_lda = gemv_m;
    constexpr int increment = 1;
    blas::_gemv(executor, trans_A_ ? 'n' : 't', gemv_m, gemv_n,
                alpha_, x_data, gemv_lda, w_data, increment, beta_,
                y_data, increment);
  } else {
    //Compute ld dimension based on transpose parameters
    auto ldc = trans_m;
    auto lda = trans_B_ ? K : trans_m;
    auto ldb = trans_A_ ? trans_n : K;

    blas::_gemm(executor, trans_B_ ? 't' : 'n',
                trans_A_ ? 't' : 'n', trans_m, trans_n, K,
                alpha_, w_data, lda, x_data, ldb, (B == nullptr) ? 0.f : beta_, y_data, ldc);
  }

  return Status::OK();
}

// REGISTER KERNEL
REGISTER_VERSIONED_GEMM_KERNEL_TYPED(float, 7, 8)
REGISTER_VERSIONED_GEMM_KERNEL_TYPED(float, 9, 10)
REGISTER_VERSIONED_GEMM_KERNEL_TYPED(float, 11, 12)
REGISTER_GEMM_KERNEL_TYPED(float, 13)

}  // namespace sycl
}  // namespace onnxruntime
