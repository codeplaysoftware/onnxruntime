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
#include "sycldnn/bias/launch.h"
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

  const T* X_data = X->template Data<T>();
  const T* W_data = W->template Data<T>();
  T* Y_data = Y->template MutableData<T>();

  // Buffer USM Interop
  cl::sycl::buffer<T, 1> X_buffer{X_data,
                                  cl::sycl::range<1>{static_cast<size_t>(M * K)},
                                  {cl::sycl::property::buffer::context_bound{Queue()->get_context()},
                                   cl::sycl::property::buffer::use_host_ptr{}}};

  cl::sycl::buffer<T, 1> W_buffer{W_data,
                                  cl::sycl::range<1>{static_cast<size_t>(K * N)},
                                  {cl::sycl::property::buffer::context_bound{Queue()->get_context()},
                                   cl::sycl::property::buffer::use_host_ptr{}}};

  cl::sycl::buffer<T, 1> Y_buffer{Y_data,
                                  cl::sycl::range<1>{static_cast<size_t>(M * N)},
                                  {cl::sycl::property::buffer::context_bound{Queue()->get_context()},
                                   cl::sycl::property::buffer::use_host_ptr{}}};

  // SYCL DNN Backend
  Backend backend{*Queue()};

  using DeviceMem = Backend::internal_pointer_type<T>;

  //Creating Device Pointers to Buffers
  auto X_ = DeviceMem(X_buffer, 0);  //Offset = 0
  auto W_ = DeviceMem(W_buffer, 0);
  auto Y_ = DeviceMem(Y_buffer, 0);

  auto executor = backend.get_executor();

  //Switch M and N to meet SYCL-BLAS requirements
  auto trans_m = N;
  auto trans_n = M;

  //Compute ld dimension based on transpose parameters
  auto ldc = trans_m;
  auto lda = trans_B_ ? K : trans_m;
  auto ldb = trans_A_ ? trans_n : K;

  //Launching SYCL-BLAS Gemm
  blas::_gemm(executor, trans_B_ ? 't' : 'n',
              trans_A_ ? 't' : 'n', trans_m, trans_n, K,
              alpha_, W_, lda, X_, ldb, beta_, Y_, ldc);

  //Check if Bias Addition is required
  if (nullptr != B) {
    const T* Bdata = B->template Data<T>();
    cl::sycl::buffer<T, 1> B_buffer{Bdata,
                                    cl::sycl::range<1>{static_cast<size_t>(M * N)},
                                    {cl::sycl::property::buffer::context_bound{Queue()->get_context()},
                                     cl::sycl::property::buffer::use_host_ptr{}}};
    auto B_ = DeviceMem{B_buffer, 0};

    //Settubg Bias parameters
    snn::bias::BiasParams bias_params;
    bias_params.in_rows = 1;
    bias_params.in_cols = 1;
    bias_params.batch = 1;
    bias_params.channels = M * N;
    bias_params.bias = M * N;

    //Launching Bias addition kernel
    snn::bias::launch<T>(Y_, B_, Y_, bias_params, backend);

    //Deallocating the Bias device pointer
    backend.template deallocate(B_);
  }

  //Deallocating all the memory elements used
  backend.template deallocate(X_);
  backend.template deallocate(W_);
  backend.template deallocate(Y_);

  return Status::OK();
}

// REGISTER KERNEL
REGISTER_VERSIONED_GEMM_KERNEL_TYPED(float, 7, 8)
REGISTER_VERSIONED_GEMM_KERNEL_TYPED(float, 9, 10)
REGISTER_VERSIONED_GEMM_KERNEL_TYPED(float, 11, 12)
REGISTER_GEMM_KERNEL_TYPED(float, 13)

}  // namespace sycl
}  // namespace onnxruntime
