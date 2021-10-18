// Codeplay Software Ltd.

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

// MatMul Y = X1*X2 (Matrix multiplication)
template <typename T>
Status MatMul<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* X1 = context->Input<Tensor>(0);
  const Tensor* X2 = context->Input<Tensor>(1);

  MatMulComputeHelper helper;
  ORT_RETURN_IF_ERROR(helper.Compute(X1->Shape(), X2->Shape()));

  Tensor* Y = context->Output(0, helper.OutputShape());

  if (Y->Shape().Size() == 0)
    return Status::OK();

  const size_t M = static_cast<size_t>(helper.M());
  const size_t N = static_cast<size_t>(helper.N());
  const size_t K = static_cast<size_t>(helper.K());

  const T* X1_data = X1->template Data<T>();
  const T* X2_data = X2->template Data<T>();
  T* Y_data = Y->template MutableData<T>();

  // Buffer USM Interop
  cl::sycl::buffer<T, 1> X1_buffer{X1_data,
                                   cl::sycl::range<1>{M * K},
                                   {cl::sycl::property::buffer::context_bound{Queue()->get_context()},
                                    cl::sycl::property::buffer::use_host_ptr{}}};

  cl::sycl::buffer<T, 1> X2_buffer{X2_data,
                                   cl::sycl::range<1>{K * N},
                                   {cl::sycl::property::buffer::context_bound{Queue()->get_context()},
                                    cl::sycl::property::buffer::use_host_ptr{}}};

  cl::sycl::buffer<T, 1> Y_buffer{Y_data,
                                  cl::sycl::range<1>{M * N},
                                  {cl::sycl::property::buffer::context_bound{Queue()->get_context()},
                                   cl::sycl::property::buffer::use_host_ptr{}}};

  // SYCL DNN Backend
  Backend backend{*Queue()};
  using DeviceMem = Backend::internal_pointer_type<T>;

  auto X1_ = DeviceMem(X1_buffer, 0);  //Offset = 0
  auto X2_ = DeviceMem(X2_buffer, 0);
  auto Y_ = DeviceMem(Y_buffer, 0);

  backend.template matmul<false, false, T, int>(X1_, X2_, Y_, 0.f, M, K, N);

  backend.template deallocate(X1_);
  backend.template deallocate(X2_);
  backend.template deallocate(Y_);

  return Status::OK();
}

// REGISTER KERNEL
REGISTER_VERSIONED_MATMUL_KERNEL_TYPED(float, 1, 8)
REGISTER_VERSIONED_MATMUL_KERNEL_TYPED(float, 9, 12)
REGISTER_MATMUL_KERNEL_TYPED(float, 13)

}  // namespace sycl
}  // namespace onnxruntime
