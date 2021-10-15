// Codeplay Software Ltd.

#include "core/providers/sycl/relu.h"

#include <CL/sycl.hpp>

#include "sycldnn/backend/snn_backend.h"
#include "sycldnn/pointwise/launch.h"
#include "sycldnn/status.h"

namespace snn = sycldnn;
using Backend = snn::backend::SNNBackend;

namespace onnxruntime {
namespace sycl {

// Registering VERSIONNED TYPED Kernels
#define REGISTER_RELU_KERNEL_TYPED(T)                             \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                        \
      Relu,                                                       \
      kOnnxDomain,                                                \
      1,                                                          \
      12,                                                         \
      T,                                                          \
      kSyclExecutionProvider,                                     \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Relu<T>);

// Relu Y = max{X1, 0}
template <typename T>
Status Relu<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* X1 = context->Input<Tensor>(0);
  Tensor* Y = context->Output(0, X1->Shape());

  size_t dataSize = Y->SizeInBytes() / sizeof(T);

  if (Y->Shape().Size() == 0)
    return Status::OK();

  const T* X1_data = X1->template Data<T>();
  T* Y_data = Y->template MutableData<T>();

  // Buffer USM Interop
  cl::sycl::buffer<T, 1> X1_buffer{X1_data,
                                   cl::sycl::range<1>{dataSize},
                                   {cl::sycl::property::buffer::context_bound{Queue()->get_context()},
                                    cl::sycl::property::buffer::use_host_ptr{}}};

  cl::sycl::buffer<T, 1> Y_buffer{Y_data,
                                  cl::sycl::range<1>{dataSize},
                                  {cl::sycl::property::buffer::context_bound{Queue()->get_context()},
                                   cl::sycl::property::buffer::use_host_ptr{}}};

  // SYCL DNN Backend
  Backend backend{*Queue()};
  using DeviceMem = Backend::internal_pointer_type<T>;

  auto X1_ = DeviceMem(X1_buffer, 0);  //Offset = 0
  auto Y_ = DeviceMem(Y_buffer, 0);

  // Launch kernel
  snn::pointwise::launch<float, snn::pointwise::Relu, snn::pointwise::Forward>(X1_, Y_, dataSize, backend);

  backend.template deallocate(X1_);
  backend.template deallocate(Y_);

  return Status::OK();
}

// REGISTER KERNEL
REGISTER_RELU_KERNEL_TYPED(float)

}  // namespace sycl
}  // namespace onnxruntime
