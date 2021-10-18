// Codeplay Software Ltd.

#include "core/providers/sycl/binary_elementwise_ops.h"

#include <CL/sycl.hpp>

#include "sycldnn/backend/snn_backend.h"
#include "sycldnn/pointwise/launch.h"
#include "sycldnn/status.h"

namespace snn = sycldnn;
using Backend = snn::backend::SNNBackend;

namespace onnxruntime {

namespace sycl {

// Registering VERSIONNED TYPED Kernels
#define REGISTER_BIN_ELEM_WISE_KERNELS_TYPED(T)                   \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                        \
      Add,                                                        \
      kOnnxDomain,                                                \
      7,                                                          \
      12,                                                         \
      T,                                                          \
      kSyclExecutionProvider,                                     \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Add<T>);

// Add Operator
template <typename T>
Status Add<T>::ComputeInternal(OpKernelContext* context) const {
  // INPUT
  const Tensor* X1 = context->Input<Tensor>(0);
  const Tensor* X2 = context->Input<Tensor>(1);

  // OUTPUT
  Tensor* Y = context->Output(0, X1->Shape());

  // RAW DATA PTRs
  const T* X1_data = nullptr;
  const T* X2_data = nullptr;
  T* Y_data = nullptr;

  X1_data = X1->template Data<T>();
  X2_data = X2->template Data<T>();
  Y_data = Y->template MutableData<T>();

  size_t dataSize = Y->SizeInBytes() / sizeof(T);

  // Buffer USM Interop
  cl::sycl::buffer<T, 1> X1_buffer{X1_data,
                                   cl::sycl::range<1>{dataSize},
                                   {cl::sycl::property::buffer::context_bound{Queue()->get_context()},
                                    cl::sycl::property::buffer::use_host_ptr{}}};

  // SYCL DNN Backend
  auto queue = *Queue();
  Backend backend(queue);

  using DeviceMem = Backend::internal_pointer_type<T>;

  auto X1_ = DeviceMem(X1_buffer, 0);

  DeviceMem input = backend.template allocate<T>(dataSize);

  auto event = queue.submit([&](cl::sycl::handler& cgh) {
    auto buf = input.get_buffer();
    auto acc = buf.template get_access<cl::sycl::access::mode::discard_write>(cgh);
    cgh.copy(X2_data, acc);
  });
  event.wait_and_throw();

  snn::pointwise::launch<T, snn::pointwise::ResidualAdd, snn::pointwise::Forward, Backend>(X1_, input, dataSize, backend);

  event = queue.submit([&](cl::sycl::handler& cgh) {
    auto buf = input.get_buffer();
    auto acc = buf.template get_access<cl::sycl::access::mode::read>(cgh);
    cgh.copy(acc, Y_data);
  });
  event.wait_and_throw();

  return Status::OK();
}

REGISTER_BIN_ELEM_WISE_KERNELS_TYPED(float)

}  // namespace sycl
}  // namespace onnxruntime
