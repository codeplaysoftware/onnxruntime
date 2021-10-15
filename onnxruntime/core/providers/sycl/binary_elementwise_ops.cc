// Codeplay Software Ltd.

#include "core/providers/sycl/sycl_kernel.h"
#include "core/framework/data_types_internal.h"
#include "core/framework/op_kernel.h"
#include "core/providers/sycl/sycl_fwd.h"
#include "core/providers/sycl/binary_elementwise_ops.h"

#include <CL/sycl.hpp>

#include "sycldnn/backend/snn_backend.h"
#include "sycldnn/bias/launch.h"
#include "sycldnn/bias/params.h"
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

  cl::sycl::buffer<T, 1> X2_buffer{X2_data,
                                   cl::sycl::range<1>{dataSize},
                                   {cl::sycl::property::buffer::context_bound{Queue()->get_context()},
                                    cl::sycl::property::buffer::use_host_ptr{}}};

  cl::sycl::buffer<T, 1> Y_buffer{Y_data,
                                  cl::sycl::range<1>{dataSize},
                                  {cl::sycl::property::buffer::context_bound{Queue()->get_context()},
                                   cl::sycl::property::buffer::use_host_ptr{}}};

  // SYCL DNN Backend
  Backend backend{*Queue()};
  snn::bias::BiasParams bias_params{};
  bias_params.channels = 1;
  bias_params.batch = 1;
  bias_params.in_rows = dataSize;
  bias_params.in_cols = 1;
  bias_params.bias = dataSize;

  auto input_ = snn::backend::DeviceMemPointer(X1_buffer, 0);  //Offset = 0
  auto biases_ = snn::backend::DeviceMemPointer(X2_buffer, 0);
  auto output_ = snn::backend::DeviceMemPointer(Y_buffer, 0);

  // Launch Bias Add kernel
  auto ev = snn::bias::launch<float>(input_,
                                     biases_,
                                     output_,
                                     bias_params,
                                     backend);
  ev.event.wait_and_throw();

  return Status::OK();
}

REGISTER_BIN_ELEM_WISE_KERNELS_TYPED(float)

}  // namespace sycl
}  // namespace onnxruntime
