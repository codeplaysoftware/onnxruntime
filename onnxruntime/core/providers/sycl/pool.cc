// Codeplay Software Ltd.

#include "core/providers/sycl/pool.h"

#include <CL/sycl.hpp>

#include "sycldnn/backend/snn_backend.h"
#include "sycldnn/pooling/launch.h"
#include "sycldnn/pooling/operators.h"
#include "sycldnn/transpose/launch.h"
#include "sycldnn/status.h"

namespace snn = sycldnn;
using Backend = snn::backend::SNNBackend;

namespace onnxruntime {
namespace sycl {

// Registering VERSIONNED TYPED Kernels
#define REGISTER_POOLING_VERSIONED_KERNEL_TYPED(T, op, pool_type, start, end) \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                    \
      op,                                                                     \
      kOnnxDomain,                                                            \
      start,                                                                  \
      end,                                                                    \
      T,                                                                      \
      kSyclExecutionProvider,                                                 \
      KernelDefBuilder()                                                      \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),             \
      Pool<T, pool_type>);

#define REGISTER_POOLING_KERNEL_TYPED(T, op, pool_type, start)    \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      op,                                                         \
      kOnnxDomain,                                                \
      start,                                                      \
      T,                                                          \
      kSyclExecutionProvider,                                     \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Pool<T, pool_type>);

// MaxPool
template <typename T, typename PoolType>
Status Pool<T, PoolType>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* X_data = context->Input<Tensor>(0);
  const TensorShape& x_shape = X_data->Shape();
  size_t input_size = X_data->SizeInBytes() / sizeof(T);
  const int64_t N = X_data->Shape()[0];
  const int64_t C = X_data->Shape()[1];
  const int64_t H_in = X_data->Shape()[2];
  const int64_t W_in = X_data->Shape()[3];

  size_t input_dims = x_shape.NumDimensions();
  ORT_RETURN_IF_NOT(input_dims >= 3, "Input dimension cannot be less than 3.");

  size_t pooling_dims = input_dims - 2;
  if (pooling_dims > 3) {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "Unsupported pooling size.");
  }
  if (!pool_attrs_.global_pooling) {
    ORT_RETURN_IF_NOT(pooling_dims == pool_attrs_.kernel_shape.size(),
                      "kernel_shape num_dims is not compatible with X num_dims.");
  }

  std::vector<int64_t> pads = pool_attrs_.pads;
  std::vector<int64_t> output_dims = pool_attrs_.SetOutputSize(x_shape, x_shape[1], &pads);
  TensorShape output_shape(output_dims);
  Tensor* Y_data = context->Output(0, output_shape);
  size_t output_size = Y_data->SizeInBytes() / sizeof(T);

  const int64_t H_out = output_shape[2];
  const int64_t W_out = output_shape[3];

  snn::pooling::PoolingParams params;
  params.in_rows = static_cast<int>(H_in);
  params.in_cols = static_cast<int>(W_in);
  params.out_rows = static_cast<int>(H_out);
  params.out_cols = static_cast<int>(W_out);
  params.window_rows = pool_attrs_.global_pooling ? static_cast<int>(H_in) : static_cast<int>(pool_attrs_.kernel_shape[0]);
  params.window_cols = pool_attrs_.global_pooling ? static_cast<int>(W_in) : static_cast<int>(pool_attrs_.kernel_shape[1]);
  params.stride_rows = pool_attrs_.global_pooling ? 1 : static_cast<int>(pool_attrs_.strides[0]);
  params.stride_cols = pool_attrs_.global_pooling ? 1 : static_cast<int>(pool_attrs_.strides[1]);
  params.batch = static_cast<int>(N);
  params.channels = static_cast<int>(C);
  params.pad_rows = pool_attrs_.global_pooling ? 0 : static_cast<int>(pool_attrs_.pads[0]);
  params.pad_cols = pool_attrs_.global_pooling ? 0 : static_cast<int>(pool_attrs_.pads[2]);
  
  // edge case: one or more dims with value of 0
  if (output_shape.Size() == 0)
    return Status::OK();

  const T* X_ptr = X_data->template Data<T>();
  T* Y_ptr = Y_data->template MutableData<T>();

  // Buffer USM Interop
  cl::sycl::buffer<T, 1> X_buffer{X_ptr,
                                  cl::sycl::range<1>{input_size},
                                  {cl::sycl::property::buffer::context_bound{Queue()->get_context()},
                                   cl::sycl::property::buffer::use_host_ptr{}}};

  cl::sycl::buffer<T, 1> Y_buffer{Y_ptr,
                                  cl::sycl::range<1>{output_size},
                                  {cl::sycl::property::buffer::context_bound{Queue()->get_context()},
                                   cl::sycl::property::buffer::use_host_ptr{}}};

  // SYCL DNN Backend
  Backend backend{*Queue()};

  using DeviceMem = Backend::internal_pointer_type<T>;

  auto X_ = DeviceMem(X_buffer, 0);
  auto Y_ = DeviceMem(Y_buffer, 0);

  DeviceMem input, output;
  input = backend.template allocate<T>(static_cast<size_t>(N * C * H_in * W_in));
  output = backend.template allocate<T>(static_cast<size_t>(N * C * H_out * W_out));

  const std::vector<int> input_sizes = {(int)N, (int)C, (int)H_in, (int)W_in};
  snn::transpose::convert_nchw_to_nhwc<T, Backend>(X_, input, input_sizes, backend);

  // Launch kernel
  if constexpr (PoolType::type == onnxruntime::PoolType::kAveragePool) {
    snn::pooling::launch<float, snn::pooling::Average, snn::pooling::Forward>(
        input, output, params, backend);
  } else if constexpr (PoolType::type == onnxruntime::PoolType::kMaxPool) {
    snn::pooling::launch<float, snn::pooling::Max, snn::pooling::Forward>(
        input, output, params, backend);
  }

  const std::vector<int> output_sizes = {(int)N, (int)H_out, (int)W_out, (int)C};
  snn::transpose::convert_nhwc_to_nchw<T, Backend>(output, Y_, output_sizes, backend);

  backend.template deallocate(input);
  backend.template deallocate(output);

  backend.template deallocate(X_);
  backend.template deallocate(Y_);

  return Status::OK();
}

// REGISTER KERNEL
REGISTER_POOLING_VERSIONED_KERNEL_TYPED(float, MaxPool, MaxPool<1>, 1, 7)
REGISTER_POOLING_VERSIONED_KERNEL_TYPED(float, MaxPool, MaxPool<8>, 8, 9)
REGISTER_POOLING_KERNEL_TYPED(float, MaxPool, MaxPool<8>, 10)
REGISTER_POOLING_KERNEL_TYPED(float, MaxPool, MaxPool<8>, 11)
REGISTER_POOLING_KERNEL_TYPED(float, MaxPool, MaxPool<8>, 12)
REGISTER_POOLING_KERNEL_TYPED(float, GlobalMaxPool, MaxPool<1>, 1)

REGISTER_POOLING_VERSIONED_KERNEL_TYPED(float, AveragePool, AveragePool, 7, 9)
REGISTER_POOLING_VERSIONED_KERNEL_TYPED(float, AveragePool, AveragePool, 10, 10)
REGISTER_POOLING_KERNEL_TYPED(float, AveragePool, AveragePool, 11)
REGISTER_POOLING_KERNEL_TYPED(float, GlobalAveragePool, AveragePool, 1)

}  // namespace sycl
}  // namespace onnxruntime
