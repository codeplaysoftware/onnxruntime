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

#include "core/providers/sycl/tensor/reshape.h"
#include "core/providers/common.h"
#include "core/providers/cpu/tensor/reshape_helper.h"

#include <CL/sycl.hpp>

#include "sycldnn/backend/sycl_blas_backend.h"
#include "sycldnn/status.h"

namespace snn = sycldnn;
using Backend = snn::backend::SyclBLASBackend;

namespace onnxruntime {
namespace sycl {

#define REGISTER_VERSIONED_RESHAPE_KERNEL_TYPED(start, end)     \
  ONNX_OPERATOR_VERSIONED_KERNEL_EX(                            \
      Reshape, kOnnxDomain, start, end, kSyclExecutionProvider, \
      KernelDefBuilder().TypeConstraint(                        \
          "T", DataTypeImpl::AllFixedSizeTensorTypes()),        \
      Reshape);

#define REGISTER_RESHAPE_KERNEL_TYPED(start)                                   \
  ONNX_OPERATOR_KERNEL_EX(Reshape, kOnnxDomain, start, kSyclExecutionProvider, \
                          KernelDefBuilder().TypeConstraint(                   \
                              "T", DataTypeImpl::AllFixedSizeTensorTypes()),   \
                          Reshape);

Status Reshape::ComputeInternal(OpKernelContext* context) const {
  // Copy the second input tensor into the shape vector
  const Tensor* shapeTensor = context->Input<Tensor>(1);
  ORT_ENFORCE(shapeTensor->Shape().NumDimensions() == 1,
              "A shape tensor must be a vector tensor.");

  auto nDims = static_cast<size_t>(shapeTensor->Shape()[0]);
  cl::sycl::buffer<int64_t, 1>* shape_buffer =
      const_cast<cl::sycl::buffer<int64_t, 1>*>(
          shapeTensor->template Ptr<cl::sycl::buffer<int64_t, 1>>());
  std::vector<int64_t> shape(nDims);

  const Tensor* X = context->Input<Tensor>(0);

  // need to copy the contents of shape_buffer to shape vector
  // this would help when creating the ReshapeHelper object
  Queue()
      ->submit([&](cl::sycl::handler& cgh) {
        auto acc =
            shape_buffer->template get_access<cl::sycl::access::mode::read>(
                cgh);
        cgh.copy(acc, shape.data());
      })
      .wait();

  ReshapeHelper helper(X->Shape(), shape, allow_zero_);

  Tensor* Y = context->Output(0, TensorShape(shape));

  // SYCL BUFFERS
  const cl::sycl::buffer<float, 1> X_buffer =
      *X->template Ptr<cl::sycl::buffer<float, 1>>();
  cl::sycl::buffer<float, 1> Y_buffer =
      *Y->template MutablePtr<cl::sycl::buffer<float, 1>>();

  auto count = X->SizeInBytes();

  Backend backend{*Queue()};

  using DeviceMem = Backend::internal_pointer_type<float>;

  auto x_data = DeviceMem(X_buffer, X->ByteOffset() / sizeof(float));
  auto y_data = DeviceMem(Y_buffer, Y->ByteOffset() / sizeof(float));

  auto executor = backend.get_executor();

  blas::_copy(executor, static_cast<int>(count), x_data, 1, y_data, 1);

  return Status::OK();
}

REGISTER_VERSIONED_RESHAPE_KERNEL_TYPED(1, 4)
REGISTER_VERSIONED_RESHAPE_KERNEL_TYPED(5, 12)
REGISTER_VERSIONED_RESHAPE_KERNEL_TYPED(13, 13)
REGISTER_RESHAPE_KERNEL_TYPED(14)

}  // namespace sycl
}  // namespace onnxruntime
