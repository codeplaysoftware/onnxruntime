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

#include "core/providers/sycl/reduction/reduce.h"
#include "core/providers/common.h"

#include <CL/sycl.hpp>

#include "sycldnn/backend/sycl_blas_backend.h"
#include "sycldnn/transpose/launch.h"
#include "sycldnn/status.h"

namespace snn = sycldnn;
using Backend = snn::backend::SyclBLASBackend;

namespace onnxruntime {
namespace sycl {

// Registering Kernels
#define REGISTER_VERSIONED_REDUCE_KERNELS_TYPED(T, op, start, end)         \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                 \
      op, kOnnxDomain, start, end, T, kSyclExecutionProvider,              \
      KernelDefBuilder().TypeConstraint("T",                               \
                                        DataTypeImpl::GetTensorType<T>()), \
      op<T>);

#define REGISTER_REDUCE_KERNELS_TYPED(T, op, start)                         \
  ONNX_OPERATOR_TYPED_KERNEL_EX(op, kOnnxDomain, start, T,                  \
                                kSyclExecutionProvider,                     \
                                KernelDefBuilder().TypeConstraint(          \
                                    "T", DataTypeImpl::GetTensorType<T>()), \
                                op<T>);

template <typename T>
Status ReduceMean<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* X = context->Input<Tensor>(0);
  const TensorShape& x_shape = X->Shape();
  size_t x_dims = x_shape.NumDimensions();

  std::vector<int> input_shape(x_dims);
  std::vector<int> transpose_permutations(x_dims + axes_.size());
  for (size_t i = 0; i < x_dims; i++) {
    input_shape[i] = gsl::narrow_cast<int>(x_shape[i]);
    transpose_permutations[i] = gsl::narrow_cast<int>(i);
  }

  if (axes_.size() > 0) {
    // Push the dimensions to be reduced to the far right
    for (size_t i = 0; i < axes_.size(); i++) {
      int axis = gsl::narrow_cast<int>(HandleNegativeAxis(axes_[i], x_dims));
      transpose_permutations[x_dims] = transpose_permutations[axis - i];
      transpose_permutations.erase(transpose_permutations.begin() + axis - i);
    }
  }

  std::vector<int64_t> y_shape;

  if (axes_.size() > 0) {
    // Compute y_shape
    for (size_t i = 0, j = 0; i < input_shape.size(); i++) {
      if (i == static_cast<size_t>(axes_[j]) && j < axes_.size()) {
        if (keepdims_) {
          y_shape.push_back(1);
        }
        j++;
      } else {
        y_shape.push_back(static_cast<int64_t>(input_shape[i]));
      }
    }
  } else {
    if (keepdims_) {
      for (size_t i = 0; i < x_dims; i++) {
        y_shape.push_back(1);
      }
    }
  }

  Tensor* Y = context->Output(0, y_shape);

  const cl::sycl::buffer<T, 1> X_buffer =
      *X->template Ptr<cl::sycl::buffer<T, 1>>();
  cl::sycl::buffer<T, 1> Y_buffer =
      *Y->template MutablePtr<cl::sycl::buffer<T, 1>>();

  // SYCL DNN Backend
  Backend backend{*Queue()};

  using DeviceMem = Backend::internal_pointer_type<T>;

  // Creating Device Pointers to Buffers
  auto x_data =
      DeviceMem(X_buffer, static_cast<size_t>(X->ByteOffset() / sizeof(T)));
  auto y_data =
      DeviceMem(Y_buffer, static_cast<size_t>(Y->ByteOffset() / sizeof(T)));

  int preserve_dims, reduce_dims;
  preserve_dims = reduce_dims = 1;

  auto executor = backend.get_executor();

  if (axes_.size() > 0) {
    // Compute preserve_dims and reduce_dims
    for (size_t i = 0, j = 0; i < input_shape.size(); i++) {
      if (i == static_cast<size_t>(axes_[j]) && j < axes_.size()) {
        reduce_dims *= input_shape[i];
        j++;
      } else {
        preserve_dims *= input_shape[i];
      }
    }
    // Allocate transpose memory to re-order input data such that all of the
    // reduction axes become the inner most dimensions
    DeviceMem transpose_data =
        backend.template allocate<T>(X->SizeInBytes() / sizeof(T));

    // Make input shape and transpose permutation vectors to be 4D
    // for correct invocation of SYCL-DNN transpose
    if (x_dims < 4) {
      for (size_t i = 0; i < 4 - x_dims; i++) {
        input_shape.push_back(1);
        transpose_permutations.push_back(gsl::narrow_cast<int>(x_dims + i));
      }
    }

    // Launch the transpose kernel to make all reductions axes inner most
    // dimensions
    snn::transpose::launch<T, Backend>(x_data, transpose_data, input_shape,
                                       transpose_permutations, backend);

    blas::extension::_reduction<blas::MeanOperator, T>(
        executor, transpose_data, reduce_dims, y_data, reduce_dims,
        preserve_dims, blas::reduction_dim_t::inner);

    backend.template deallocate(transpose_data);
  } else {
    preserve_dims = 1;
    for (size_t i = 0; i < x_dims; i++) {
      reduce_dims *= input_shape[i];
    }
    blas::extension::_reduction<blas::MeanOperator, T>(
        executor, x_data, reduce_dims, y_data, reduce_dims, preserve_dims,
        blas::reduction_dim_t::inner);
  }

  return Status::OK();
}

REGISTER_VERSIONED_REDUCE_KERNELS_TYPED(float, ReduceMean, 1, 10)
REGISTER_VERSIONED_REDUCE_KERNELS_TYPED(float, ReduceMean, 11, 12)
REGISTER_REDUCE_KERNELS_TYPED(float, ReduceMean, 13)

}  // namespace sycl
}  // namespace onnxruntime
