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

#include "core/providers/sycl/tensor/transpose.h"
#include "core/providers/cpu/tensor/utils.h"

#include <CL/sycl.hpp>

#include "sycldnn/backend/snn_backend.h"
#include "sycldnn/transpose/launch.h"
#include "sycldnn/status.h"

namespace snn = sycldnn;
using Backend = snn::backend::SNNBackend;

namespace onnxruntime {
namespace sycl {

#define REGISTER_VERSIONED_TRANSPOSE_KERNEL_TYPED(T, start, end)              \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                    \
      Transpose, kOnnxDomain, start, end, T, kSyclExecutionProvider,          \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllTensorTypes()), \
      Transpose<T>);

#define REGISTER_TRANSPOSE_KERNEL_TYPED(T, start)                             \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                              \
      Transpose, kOnnxDomain, start, T, kSyclExecutionProvider,               \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllTensorTypes()), \
      Transpose<T>);

template <typename OutTy, typename InTy>
static std::vector<OutTy> NarrowCastVector(const gsl::span<const InTy>& vec) {
  std::vector<OutTy> out;
  for (size_t i = 0; i < vec.size(); ++i) {
    out.push_back(gsl::narrow_cast<OutTy>(vec[i]));
  }

  return out;
}

template <typename OutTy, typename InTy>
static std::vector<OutTy> NarrowCastVector(const std::vector<InTy>& vec) {
  std::vector<OutTy> out;
  for (size_t i = 0; i < vec.size(); ++i) {
    out.push_back(gsl::narrow_cast<OutTy>(vec[i]));
  }

  return out;
}

template <typename T>
Status Transpose<T>::ComputeInternal(OpKernelContext* ctx) const {
  static_assert(std::is_same_v<T, float>, "Only float supported for Transpose");

  const Tensor* X = ctx->Input<Tensor>(0);
  const auto& in_dims = X->Shape().GetDims();
  const auto rank = in_dims.size();

  if (!(in_dims.size() >= 1 && in_dims.size() <= 6)) {
    return Status(common::ONNXRUNTIME, common::FAIL,
                  "Transpose only supported for tensors with number of "
                  "dimenions > 0 && < 7");
  }

  if (!X->IsDataType<float>()) {
    return Status(common::ONNXRUNTIME, common::FAIL,
                  "Transpose only supported for float tensors");
  }

  Backend backend{*Queue()};
  using DeviceMem = Backend::internal_pointer_type<T>;

  std::vector<int64_t> output_dims(rank);
  std::vector<size_t> default_perm(rank);
  const std::vector<size_t>* p_perm = nullptr;
  const auto& status =
      ComputeOutputShape(*X, output_dims, default_perm, p_perm);
  if (!status.IsOK()) {
    return status;
  }

  Tensor* Y = ctx->Output(0, {output_dims});
  if (Y->Shape().Size() == 0) {
    return Status::OK();
  }
  cl::sycl::buffer<T, 1> Y_buffer =
      *Y->template MutablePtr<cl::sycl::buffer<T, 1>>();
  auto y_data =
      DeviceMem(Y_buffer, static_cast<size_t>(Y->ByteOffset() / sizeof(T)));

  const cl::sycl::buffer<T, 1> X_buffer =
      *X->template Ptr<cl::sycl::buffer<T, 1>>();
  auto x_data =
      DeviceMem(X_buffer, static_cast<size_t>(X->ByteOffset() / sizeof(T)));

  snn::transpose::launch<T>(x_data, y_data, NarrowCastVector<int32_t>(in_dims),
                            NarrowCastVector<int32_t>(*p_perm), backend);

  return Status::OK();
}

REGISTER_VERSIONED_TRANSPOSE_KERNEL_TYPED(float, 1, 12)
REGISTER_TRANSPOSE_KERNEL_TYPED(float, 13)

}  // namespace sycl
}  // namespace onnxruntime
