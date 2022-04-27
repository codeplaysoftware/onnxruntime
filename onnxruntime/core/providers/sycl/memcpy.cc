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

#include "core/providers/sycl/memcpy.h"

using namespace onnxruntime::common;
using namespace std;

namespace onnxruntime {

// This implementation is basically useful for CPU-SYCL EPs interoperability
// (e.g. subgraphs/nodes might be assigned to CPU EP as a fallback and this
// Memcpy ensures proper data movements between the two.
Status Memcpy::Compute(OpKernelContext* ctx) const {
  auto X_type = ctx->InputType(0);
  if (X_type->IsTensorType()) {
    const auto* X = ctx->Input<Tensor>(0);
    ORT_ENFORCE(X != nullptr, "Memcpy: Input tensor is nullptr.");
    // Triggers allocation of the node's output Y (SYCLHostAllocator::Alloc() if
    // dst = [CPU], and SYCLAllocator::Alloc() if dst = [SYCL_DEVICE]).
    Tensor* Y = ctx->Output(0, X->Shape());
    ORT_ENFORCE(Y != nullptr, "Memcpy: Failed to allocate output tensor.");
    return Info().GetDataTransferManager().CopyTensor(*X, *Y);
  } else {
    return Status(common::ONNXRUNTIME, common::FAIL,
                  "Memcpy: Unsupported input type.");
  }
}

namespace sycl {

ONNX_OPERATOR_KERNEL_EX(
    MemcpyFromHost, kOnnxDomain, 1, kSyclExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPUInput, 0)
        .InputMemoryType(OrtMemTypeDefault, 0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    Memcpy);

ONNX_OPERATOR_KERNEL_EX(
    MemcpyToHost, kOnnxDomain, 1, kSyclExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeDefault, 0)
        .OutputMemoryType(OrtMemTypeCPUOutput, 0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    Memcpy);

}  // namespace sycl
}  // namespace onnxruntime
