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

#include "core/providers/sycl/sycl_kernel.h"
#include "core/framework/data_types_internal.h"
#include "core/providers/cpu/tensor/transpose.h"
#include "core/framework/op_kernel.h"
#include "core/providers/sycl/sycl_fwd.h"

namespace onnxruntime {
namespace sycl {

template <typename T>
class Transpose final : public SyclKernel, public TransposeBase {
 public:
  Transpose(const OpKernelInfo& info) : SyclKernel(info), TransposeBase(info) {}

  Status ComputeInternal(OpKernelContext* ctx) const override;
};

}  // namespace sycl
}  // namespace onnxruntime
