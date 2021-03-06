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

#pragma once

#include "core/providers/sycl/sycl_kernel.h"
#include "core/framework/data_types_internal.h"
#include "core/framework/op_kernel.h"
#include "core/providers/sycl/sycl_fwd.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace sycl {

template <typename T>
class Softmax final : public SyclKernel {
 public:
  Softmax(const OpKernelInfo& info) : SyclKernel(info) {
    const auto& node = info.node();
    opset_ = node.SinceVersion();

    int64_t axis;
    Status status = info.GetAttr<int64_t>("axis", &axis);

    if (status.IsOK()) {
      axis_ = gsl::narrow_cast<int>(axis);
    } else {
      if (opset_ < 13) {
        axis_ = 1;  // opset-12 and below, the default axis value is 1
      } else {
        axis_ = -1;  // opset-13, the default axis value is -1
      }
    }
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int axis_;   // Reduction axis (has different semantics depending on opset
               // version)
  int opset_;  // onnxruntime Operators Set Version
};
}  // namespace sycl
}  // namespace onnxruntime
