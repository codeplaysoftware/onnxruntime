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
#include "core/framework/op_kernel.h"
#include "core/providers/sycl/sycl_fwd.h"

namespace onnxruntime {
namespace sycl {

template <typename T>
class ReduceMean final : public SyclKernel {
 public:
  ReduceMean(const OpKernelInfo& info) : SyclKernel(info) {
    axes_ = info.GetAttrsOrDefault<int64_t>("axes");
    int64_t keepdims = 1;
    ORT_ENFORCE(info.GetAttr("keepdims", &keepdims).IsOK());
    keepdims_ = (keepdims == 1);
    int64_t noop_with_empty_axes = info.GetAttrOrDefault<int64_t>("noop_with_empty_axes", 0);
    noop_with_empty_axes_ = (noop_with_empty_axes == 1);
    int64_t select_last_index = info.GetAttrOrDefault<int64_t>("select_last_index", 0);
    select_last_index_ = (select_last_index != 0);
  }
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  std::vector<int64_t> axes_;
  bool keepdims_;
  bool noop_with_empty_axes_;
  bool select_last_index_;
};

}  // namespace sycl
}  // namespace onnxruntime
