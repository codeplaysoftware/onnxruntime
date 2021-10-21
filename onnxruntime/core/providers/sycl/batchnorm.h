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
#include "core/providers/cpu/nn/conv_attributes.h"
#include "core/providers/sycl/sycl_fwd.h"

namespace onnxruntime {
namespace sycl {

template <typename T>
class BatchNorm final : public SyclKernel {
 public:
  BatchNorm(const OpKernelInfo& info) : SyclKernel(info) {
    ORT_ENFORCE(info.GetAttr<float>("epsilon", &epsilon_).IsOK());

    //TODO: need to add support for training mode, spatial_ and momentum_
    // leaving these as they are not used for now
    is_training_mode_ = (info.GetAttrOrDefault<int64_t>("training_mode", 0) == 1);
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  float epsilon_;
  int64_t spatial_ = 1;        // default as per spec
  bool is_training_mode_ = 0;  //default as per spec
};

}  // namespace sycl
}  // namespace onnxruntime
