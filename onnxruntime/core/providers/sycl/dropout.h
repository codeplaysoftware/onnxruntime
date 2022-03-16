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

#include "core/common/common.h"
#include "core/providers/sycl/sycl_kernel.h"
#include "core/framework/data_types_internal.h"
#include "core/framework/op_kernel.h"
#include "core/providers/sycl/sycl_fwd.h"
#include "core/framework/random_generator.h"

namespace onnxruntime {
namespace sycl {

template <typename T>
struct GetRatioDataImpl {
  void operator()(const Tensor* ratio, float& ratio_data) const {
    ratio_data = static_cast<float>(*(ratio->template Ptr<T>()));
    ORT_ENFORCE(ratio_data >= 0.0f && ratio_data < 1.0f,
                "ratio_data is outside range [0, 1)");
  }
};

namespace {
constexpr float k_default_ratio{0.5f};

template <typename T2>
float GetRatioOrDefault(const Tensor* ratio_tensor) {
  if (ratio_tensor) {
    ORT_ENFORCE(ratio_tensor->Shape().Size() == 1,
                "ratio input should have a single value.");
#ifdef _WIN32
#pragma warning(disable : 4244)
#endif
    const float ratio_value = *ratio_tensor->Ptr<T2>();
    ORT_ENFORCE(0.0f <= ratio_value && ratio_value < 1.0f,
                "ratio must be in the range [0, 1)");
    return ratio_value;
  }
  return k_default_ratio;
}
}  // namespace

class Dropout final : public SyclKernel {
 public:
  Dropout(const OpKernelInfo& info) : SyclKernel(info) {
    int64_t seed = 0;
    if (info.GetAttr<int64_t>("seed", &seed).IsOK()) {
      generator_ =
          std::make_unique<PhiloxGenerator>(static_cast<uint64_t>(seed));
    }
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  mutable std::unique_ptr<PhiloxGenerator> generator_;
  static constexpr float default_ratio_ = 0.5f;
};

}  // namespace sycl
}  // namespace onnxruntime
