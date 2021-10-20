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

#include "core/providers/sycl/sycl_execution_provider_info.h"
#include "core/framework/provider_options_utils.h"

namespace onnxruntime {
namespace sycl {
namespace provider_option_names {
constexpr const char* kDeviceSelector = "device_selector";

}  // namespace provider_option_names
}  // namespace sycl

SYCLExecutionProviderInfo SYCLExecutionProviderInfo::FromProviderOptions(const ProviderOptions& options) {
  SYCLExecutionProviderInfo info{};
  ProviderOptionsParser{}.AddAssignmentToReference(sycl::provider_option_names::kDeviceSelector, info.device_selector).Parse(options);
  return info;
}

ProviderOptions SYCLExecutionProviderInfo::ToProviderOptions(const SYCLExecutionProviderInfo& info) {
  const ProviderOptions options{
      {sycl::provider_option_names::kDeviceSelector, MakeStringWithClassicLocale(info.device_selector)},
  };

  return options;
}
}  // namespace onnxruntime
