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
#include "core/common/parse_string.h"

namespace onnxruntime {
namespace sycl {
namespace provider_option_names {
constexpr const char* kDeviceSelector = "device_selector";
constexpr const char* kDeviceId = "device_id";
constexpr const char* kDeviceVendor = "device_vendor";
}  // namespace provider_option_names
}  // namespace sycl

SYCLExecutionProviderInfo SYCLExecutionProviderInfo::FromProviderOptions(const ProviderOptions& options) {
  SYCLExecutionProviderInfo info{};
  ORT_THROW_IF_ERROR(
      ProviderOptionsParser{}
          .AddAssignmentToReference(sycl::provider_option_names::kDeviceVendor, info.device_vendor)
          .AddAssignmentToReference(sycl::provider_option_names::kDeviceSelector, info.device_selector)
          .Parse(options));
  // Device id is not part of SYCL Options (since not provided by user)
  // but is set to 0 by default and always in SYCL Info.
  info.device_id = 0;
  return info;
}

ProviderOptions SYCLExecutionProviderInfo::ToProviderOptions(const SYCLExecutionProviderInfo& info) {
  const ProviderOptions options{
      {sycl::provider_option_names::kDeviceSelector, MakeStringWithClassicLocale(info.device_selector)},
      {sycl::provider_option_names::kDeviceVendor, MakeStringWithClassicLocale(info.device_vendor)},

  };

  return options;
}
}  // namespace onnxruntime
