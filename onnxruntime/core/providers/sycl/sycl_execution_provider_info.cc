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
}  // namespace provider_option_names
}  // namespace sycl

namespace {
// Target device selectors for SYCL
const EnumNameMapping<OrtSYCLDeviceSelector> ort_sycl_device_selector_mapping{
    {DEFAULT, "DEFAULT"},
    {GPU, "GPU"},
    {CPU, "CPU"},
    {HOST, "HOST"},
};
}  // namespace

SYCLExecutionProviderInfo SYCLExecutionProviderInfo::FromProviderOptions(const ProviderOptions& options) {
  SYCLExecutionProviderInfo info{};
  ORT_THROW_IF_ERROR(
      ProviderOptionsParser{}
          .AddValueParser(
              sycl::provider_option_names::kDeviceId,
              [&info](const std::string& value_str) -> Status {
                ORT_RETURN_IF_ERROR(ParseStringWithClassicLocale(value_str, info.device_id));
                ORT_RETURN_IF_NOT(
                    0 <= info.device_id,
                    "Invalid device ID: ", info.device_id,
                    "Device ID must be positive");  // Further checks can be added on device_id once mapped to a CL device_id
                return Status::OK();
              })
          .AddAssignmentToEnumReference(
              sycl::provider_option_names::kDeviceSelector,
              ort_sycl_device_selector_mapping, info.device_selector)
          .Parse(options));

  return info;
}

ProviderOptions SYCLExecutionProviderInfo::ToProviderOptions(const SYCLExecutionProviderInfo& info) {
  const ProviderOptions options{
      {sycl::provider_option_names::kDeviceSelector,
       EnumToName(ort_sycl_device_selector_mapping, info.device_selector)},
      {sycl::provider_option_names::kDeviceId, MakeStringWithClassicLocale(info.device_id)},
  };

  return options;
}
}  // namespace onnxruntime
