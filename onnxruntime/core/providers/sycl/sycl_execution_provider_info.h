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

#include "core/framework/ortdevice.h"
#include "core/framework/provider_options.h"
#include "core/session/onnxruntime_c_api.h"
#include <string>

namespace onnxruntime {
struct SYCLExecutionProviderInfo {
  // Conversion methods : Infos<->Options
  static SYCLExecutionProviderInfo FromProviderOptions(const ProviderOptions& options);
  static ProviderOptions ToProviderOptions(const SYCLExecutionProviderInfo& info);

  // Main infos
  std::string device_selector{""};  // SYCL Device selector type : "CPU", "GPU", "HOST", "ACC", or "" for default selector
  std::string device_vendor{""};     // SYCL Device manufacturer name
  OrtDevice::DeviceId device_id{0};  // SYCL Device id (always = 0 since device is defined by selector not id)
};
}  // namespace onnxruntime
