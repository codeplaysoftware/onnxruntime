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

namespace onnxruntime {
struct SYCLExecutionProviderInfo {
  explicit SYCLExecutionProviderInfo(bool use_gpu) : device_selector(use_gpu) {}
  SYCLExecutionProviderInfo() = default;

  // Conversion methods : Infos<->Options (Seems optional, not used so far)
  static SYCLExecutionProviderInfo FromProviderOptions(const ProviderOptions& options);
  static ProviderOptions ToProviderOptions(const SYCLExecutionProviderInfo& info);

  // Main infos (to be extended)
  bool device_selector{false};
};
}  // namespace onnxruntime
