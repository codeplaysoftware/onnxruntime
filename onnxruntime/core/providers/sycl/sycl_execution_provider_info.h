// Codeplay Software Ltd.

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
