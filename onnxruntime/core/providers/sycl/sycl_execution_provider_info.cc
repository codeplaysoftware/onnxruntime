// Codeplay Software Ltd.

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
