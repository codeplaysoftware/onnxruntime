// Codeplay Software Ltd.

#include "core/providers/sycl/sycl_provider_factory_creator.h"
#include "core/providers/sycl/sycl_provider_factory.h"

#include <memory>

#include "core/providers/sycl/sycl_execution_provider.h"
#include "core/providers/sycl/sycl_execution_provider_info.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/ort_apis.h"

using namespace onnxruntime;

namespace onnxruntime {

// Factory structure
struct SYCLProviderFactory : IExecutionProviderFactory {
  SYCLProviderFactory(const SYCLExecutionProviderInfo& info)
      : info_{info} {
  }
  ~SYCLProviderFactory() override = default;

  std::unique_ptr<IExecutionProvider> CreateProvider() override;

 private:
  SYCLExecutionProviderInfo info_;
};

std::unique_ptr<IExecutionProvider> SYCLProviderFactory::CreateProvider() {
  return std::make_unique<SYCLExecutionProvider>(info_);
}

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_SYCL(const SYCLExecutionProviderInfo& info) {
  return std::make_shared<onnxruntime::SYCLProviderFactory>(info);
}
}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_SYCL,
                    _In_ OrtSessionOptions* options,
                    int device_selector) {
  SYCLExecutionProviderInfo info{};
  info.device_selector = (device_selector != 0);  //0 : false (cpu selector), 1 : true (gpu selector)
  options->provider_factories.push_back(onnxruntime::CreateExecutionProviderFactory_SYCL(info));
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtApis::SessionOptionsAppendExecutionProvider_SYCL,
                    _In_ OrtSessionOptions* options,
                    _In_ const OrtSYCLProviderOptions* sycl_options) {
  SYCLExecutionProviderInfo info{};
  info.device_selector = (sycl_options->device_selector != 0);  //0 : false (cpu selector), 1 : true (gpu selector)
  options->provider_factories.push_back(onnxruntime::CreateExecutionProviderFactory_SYCL(info));
  return nullptr;
}
