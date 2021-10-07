// Codeplay Software Ltd.

#pragma once

#include <memory>

#include "core/providers/providers.h"
#include "core/providers/sycl/sycl_execution_provider_info.h"

namespace onnxruntime {

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_SYCL(const SYCLExecutionProviderInfo& info);

}  // namespace onnxruntime
