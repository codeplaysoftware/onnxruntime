// Codeplay Software Ltd.

#include "core/providers/sycl/sycl_execution_provider.h"
#include "core/graph/constants.h"
#include <CL/sycl.hpp>
#include <iostream>

namespace onnxruntime {

SYCLExecutionProvider::SYCLExecutionProvider() : IExecutionProvider{onnxruntime::kSyclExecutionProvider} {
}

SYCLExecutionProvider::~SYCLExecutionProvider() {
}

}  // namespace onnxruntime