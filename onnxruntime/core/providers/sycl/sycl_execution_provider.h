// Codeplay Software Ltd.

#pragma once
#include <algorithm>

#include "core/framework/execution_provider.h"
namespace onnxruntime {

// Logical device representation.
class SYCLExecutionProvider : public IExecutionProvider {
 public:
  SYCLExecutionProvider();
  virtual ~SYCLExecutionProvider();
};

}  // namespace onnxruntime