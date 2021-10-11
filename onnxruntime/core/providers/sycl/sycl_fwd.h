// Codeplay Software Ltd.

#pragma once

#include "core/framework/op_kernel.h"

namespace onnxruntime {

namespace sycl {
template <typename T>
KernelCreateInfo BuildKernelCreateInfo();
}  // namespace sycl
}  // namespace onnxruntime
