// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include "core/framework/kernel_registry.h"

namespace onnxruntime {
namespace contrib {
void RegisterContribKernels(std::function<void(KernelCreateInfo&&)> create_fn);
} // namespace contrib
}  // namespace onnxruntime
