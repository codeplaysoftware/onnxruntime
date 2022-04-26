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

#include "core/framework/data_types_internal.h"
#include "core/framework/op_kernel.h"
#include "core/common/status.h"
#include "core/common/common.h"

#include "sycl_fwd.h"
#include "sycl_execution_provider.h"

#include <CL/sycl.hpp>

namespace onnxruntime {
namespace sycl {

// ---------------------
// Base class for SYCL kernels (Needed to access EP's sycl queue when submitting
// kernels)
// ---------------------
class SyclKernel : public OpKernel {
 private:
  SYCLExecutionProvider* provider_;

 public:
  explicit SyclKernel(const OpKernelInfo& info)
      : OpKernel(info),
        provider_(const_cast<SYCLExecutionProvider*>(
            static_cast<const SYCLExecutionProvider*>(
                info.GetExecutionProvider()))) {}

  Status Compute(OpKernelContext* p_op_kernel_context) const override {
    auto s = ComputeInternal(p_op_kernel_context);

    if (!s.IsOK()) {
      LOGS_DEFAULT(INFO) << s.ErrorMessage();
    }

    return s;
  }

  inline void wait() const { Queue()->wait_and_throw(); }

  virtual Status ComputeInternal(
      OpKernelContext* p_op_kernel_context) const = 0;

  inline cl::sycl::queue* Queue() const { return provider_->GetQueue(); };

  inline std::unique_ptr<onnxruntime::IDataTransfer> GetDataTransfer() const {
    return provider_->GetDataTransfer();
  };

  inline int GetDeviceId() const { return provider_->GetDeviceId(); }
};
}  // namespace sycl
}  // namespace onnxruntime
