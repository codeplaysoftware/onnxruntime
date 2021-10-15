// Codeplay Software Ltd.

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
// Base class for SYCL kernels (Needed to access EP's sycl queue when submitting kernels)
// ---------------------
class SyclKernel : public OpKernel {
 private:
  SYCLExecutionProvider* provider_;

 public:
  explicit SyclKernel(const OpKernelInfo& info) : OpKernel(info),
                                                  provider_(const_cast<SYCLExecutionProvider*>(static_cast<const SYCLExecutionProvider*>(info.GetExecutionProvider()))) {
  }

  Status Compute(OpKernelContext* p_op_kernel_context) const override {
    auto s = ComputeInternal(p_op_kernel_context);

    if (!s.IsOK()) {
      LOGS_DEFAULT(INFO) << s.ErrorMessage();
    }

    return s;
  }

  virtual Status ComputeInternal(OpKernelContext* p_op_kernel_context) const = 0;

  inline cl::sycl::queue* Queue() const { return provider_->GetQueue(); };

  inline std::unique_ptr<onnxruntime::IDataTransfer> GetDataTransfer() const { return provider_->GetDataTransfer(); };

  inline int GetDeviceId() const {
    return provider_->GetDeviceId();
  }
};
}  // namespace sycl
}  // namespace onnxruntime
