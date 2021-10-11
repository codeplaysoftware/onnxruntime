// Codeplay Software Ltd.

#pragma once
#include <algorithm>

#include "core/framework/execution_provider.h"
#include "core/providers/sycl/sycl_execution_provider_info.h"
#include "CL/sycl.hpp"

namespace onnxruntime {

// Logical device representation.
class SYCLExecutionProvider : public IExecutionProvider {
 public:
  SYCLExecutionProvider();
  explicit SYCLExecutionProvider(const SYCLExecutionProviderInfo& info);
  virtual ~SYCLExecutionProvider();

  int GetDeviceId() const override { return (int)info_.device_selector; }

  cl::sycl::queue* GetQueue() const;

  AllocatorPtr GetAllocator(int id, OrtMemType mem_type) const override;

  void RegisterAllocator(std::shared_ptr<AllocatorManager> allocator_manager) override;

  static AllocatorPtr CreateSYCLAllocator(std::shared_ptr<cl::sycl::queue> q);

  std::unique_ptr<onnxruntime::IDataTransfer> GetDataTransfer() const override;

  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;

 private:
  SYCLExecutionProviderInfo info_;
  std::shared_ptr<cl::sycl::queue> queue_;
};

}  // namespace onnxruntime