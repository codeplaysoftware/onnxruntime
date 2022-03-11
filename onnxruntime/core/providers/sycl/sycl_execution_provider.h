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
#include <algorithm>

#include "core/framework/execution_provider.h"
#include "core/providers/sycl/sycl_execution_provider_info.h"
#include "core/providers/sycl/sycl_device_selector.h"
namespace onnxruntime {

// Logical device representation.
class SYCLExecutionProvider : public IExecutionProvider {
 public:
  SYCLExecutionProvider();
  explicit SYCLExecutionProvider(const SYCLExecutionProviderInfo& info);
  virtual ~SYCLExecutionProvider();

  int GetDeviceId() const override { return (int)info_.device_id; }

  cl::sycl::queue* GetQueue() const;

  AllocatorPtr GetAllocator(int id, OrtMemType mem_type) const override;

  void RegisterAllocator(
      std::shared_ptr<AllocatorManager> allocator_manager) override;

  static AllocatorPtr CreateSYCLAllocator(std::shared_ptr<cl::sycl::queue> q,
                                          OrtDevice::DeviceId device_id);

  std::unique_ptr<onnxruntime::IDataTransfer> GetDataTransfer() const override;

  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;

 private:
  SYCLExecutionProviderInfo info_;
  std::shared_ptr<cl::sycl::queue> queue_;
};

}  // namespace onnxruntime