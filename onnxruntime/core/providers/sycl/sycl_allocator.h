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

#include "core/common/logging/logging.h"
#include "core/framework/allocator.h"
#include <CL/sycl.hpp>

#include <algorithm>

namespace onnxruntime {

// Base SYCL EP Allocator using buffer, targeting devices of type OrtDevice::SYCL_DEVICE
// with memory type OrtDevice::MemType::SYCL_MEMORY
class SYCLAllocator : public IAllocator {
 public:
  SYCLAllocator(std::shared_ptr<cl::sycl::queue> q, OrtDevice::DeviceId device_id) : IAllocator(OrtMemoryInfo("sycl", OrtAllocatorType::OrtDeviceAllocator, OrtDevice(OrtDevice::SYCL_DEVICE, OrtDevice::MemType::SYCL_MEMORY, device_id), device_id, OrtMemTypeDefault)), q_{q} {
  }

  void* Alloc(size_t) override;
  void Free(void*) override;
  bool SupportPointerArithmetic() const override {
    return false;
  }
  void* TypeAlloc(size_t, int32_t) override;

 private:
  std::shared_ptr<cl::sycl::queue> q_;
};

// SYCL Host allocator targetting host device of type OrtDevice::CPU
// and memory type OrtDevice::MemType::DEFAULT. Such a memory is allocated
// by SYCL EP and used by CPU EP for computation (for e.g. when immediate
// output of a SYCL Node is consumed by a CPU EP Node).
class SYCLHostAllocator : public IAllocator {
 public:
  SYCLHostAllocator(OrtDevice::DeviceId device_id, const char* name)
      : IAllocator(
            OrtMemoryInfo(name, OrtAllocatorType::OrtDeviceAllocator,
                          OrtDevice(OrtDevice::CPU, OrtDevice::MemType::DEFAULT, device_id),
                          device_id, OrtMemTypeCPUOutput)) {
  }

  void* Alloc(size_t size) override;
  void Free(void* p) override;
};

}  // namespace onnxruntime
