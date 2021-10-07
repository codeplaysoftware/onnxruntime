//Codeplay Software Ltd.

#pragma once

#include "core/common/logging/logging.h"
#include "core/framework/allocator.h"
#include <CL/sycl.hpp>

#include <algorithm>

namespace onnxruntime {

class SYCLAllocator : public IAllocator {
 public:
  SYCLAllocator(std::shared_ptr<cl::sycl::queue> q) : IAllocator(OrtMemoryInfo("sycl",
                                                                               OrtAllocatorType::OrtDeviceAllocator,
                                                                               OrtDevice(q->get_device().is_cpu() ? OrtDevice::CPU : OrtDevice::GPU,
                                                                                         OrtDevice::MemType::DEFAULT,
                                                                                         0),
                                                                               0,
                                                                               OrtMemTypeDefault)),
                                                      q_{q} {
  }

  void* Alloc(size_t size) override;
  void Free(void* p) override;

 private:
  std::shared_ptr<cl::sycl::queue> q_;
};
}  // namespace onnxruntime