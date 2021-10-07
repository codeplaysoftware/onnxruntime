// Codeplay Software Ltd.

#include "core/providers/sycl/sycl_execution_provider.h"
#include "core/providers/sycl/sycl_execution_provider_info.h"
#include "core/providers/sycl/sycl_allocator.h"
#include "core/graph/constants.h"
#include <CL/sycl.hpp>

namespace onnxruntime {

SYCLExecutionProvider::SYCLExecutionProvider() : IExecutionProvider{onnxruntime::kSyclExecutionProvider},
                                                 queue_{std::make_shared<cl::sycl::queue>(cl::sycl::default_selector{})} {
}

SYCLExecutionProvider::SYCLExecutionProvider(const SYCLExecutionProviderInfo& info)
    : IExecutionProvider{onnxruntime::kSyclExecutionProvider},
      info_{info},
      queue_{info.device_selector ? std::make_shared<cl::sycl::queue>(cl::sycl::gpu_selector{}) : std::make_shared<cl::sycl::queue>(cl::sycl::cpu_selector{})} {
  LOGS_DEFAULT(INFO) << "SYCL EP instantiated using selector : \n\tdevice's name : " << queue_->get_device().get_info<cl::sycl::info::device::name>() << "\n\tdevice's vendor : " << queue_->get_device().get_info<cl::sycl::info::device::vendor>();
}

SYCLExecutionProvider::~SYCLExecutionProvider() {
}

cl::sycl::queue* SYCLExecutionProvider::GetQueue() const {
  return (queue_.get());
}

// Register Allocator
void SYCLExecutionProvider::RegisterAllocator(std::shared_ptr<AllocatorManager> allocator_manager) {
  auto sycl_alloc = allocator_manager->GetAllocator(info_.device_selector ? 0 : 0, OrtMemTypeDefault);
  if (nullptr == sycl_alloc) {
    sycl_alloc = CreateSYCLAllocator(queue_);
    allocator_manager->InsertAllocator(sycl_alloc);
    LOGS_DEFAULT(INFO) << "SYCL allocator inserted within allocator_manager";
  }
  TryInsertAllocator(sycl_alloc);
}

// CreateSYCLAllocator
AllocatorPtr SYCLExecutionProvider::CreateSYCLAllocator(std::shared_ptr<cl::sycl::queue> q) {
  AllocatorCreationInfo default_memory_info(
      [&q](OrtDevice::DeviceId id) {
        // TODO : this log is a temporary fix to unused variable id. Not needed for SYCLAllocator
        LOGS_DEFAULT(INFO) << "Device Id : " << id;
        return std::make_unique<SYCLAllocator>(q);
      },
      0,       //device_id always 0. Should probably be tuned later !
      false);  //usearena set to false
               //4th argument is OrtArenaCfg arena_cfg0 which is {0, -1, -1, -1, -1} by default.
               //Not needed anyways because usearena is false

  return CreateAllocator(default_memory_info);
}

// GetAllocator
AllocatorPtr SYCLExecutionProvider::GetAllocator(int id, OrtMemType mem_type) const {
  return IExecutionProvider::GetAllocator(id, mem_type);
}

}  // namespace onnxruntime