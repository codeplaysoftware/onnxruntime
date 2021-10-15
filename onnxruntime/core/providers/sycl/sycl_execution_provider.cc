// Codeplay Software Ltd.

#include "core/providers/sycl/sycl_execution_provider.h"
#include "core/providers/sycl/sycl_execution_provider_info.h"
#include "core/providers/sycl/sycl_allocator.h"
#include "core/providers/sycl/sycl_data_transfer.h"
#include "core/providers/sycl/sycl_fwd.h"

#include "core/framework/kernel_registry.h"
#include "core/graph/constants.h"

#include <CL/sycl.hpp>

using namespace onnxruntime::common;

namespace {
struct KernelRegistryAndStatus {
  std::shared_ptr<onnxruntime::KernelRegistry> kernel_registry = std::make_shared<onnxruntime::KernelRegistry>();
  Status st;
};
}  // namespace

namespace onnxruntime {

SYCLExecutionProvider::SYCLExecutionProvider() : IExecutionProvider{onnxruntime::kSyclExecutionProvider}, queue_{std::make_shared<cl::sycl::queue>(cl::sycl::default_selector{})} {
}

SYCLExecutionProvider::SYCLExecutionProvider(const SYCLExecutionProviderInfo& info)
    : IExecutionProvider{onnxruntime::kSyclExecutionProvider}, info_{info}, queue_{info.device_selector ? std::make_shared<cl::sycl::queue>(cl::sycl::gpu_selector{}) : std::make_shared<cl::sycl::queue>(cl::sycl::cpu_selector{})} {
  LOGS_DEFAULT(INFO) << "SYCL EP instantiated using selector : \n\tdevice's name : " << queue_->get_device().get_info<cl::sycl::info::device::name>() << "\n\tdevice's vendor : " << queue_->get_device().get_info<cl::sycl::info::device::vendor>();
}

SYCLExecutionProvider::~SYCLExecutionProvider() {
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

// Create Allocator
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

// Get Allocator
AllocatorPtr SYCLExecutionProvider::GetAllocator(int id, OrtMemType mem_type) const {
  return IExecutionProvider::GetAllocator(id, mem_type);
}

// Get DataTransfer
std::unique_ptr<IDataTransfer> SYCLExecutionProvider::GetDataTransfer() const {
  return std::make_unique<SYCLDataTransfer>(queue_);
}

cl::sycl::queue* SYCLExecutionProvider::GetQueue() const {
  return (queue_.get());
}

namespace sycl {
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 7, 12, float, Add);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 1, 12, float, MatMul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 1, 12, float, Gemm);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 1, 12, float, Relu);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 1, 12, float, MaxPool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 1, 12, float, Conv);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 1, 12, float, AveragePool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 1, 12, float, Softmax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 1, 12, float, Flatten);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 1, 12, Dropout);

template <>
KernelCreateInfo BuildKernelCreateInfo<void>() {
  KernelCreateInfo info;
  return info;
}

static Status RegisterSyclKernels(KernelRegistry& kernel_registry) {
  static const BuildKernelCreateInfoFn function_table[] = {
      BuildKernelCreateInfo<void>,  //default entry to avoid the list become empty after ops-reducing
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 7, 12, float, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 1, 12, float, MatMul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 1, 12, float, Gemm)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 1, 12, float, Relu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 1, 12, float, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 1, 12, float, Conv)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 1, 12, float, AveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 1, 12, float, Softmax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 1, 12, float, Flatten)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 1, 12, Dropout)>,

  };

  for (auto& function_table_entry : function_table) {
    KernelCreateInfo info = function_table_entry();
    if (info.kernel_def != nullptr) {  // filter disabled entries where type is void
      ORT_RETURN_IF_ERROR(kernel_registry.Register(std::move(info)));
    }
  }

  return Status::OK();
}

KernelRegistryAndStatus GetSyclKernelRegistry() {
  KernelRegistryAndStatus ret;
  ret.st = RegisterSyclKernels(*ret.kernel_registry);
  return ret;
}

}  // namespace sycl

std::shared_ptr<KernelRegistry> SYCLExecutionProvider::GetKernelRegistry() const {
  static KernelRegistryAndStatus k = onnxruntime::sycl::GetSyclKernelRegistry();

  //Throw if the registry failed to initialize
  ORT_THROW_IF_ERROR(k.st);
  return k.kernel_registry;
}

}  // namespace onnxruntime
