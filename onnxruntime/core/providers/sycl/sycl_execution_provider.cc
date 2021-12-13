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

#include "core/providers/sycl/sycl_execution_provider.h"
#include "core/providers/sycl/sycl_execution_provider_info.h"
#include "core/providers/sycl/sycl_allocator.h"
#include "core/providers/sycl/sycl_data_transfer.h"
#include "core/providers/sycl/sycl_fwd.h"

#include "core/framework/data_transfer_manager.h"

#include "core/framework/kernel_registry.h"
#include "core/graph/constants.h"

#include <CL/sycl.hpp>

using namespace onnxruntime::common;
using namespace std;

namespace {
struct KernelRegistryAndStatus {
  shared_ptr<onnxruntime::KernelRegistry> kernel_registry = make_shared<onnxruntime::KernelRegistry>();
  Status st;
};
}  // namespace

namespace onnxruntime {

SYCLExecutionProvider::SYCLExecutionProvider() : IExecutionProvider{onnxruntime::kSyclExecutionProvider}, queue_{make_shared<cl::sycl::queue>(cl::sycl::default_selector{})} {
  LOGS_DEFAULT(INFO) << "SYCL EP instantiated using DEFAULT SYCL Selector : \n\tdevice's name : " << queue_->get_device().get_info<cl::sycl::info::device::name>() << "\n\tdevice's vendor : "
                     << queue_->get_device().get_info<cl::sycl::info::device::vendor>();
}

SYCLExecutionProvider::SYCLExecutionProvider(const SYCLExecutionProviderInfo& info) : IExecutionProvider{onnxruntime::kSyclExecutionProvider}, info_{info}, queue_{info.device_selector.size() == 0 ? make_shared<cl::sycl::queue>(cl::sycl::default_selector{}) : make_shared<cl::sycl::queue>(sycl_device_selector{info.device_selector, info.device_vendor})} {
  // Checking if device was correctly selected, otherwise LOG warning/error

  // Device vendor name check
  auto deviceInfo = queue_->get_device().get_info<cl::sycl::info::device::vendor>();
  // Upper case formatting for string check purposes
  for_each(deviceInfo.begin(), deviceInfo.end(), [](char& c) {
    c = ::toupper(c);
  });
  bool vendor_match = info.device_vendor.size() > 0 ? deviceInfo.find(info.device_vendor) != string::npos : true;

  // Device Selector (type) check
  bool selector_match = true;
  std::string device_type;

  if (!std::strcmp(info.device_selector.c_str(), "CPU")) {
    selector_match = queue_->get_device().is_cpu();
    device_type = selector_match ? "CPU" : "";
  } else if (!std::strcmp(info.device_selector.c_str(), "GPU")) {
    selector_match = queue_->get_device().is_gpu();
    device_type = selector_match ? "GPU" : "";
  } else if (!std::strcmp(info.device_selector.c_str(), "ACC")) {
    selector_match = queue_->get_device().is_accelerator();
    device_type = selector_match ? "ACCELERATOR" : "";
  } else if (!std::strcmp(info.device_selector.c_str(), "HOST")) {
    selector_match = queue_->get_device().is_host();
    device_type = selector_match ? "HOST" : "";
  } else {
    selector_match = true;
    device_type = selector_match ? "DEFAULT" : "";
  }

  // Warning if selected device doesn't match vendor name provided
  if (selector_match && !vendor_match) {
    LOGS_DEFAULT(WARNING) << "Specified SYCL Device Vendor Name : [" << info_.device_vendor << "] couldn't be found for device type : [" << device_type << "]";
  }

  // Log selected device informations (name & vendor)
  LOGS_DEFAULT(INFO)
      << "SYCL EP instantiated using Device Selector : \n\tdevice's type : " << device_type << "\n\tdevice's name : " << queue_->get_device().get_info<cl::sycl::info::device::name>() << "\n\tdevice's vendor : " << queue_->get_device().get_info<cl::sycl::info::device::vendor>();
}

SYCLExecutionProvider::~SYCLExecutionProvider() {
}

// Register Allocator
void SYCLExecutionProvider::RegisterAllocator(shared_ptr<AllocatorManager> allocator_manager) {
  // GetAllocator requires a device_id (first argument, 0 by default).
  // Should eventually change later once mapped to an openCL device_id
  auto sycl_alloc = allocator_manager->GetAllocator(info_.device_id, OrtMemTypeDefault);

  if (nullptr == sycl_alloc) {
    sycl_alloc = CreateSYCLAllocator(queue_, info_.device_id);
    allocator_manager->InsertAllocator(sycl_alloc);
  }
  TryInsertAllocator(sycl_alloc);

  // OrtMemTypeCPUOutput -- allocated by SYCLHostAllocator, used to copy SYCL device memory to CPU
  // Used by node MemcpyToHost only
  auto sycl_host_alloc = allocator_manager->GetAllocator(DEFAULT_CPU_ALLOCATOR_DEVICE_ID, OrtMemTypeCPUOutput);
  if (nullptr == sycl_host_alloc) {
    AllocatorCreationInfo sycl_host_memory_info(
        [](OrtDevice::DeviceId device_id) {
          return std::make_unique<SYCLHostAllocator>(device_id, "sycl_host");
        },
        DEFAULT_CPU_ALLOCATOR_DEVICE_ID);

    sycl_host_alloc = CreateAllocator(sycl_host_memory_info);
    allocator_manager->InsertAllocator(sycl_host_alloc);
  }
  TryInsertAllocator(sycl_host_alloc);
}

// Create Allocator
AllocatorPtr SYCLExecutionProvider::CreateSYCLAllocator(shared_ptr<cl::sycl::queue> q, OrtDevice::DeviceId device_id) {
  AllocatorCreationInfo sycl_memory_info(
      [&q](OrtDevice::DeviceId id) {
        return make_unique<SYCLAllocator>(q, id);
      },
      device_id,  //device_id is 0 by default
      false);     //usearena set to false, following parameters thus not needed

  return CreateAllocator(sycl_memory_info);
}

// Get Allocator
AllocatorPtr SYCLExecutionProvider::GetAllocator(int id, OrtMemType mem_type) const {
  return IExecutionProvider::GetAllocator(id, mem_type);
}

// Get DataTransfer
unique_ptr<IDataTransfer> SYCLExecutionProvider::GetDataTransfer() const {
  return make_unique<SYCLDataTransfer>(queue_);
}

cl::sycl::queue* SYCLExecutionProvider::GetQueue() const {
  return (queue_.get());
}

namespace sycl {
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 1, MemcpyFromHost);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 1, MemcpyToHost);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 7, 12, float, Add);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 13, 13, float, Add);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 7, 9, float, AveragePool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 10, 10, float, AveragePool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 7, 8, float, BatchNormalization);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 9, 13, float, BatchNormalization);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 14, 14, float, BatchNormalization);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 1, 10, float, Conv);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 12, 12, Dropout);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 1, 8, Flatten);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 9, 10, Flatten);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 11, 12, Flatten);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 7, 8, float, Gemm);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 9, 10, float, Gemm);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 11, 12, float, Gemm);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 1, 8, float, MatMul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 9, 12, float, MatMul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 1, 7, float, MaxPool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 8, 9, float, MaxPool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 1, 10, float, ReduceMean);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 11, 12, float, ReduceMean);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 6, 12, float, Relu);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 13, 13, float, Relu);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 1, 4, Reshape);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 5, 12, Reshape);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 13, 13, Reshape);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 1, 10, float, Softmax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 11, 12, float, Softmax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 1, 12, float, Transpose);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 14, float, Add);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 11, float, AveragePool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 15, float, BatchNormalization);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 11, float, Conv);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 13, Dropout);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 13, Flatten);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 13, float, Gemm);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 1, float, GlobalAveragePool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 1, float, GlobalMaxPool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 13, float, MatMul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 10, float, MaxPool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 11, float, MaxPool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 12, float, MaxPool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 13, float, ReduceMean);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 14, float, Relu);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 14, Reshape);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 13, float, Softmax);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 13, float, Transpose);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 1, MemcpyFromHost);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 1, MemcpyToHost);

template <>
KernelCreateInfo BuildKernelCreateInfo<void>() {
  KernelCreateInfo info;
  return info;
}

static Status RegisterSyclKernels(KernelRegistry& kernel_registry) {
  static const BuildKernelCreateInfoFn function_table[] = {
      BuildKernelCreateInfo<void>,  //default entry to avoid the list become empty after ops-reducing
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 1, MemcpyFromHost)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 1, MemcpyToHost)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 7, 12, float, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 13, 13, float, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 7, 9, float, AveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 10, 10, float, AveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 7, 8, float, BatchNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 9, 13, float, BatchNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 14, 14, float, BatchNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 1, 10, float, Conv)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 12, 12, Dropout)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 1, 8, Flatten)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 9, 10, Flatten)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 11, 12, Flatten)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 7, 8, float, Gemm)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 9, 10, float, Gemm)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 11, 12, float, Gemm)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 1, 8, float, MatMul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 9, 12, float, MatMul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 1, 7, float, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 8, 9, float, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 1, 10, float, ReduceMean)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 11, 12, float, ReduceMean)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 6, 12, float, Relu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 13, 13, float, Relu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 1, 4, Reshape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 5, 12, Reshape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 13, 13, Reshape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 1, 10, float, Softmax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 11, 12, float, Softmax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 1, 12, float, Transpose)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 14, float, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 11, float, AveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 15, float, BatchNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 11, float, Conv)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 13, Dropout)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 13, Flatten)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 13, float, Gemm)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 1, float, GlobalAveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 1, float, GlobalMaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 13, float, MatMul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 10, float, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 11, float, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 12, float, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 13, float, ReduceMean)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 14, float, Relu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 14, Reshape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 13, float, Softmax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kSyclExecutionProvider, kOnnxDomain, 13, float, Transpose)>,

  };

  for (auto& function_table_entry : function_table) {
    KernelCreateInfo info = function_table_entry();
    if (info.kernel_def != nullptr) {  // filter disabled entries where type is void
      ORT_RETURN_IF_ERROR(kernel_registry.Register(move(info)));
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

shared_ptr<KernelRegistry> SYCLExecutionProvider::GetKernelRegistry() const {
  static KernelRegistryAndStatus k = onnxruntime::sycl::GetSyclKernelRegistry();

  //Throw if the registry failed to initialize
  ORT_THROW_IF_ERROR(k.st);
  return k.kernel_registry;
}

}  // namespace onnxruntime
