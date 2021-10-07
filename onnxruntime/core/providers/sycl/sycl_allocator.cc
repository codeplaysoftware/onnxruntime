// Codeplay Software Ltd.

#include "sycl_allocator.h"
#include "core/framework/allocatormgr.h"
#include <CL/sycl.hpp>

namespace onnxruntime {

void* SYCLAllocator::Alloc(size_t size) {
  void* p = nullptr;
  if (size > 0) {
    p = cl::sycl::malloc_device(size, *q_.get());
  }
  LOGS_DEFAULT(INFO) << "Memory allocated with SYCL [ " << size << " bytes ]";
  return p;
}

void SYCLAllocator::Free(void* p) {
  LOGS_DEFAULT(INFO) << "Memory freed with SYCL";
  cl::sycl::free(p, *q_.get());
}
}  // namespace onnxruntime
