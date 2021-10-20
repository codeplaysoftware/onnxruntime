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
