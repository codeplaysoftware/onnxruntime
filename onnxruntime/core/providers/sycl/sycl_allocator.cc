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
#include "core/framework/data_types.h"
#include "core/framework/allocatormgr.h"
#include <CL/sycl.hpp>

namespace onnxruntime {

namespace sycl {

template <typename T>
inline void* SyclAlloc(size_t size, std::shared_ptr<cl::sycl::queue> q_) {
  cl::sycl::buffer<T, 1>* X_buffer = nullptr;
  if (size > 0) {
    X_buffer = new cl::sycl::buffer<T, 1>{
        cl::sycl::range<1>{size / sizeof(T)},
        {cl::sycl::property::buffer::context_bound{q_->get_context()}}};
  }
  LOGS_DEFAULT(INFO) << "Memory allocated with SYCL [ " << size << " bytes ]";
  return reinterpret_cast<void*>(X_buffer);
}
}  // namespace sycl

void* SYCLAllocator::Alloc(size_t size) {
  auto type = ONNX_NAMESPACE::TensorProto_DataType_UINT8;
  return TypeAlloc(size, type);
}

void* SYCLAllocator::TypeAlloc(size_t size, int32_t dtype) {
  switch (dtype) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      return sycl::SyclAlloc<float>(size, q_);
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE:
      return sycl::SyclAlloc<double>(size, q_);
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_INT8:
      return sycl::SyclAlloc<int8_t>(size, q_);
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
      return sycl::SyclAlloc<uint8_t>(size, q_);
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
      return sycl::SyclAlloc<cl::sycl::half>(size, q_);
      break;
    default:
      ORT_THROW("Unexpected data type");
  }
}

void SYCLAllocator::Free(void* p) {
  LOGS_DEFAULT(INFO) << "Memory freed with SYCL";
  delete reinterpret_cast<cl::sycl::buffer<uint8_t, 1>*>(p);
}
}  // namespace onnxruntime
