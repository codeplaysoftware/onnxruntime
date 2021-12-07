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

#include <CL/sycl.hpp>
#include "core/providers/sycl/sycl_data_transfer.h"

namespace onnxruntime {

SYCLDataTransfer::SYCLDataTransfer(std::shared_ptr<cl::sycl::queue> q) : queue_{q} {
}

namespace sycl {

template <typename T>
common::Status SyclCopy(const Tensor& src, Tensor& dst, std::shared_ptr<cl::sycl::queue> queue_) {
  auto& src_device = src.Location().device;
  auto& dst_device = dst.Location().device;

  auto src_bytes_ = src.SizeInBytes();
  auto dst_bytes_ = dst.SizeInBytes();

  assert(src_bytes_ == dst_bytes_ && "Size mismatch for SYCL Tensors");

  // Empty tensor needs no copy
  if (!src_bytes_) {
    return Status::OK();
  }

  if (dst_device.Type() == OrtDevice::CPU && src_device.Type() != OrtDevice::CPU) {
    cl::sycl::buffer<T, 1>* src_data = const_cast<cl::sycl::buffer<T, 1>*>(src.Data<cl::sycl::buffer<T, 1>>());
    T* dst_data = dst.MutableData<T>();
    queue_->submit([&](cl::sycl::handler& cgh) {
            auto X_acc = cl::sycl::accessor<T, 1, cl::sycl::access::mode::read>(*src_data,
                                                                                cgh, cl::sycl::range<1>(src_bytes_ / sizeof(T)),
                                                                                cl::sycl::id<1>(src.ByteOffset() / sizeof(T)));
            cgh.copy(X_acc, dst_data);
          })
        .wait();

  } else if (src_device.Type() == OrtDevice::CPU && dst_device.Type() != OrtDevice::CPU) {
    cl::sycl::buffer<T, 1>* dst_data = dst.MutableData<cl::sycl::buffer<T, 1>>();
    const T* src_data = src.Data<T>();
    queue_->submit([&](cl::sycl::handler& cgh) {
            auto Y_acc = cl::sycl::accessor<T, 1, cl::sycl::access::mode::discard_write>(*dst_data,
                                                                                         cgh, cl::sycl::range<1>(dst_bytes_ / sizeof(T)),
                                                                                         cl::sycl::id<1>(dst.ByteOffset() / sizeof(T)));
            cgh.copy(src_data, Y_acc);
          })
        .wait();
  } else {
    cl::sycl::buffer<T, 1>* src_data = const_cast<cl::sycl::buffer<T, 1>*>(src.Data<cl::sycl::buffer<T, 1>>());
    cl::sycl::buffer<T, 1>* dst_data = dst.MutableData<cl::sycl::buffer<T, 1>>();
    queue_->submit([&](cl::sycl::handler& cgh) {
      auto X_acc = cl::sycl::accessor<T, 1, cl::sycl::access::mode::read>(*src_data,
                                                                          cgh, cl::sycl::range<1>(src_bytes_ / sizeof(T)),
                                                                          cl::sycl::id<1>(src.ByteOffset() / sizeof(T)));
      auto Y_acc = cl::sycl::accessor<T, 1, cl::sycl::access::mode::discard_write>(*dst_data,
                                                                                   cgh, cl::sycl::range<1>(dst_bytes_ / sizeof(T)),
                                                                                   cl::sycl::id<1>(dst.ByteOffset() / sizeof(T)));
      cgh.copy(X_acc, Y_acc);
    });
  }

  return Status::OK();
}
}  // namespace sycl

common::Status SYCLDataTransfer::CopyTensor(const Tensor& src, Tensor& dst, int /*exec_queue_id*/) const {
  switch (src.GetElementType()) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      return sycl::SyclCopy<float>(src, dst, queue_);
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE:
      return sycl::SyclCopy<double>(src, dst, queue_);
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_INT8:
      return sycl::SyclCopy<int8_t>(src, dst, queue_);
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
      return sycl::SyclCopy<uint8_t>(src, dst, queue_);
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_INT64:
      return sycl::SyclCopy<int64_t>(src, dst, queue_);
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
      return sycl::SyclCopy<cl::sycl::half>(src, dst, queue_);
      break;
    default:
      ORT_THROW("Unexpected data type");
  }
}
}  // namespace onnxruntime
