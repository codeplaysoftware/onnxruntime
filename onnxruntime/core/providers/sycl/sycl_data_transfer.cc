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

common::Status SYCLDataTransfer::CopyTensor(const Tensor& src, Tensor& dst, int /*exec_queue_id*/) const {
  size_t size = src.SizeInBytes();
  const uint8_t* src_data = static_cast<const uint8_t*>(src.DataRaw());
  uint8_t* dst_data = static_cast<uint8_t*>(dst.MutableDataRaw());

  auto& src_device = src.Location().device;
  auto& dst_device = dst.Location().device;

  // Optional log of sr/dst devices
  LOGS_DEFAULT(INFO) << "SYCL copy from device : " << src_device.ToString() << " to device : " << dst_device.ToString();

  if (dst_device.Type() == OrtDevice::CPU && src_device.Type() != OrtDevice::CPU) {
    cl::sycl::buffer<uint8_t, 1> X_buffer{src_data,
                                          cl::sycl::range<1>{size},
                                          {cl::sycl::property::buffer::context_bound{queue_->get_context()},
                                           cl::sycl::property::buffer::use_host_ptr{}}};

    queue_->submit([&](cl::sycl::handler& cgh) {
      auto X_acc = X_buffer.get_access<cl::sycl::access::mode::read>(cgh);
      cgh.copy(X_acc, dst_data);
    });
  } else if (src_device.Type() == OrtDevice::CPU && dst_device.Type() != OrtDevice::CPU) {
    cl::sycl::buffer<uint8_t, 1> Y_buffer{dst_data,
                                          cl::sycl::range<1>{size},
                                          {cl::sycl::property::buffer::context_bound{queue_->get_context()},
                                           cl::sycl::property::buffer::use_host_ptr{}}};

    queue_->submit([&](cl::sycl::handler& cgh) {
      auto Y_acc = Y_buffer.get_access<cl::sycl::access::mode::discard_write>(cgh);
      cgh.copy(src_data, Y_acc);
    });
  } else {
    cl::sycl::buffer<uint8_t, 1> X_buffer{src_data,
                                          cl::sycl::range<1>{size},
                                          {cl::sycl::property::buffer::context_bound{queue_->get_context()},
                                           cl::sycl::property::buffer::use_host_ptr{}}};

    cl::sycl::buffer<uint8_t, 1> Y_buffer{dst_data,
                                          cl::sycl::range<1>{size},
                                          {cl::sycl::property::buffer::context_bound{queue_->get_context()},
                                           cl::sycl::property::buffer::use_host_ptr{}}};

    queue_->submit([&](cl::sycl::handler& cgh) {
      auto X_acc = X_buffer.get_access<cl::sycl::access::mode::read>(cgh);
      auto Y_acc = Y_buffer.get_access<cl::sycl::access::mode::discard_write>(cgh);
      cgh.copy(X_acc, Y_acc);
    });
  }
  queue_->wait();

  return Status::OK();
}
}  // namespace onnxruntime
