// Codeplay Software Ltd.

#include <CL/sycl.hpp>
#include "sycl_data_transfer.h"

namespace onnxruntime {

SYCLDataTransfer::SYCLDataTransfer(std::shared_ptr<cl::sycl::queue> q) : queue_{q} {
}

common::Status SYCLDataTransfer::CopyTensor(const Tensor& src, Tensor& dst, int /*exec_queue_id*/) const {
  size_t bytes = src.SizeInBytes();
  const void* src_data = src.DataRaw();
  void* dst_data = dst.MutableDataRaw();

  auto& src_device = src.Location().device;
  auto& dst_device = dst.Location().device;

  // Optional log of sr/dst devices
  LOGS_DEFAULT(INFO) << "SYCL copy from device : " << src_device.ToString() << " to device : " << dst_device.ToString();

  queue_.get()->submit([&](cl::sycl::handler& cgh) {
    cgh.memcpy(dst_data, src_data, bytes);
  });
  
  queue_->wait();

  return Status::OK();
}
}  // namespace onnxruntime