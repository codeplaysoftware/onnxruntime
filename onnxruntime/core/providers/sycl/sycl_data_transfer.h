// Codeplay Software Ltd.

#pragma once

#include "core/common/logging/logging.h"
#include "core/framework/data_transfer.h"
#include "core/framework/ortdevice.h"
#include "core/framework/tensor.h"
#include <CL/sycl.hpp>
#include <algorithm>
#include <iostream>

namespace onnxruntime {

class SYCLDataTransfer : public IDataTransfer {
 public:
  SYCLDataTransfer() = default;
  SYCLDataTransfer(std::shared_ptr<cl::sycl::queue> q);
  ~SYCLDataTransfer(){};

  // Whether copy can be made or not (TODO : Implement an actual copy check )
  bool CanCopy(const OrtDevice& src_device, const OrtDevice& dst_device) const {
    LOGS_DEFAULT(INFO) << "Source device : " << src_device.ToString();
    LOGS_DEFAULT(INFO) << "Destination Device" << dst_device.ToString();
    return true;
  }

  common::Status CopyTensor(const Tensor& src, Tensor& dst, int exec_queue_id) const;

 private:
  std::shared_ptr<cl::sycl::queue> queue_;
};

}  // namespace onnxruntime