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

  bool CanCopy(const OrtDevice& src_device,
               const OrtDevice& dst_device) const override;

  common::Status CopyTensor(const Tensor& src, Tensor& dst,
                            int exec_queue_id) const override;

 private:
  std::shared_ptr<cl::sycl::queue> queue_;
};

}  // namespace onnxruntime
