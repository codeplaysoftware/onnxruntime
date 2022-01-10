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

#include "core/providers/sycl/sycl_device_selector.h"

using namespace std;

namespace onnxruntime {

bool sycl_device_selector::one_device_available_ = false;
bool sycl_device_selector::empty_selector_ = false;

int sycl_device_selector::operator()(const cl::sycl::device& dev) const {
  int score = 0;

  // Checking device nature
  // score will be assigned 1 when the device is of the same
  // nature provided by user. Others will be 0.
  if (dev.is_gpu()) {
    score = !strcmp(device_selector_.c_str(), "GPU");

  } else if (dev.is_cpu()) {
    score = !strcmp(device_selector_.c_str(), "CPU");

  } else if (dev.is_accelerator()) {
    score = !strcmp(device_selector_.c_str(), "ACC");

  } else if (dev.is_host()) {
    score = !strcmp(device_selector_.c_str(), "HOST");

  } else {
    LOGS_DEFAULT(WARNING) << "No SYCL device was found ! ";
  }

  // Checking device vendor
  // only applies for platform(s) selected previously (score=1)
  // score will be assigned 2 if vendor name of the selected
  // platform 'matches' the vendor name provided by user.
  // otherwise, score remains = 1 and a warning is logged
  // since device of target nature will be used but its vendor
  // doesn't match the name provided by user.
  if (score) {
    one_device_available_ = true;  // At least one device will be selected (true)
    if (device_vendor_.size() > 0) {
      auto deviceInfo = dev.get_info<cl::sycl::info::device::vendor>();

      for_each(deviceInfo.begin(), deviceInfo.end(), [](char& c) {
        c = ::toupper(c);
      });

      bool vendor_match = deviceInfo.find(device_vendor_) != string::npos;  // true if device's vendor is found
      score += (int)vendor_match;                                           // score = 2 if device's vendor is matched
    }
  }

  return score;
}

}  // namespace onnxruntime
