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
#include "core/providers/sycl/sycl_execution_provider_info.h"

#include <string>
#include <CL/sycl.hpp>

#include <iostream>

using namespace std;

namespace onnxruntime {

class sycl_device_selector : public cl::sycl::device_selector {
 public:
  explicit sycl_device_selector() = default;

  explicit sycl_device_selector(std::string device_selector,
                                std::string device_vendor)
      : cl::sycl::device_selector{},
        device_selector_{device_selector},
        device_vendor_{device_vendor} {
    empty_selector_ = true;
  }

  virtual ~sycl_device_selector() {
    if (!one_device_available_ && !empty_selector_) {
      LOGS_DEFAULT(ERROR)
          << "No SYCL device with the given type could be found.";
    }
    empty_selector_ =
        false;  // Empty selector is destroyed before scoring the operational
                // selectors Thus, evaluation of device selection will be
                // considered starting next constructed selectors (using the
                // class implicit copy constructor)
  }

  // Scoring function called by SYCL for all available openCL platforms
  // returning highest score for selected device based on device nature
  // and device vendor
  int operator()(const cl::sycl::device& dev) const override;

 private:
  std::string device_selector_{""};  // Device type (Capital Letters)
  std::string device_vendor_{""};    // Device vendor name

  static bool
      one_device_available_;  // True if at least one device has a score > 0

  static bool
      empty_selector_;  // This static attribute is used as a work-around an
                        // initial instantiation of a sycl_device_selector
                        // object using explicit args constructor which is empty
                        // and doesn't serve in scoring. Its value is false by
                        // default and is set to true in the lifetime of the
                        // empty selector to avoid misleading Error logs when
                        // destroying the selector objects.
};
}  // namespace onnxruntime
