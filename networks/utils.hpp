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

#include <algorithm>
#include <fstream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "imagenet_labels.hpp"

namespace utils {

template <typename T>
inline T product(const std::vector<T>& input) {
  return std::accumulate(input.begin(), input.end(), 1, std::multiplies<T>());
}

inline std::vector<float> readImage(std::string const& name,
                                    const std::vector<size_t>& input_dims) {
  std::ifstream file(name, std::ios_base::in);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file " + name);
  }
  std::vector<float> output(product(input_dims));
  float val = 0;
  int i = 0;
  for (std::string line; std::getline(file, line); i++) {
    std::istringstream in(line);
    in >> val;
    output[i] = val;
  }

  file.close();
  return output;
}

template <typename T>
inline size_t argmax(const std::vector<T>& input) {
  const auto index = std::max_element(input.begin(), input.end());
  return std::distance(input.begin(), index);
}

inline std::string indexToLabel(size_t idx) {
  if (idx > idxToClsMap.size()) {
    std::cerr << "Invalid index\n";
    return "";
  }

  return std::string(idxToClsMap[idx]);
}

inline std::string str_tolower(std::string str) {
  std::transform(str.begin(), str.end(), str.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  return str;
}

inline OrtLoggingLevel stringToLogLevel(const std::string& logLevelStr) {
  const int logLevelInt = std::stoi(logLevelStr);
  switch (logLevelInt) {
#define CASE(INT_VAL, RETURN_VAL) \
  case INT_VAL:                   \
    return OrtLoggingLevel::ORT_LOGGING_LEVEL_##RETURN_VAL;

    CASE(0, VERBOSE)
    CASE(1, INFO)
    CASE(2, WARNING)
    CASE(3, ERROR)
    CASE(4, FATAL)
    default: {
      std::cerr << "Unknown logging level " << logLevelStr << "\n";
      return ORT_LOGGING_LEVEL_INFO;
    }
#undef CASE
  }
}

inline GraphOptimizationLevel strToGraphOptLevel(const std::string& str) {
  if (str == "disable_all") return GraphOptimizationLevel::ORT_DISABLE_ALL;
  if (str == "enable_basic") return GraphOptimizationLevel::ORT_ENABLE_BASIC;
  if (str == "enable_extended")
    return GraphOptimizationLevel::ORT_ENABLE_EXTENDED;
  if (str == "enable_all") return GraphOptimizationLevel::ORT_ENABLE_ALL;

  std::cerr << "Unknow/unsupported graph optimization level " << str << "\n";
  return GraphOptimizationLevel::ORT_DISABLE_ALL;
}

inline bool hasExtension(const std::string& fileName,
                         const std::string& expectedExt) {
  size_t extension_index = fileName.rfind(".");
  return fileName.substr(extension_index + 1) == expectedExt;
}

}  // namespace utils
