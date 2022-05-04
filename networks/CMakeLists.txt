# Copyright Codeplay Software Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

cmake_minimum_required(VERSION 3.13)

project(ORT_MODEL_RUNNER_APP LANGUAGES CXX)

# ORT path & library
find_path(ONNX_RUNTIME_SESSION_INCLUDE_DIRS onnxruntime_cxx_api.h
          HINTS ${ONNX_RUNTIME_ROOT}/include/onnxruntime/core/session/)
find_library(
  ONNX_RUNTIME_LIB onnxruntime
  HINTS
  PATHS ${ONNX_RUNTIME_ROOT}/build/Linux/${CMAKE_BUILD_TYPE})

# ORT Execution providers to enable during build
option(USE_CUDA "Use CUDA Execution Provider" OFF)
option(USE_SYCL "Use SYCL Execution Provider" OFF)

if(USE_CUDA)
    message("CUDA IS used")
    add_definitions(-DCUDA_EP)
endif()
if(USE_SYCL)
    find_path(
      ONNX_RUNTIME_PROVIDERS_INCLUDE_DIRS sycl_provider_factory.h 
      HINTS 
      PATHS ${ONNX_RUNTIME_ROOT}/include/onnxruntime/core/providers/sycl/)
    add_definitions(-DSYCL_EP)
endif()

set(EXE_NAME "testNetwork")
add_executable(${EXE_NAME} test_network.cpp)
target_compile_features(${EXE_NAME} PRIVATE cxx_std_17)
if(USE_CUDA)
target_compile_definitions(${EXE_NAME} PRIVATE CUDA_EP)
elseif(USE_SYCL)
target_compile_definitions(${EXE_NAME} PRIVATE SYCL_EP)
endif()
target_include_directories(
  ${EXE_NAME} PRIVATE ${ONNX_RUNTIME_SESSION_INCLUDE_DIRS}
                      ${ONNX_RUNTIME_PROVIDERS_INCLUDE_DIRS})
target_link_libraries(${EXE_NAME} PRIVATE ${ONNX_RUNTIME_LIB})
