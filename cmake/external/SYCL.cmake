#/***************************************************************************
# *
# *  @license
# *  Copyright (C) Codeplay Software Limited
# *  Licensed under the Apache License, Version 2.0 (the "License");
# *  you may not use this file except in compliance with the License.
# *  You may obtain a copy of the License at
# *
# *      http://www.apache.org/licenses/LICENSE-2.0
# *
# *  For your convenience, a copy of the License has been included in this
# *  repository.
# *
# *  Unless required by applicable law or agreed to in writing, software
# *  distributed under the License is distributed on an "AS IS" BASIS,
# *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# *  See the License for the specific language governing permissions and
# *  limitations under the License.
# *
# *  @filename SYCL.cmake
# *
# **************************************************************************/

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-fsycl" is_dpcpp)
if(NOT is_dpcpp)
  set (is_computecpp ON)
endif()

# COMPUTE_CPP
if(is_computecpp)
  list(APPEND COMPUTECPP_USER_FLAGS
    -O3
    -fsycl-split-modules=20
    -mllvm -inline-threshold=10000
    -Xclang -cl-mad-enable
    -no-serial-memop
  )
  set(CMAKE_CXX_STANDARD 17)
  find_package(ComputeCpp REQUIRED)
  set(SYCL_INCLUDE_DIRS ${ComputeCpp_INCLUDE_DIRS})

# DPCPP
elseif(is_dpcpp)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__SYCL_DISABLE_NAMESPACE_INLINE__=ON -O3 -Xclang -cl-mad-enable")

  set(DPCPP_SYCL_TARGET spir64-unknown-unknown-sycldevice)

  find_package(DPCPP REQUIRED)
  get_target_property(SYCL_INCLUDE_DIRS DPCPP::DPCPP INTERFACE_INCLUDE_DIRECTORIES)
endif()
