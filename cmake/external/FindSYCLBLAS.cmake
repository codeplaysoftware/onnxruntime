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
# *  @filename FindSYCLBLAS.cmake
# *
# **************************************************************************/
# Sets the following variables:
#   SYCLBLAS_FOUND        - whether the system has SYCLBLAS
#   SYCLBLAS_INCLUDE_DIRS - the SYCLBLAS include directory

find_library(SYCLBLAS_LIBRARY
   NAMES sycl_blas libsycl_blas
   PATH_SUFFIXES build
   HINTS ${onnxruntime_SYCLBLAS_HOME}
   DOC "The SYCLBLAS shared library"
)

find_path(SYCLBLAS_INCLUDE_DIR
  NAMES sycl_blas.h
  PATH_SUFFIXES include
  HINTS ${onnxruntime_SYCLBLAS_HOME}
  DOC "The SYCLBLAS include directory"
)

find_path(SYCLBLAS_VPTR_INCLUDE_DIR
  NAMES vptr/virtual_ptr.hpp
  PATH_SUFFIXES external/computecpp-sdk/include
  HINTS ${onnxruntime_SYCLBLAS_HOME}
  DOC "The SYCLBLAS virtual pointer include directory"
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(SYCLBLAS
  FOUND_VAR SYCLBLAS_FOUND
  REQUIRED_VARS SYCLBLAS_LIBRARY
                SYCLBLAS_INCLUDE_DIR
                SYCLBLAS_VPTR_INCLUDE_DIR
)

mark_as_advanced(SYCLBLAS_FOUND
                 SYCLBLAS_LIBRARY
                 SYCLBLAS_VPTR_INCLUDE_DIR
                 SYCLBLAS_INCLUDE_DIR
)

if(SYCLBLAS_FOUND)
  set(SYCLBLAS_INCLUDE_DIRS
    ${SYCLBLAS_INCLUDE_DIR}
    ${SYCLBLAS_VPTR_INCLUDE_DIR}
  )
endif()

if(SYCLBLAS_FOUND AND NOT TARGET SYCLBLAS::sycl_blas)
  add_library(SYCLBLAS::sycl_blas SHARED IMPORTED)
  set_target_properties(SYCLBLAS::sycl_blas PROPERTIES
    IMPORTED_LOCATION "${SYCLBLAS_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${SYCLBLAS_INCLUDE_DIRS}"
  )
endif()
