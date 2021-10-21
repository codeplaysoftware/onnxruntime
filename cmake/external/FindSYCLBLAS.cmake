
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
