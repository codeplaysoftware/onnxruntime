// Codeplay Software Ltd.

#include "onnxruntime_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \param device_selector SYCL device selector, temporary int for CPU : 0, GPU : 1
 */
ORT_API_STATUS(OrtSessionOptionsAppendExecutionProvider_SYCL, _In_ OrtSessionOptions* options, int device_selector);  //device selector say 0 : cpu, 1 gpu

#ifdef __cplusplus
}
#endif
