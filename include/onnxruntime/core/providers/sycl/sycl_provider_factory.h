// Codeplay Software Ltd.

#include "onnxruntime_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \param device_id SYCL device id (TODO : Map it to OpenCL device_id + Combined with device_selector)
 */
ORT_API_STATUS(OrtSessionOptionsAppendExecutionProvider_SYCL, _In_ OrtSessionOptions* options, int device_id);

#ifdef __cplusplus
}
#endif
