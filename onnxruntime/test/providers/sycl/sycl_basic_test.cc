// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/logging/logging.h"
#include "core/providers/sycl/sycl_execution_provider.h"
#include "core/providers/sycl/sycl_provider_factory.h"
#include "core/session/inference_session.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/framework/test_utils.h"
#include "test/util/include/asserts.h"
#include "test/util/include/default_providers.h"
#include "test/util/include/inference_session_wrapper.h"
#include "test/util/include/test/test_environment.h"
#include "test/util/include/test_utils.h"

#if !defined(ORT_MINIMAL_BUILD)
// if this is a full build we need the provider test utils
#include "test/providers/provider_test_utils.h"
#endif  // !(ORT_MINIMAL_BUILD)

#include "gtest/gtest.h"
#include "gmock/gmock.h"

using namespace std;
using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::logging;

namespace onnxruntime {
namespace test {

// This test should not be disabled (SYCL Default Selector)
TEST(SYCLExecutionProviderTest, MetadataTestDEFAULT) {
  SYCLExecutionProviderInfo info;
  info.device_selector = "";
  info.device_vendor = "";
  auto provider = std::make_unique<SYCLExecutionProvider>(info);
  EXPECT_TRUE(provider != nullptr);

  auto allocator_manager_ = std::make_shared<onnxruntime::AllocatorManager>();
  provider->RegisterAllocator(allocator_manager_);

  ASSERT_STREQ(provider->GetAllocator(0, OrtMemTypeDefault)->Info().name, "sycl");
}

// This test can be disabled when testing non CPU targets
TEST(SYCLExecutionProviderTest, MetadataTestCPU) {
  SYCLExecutionProviderInfo info;
  info.device_selector = "CPU";  // And Device ID being 0 by default
  info.device_vendor = "";
  auto provider = std::make_unique<SYCLExecutionProvider>(info);
  EXPECT_TRUE(provider != nullptr);

  auto allocator_manager_ = std::make_shared<onnxruntime::AllocatorManager>();
  provider->RegisterAllocator(allocator_manager_);

  ASSERT_STREQ(provider->GetAllocator(0, OrtMemTypeDefault)->Info().name, "sycl");
}

// This test can be disabled when testing non GPU targets
TEST(SYCLExecutionProviderTest, MetadataTestGPU) {
  SYCLExecutionProviderInfo info;
  info.device_selector = "GPU";  // And Device ID being 0 by default
  info.device_vendor = "";
  auto provider = std::make_unique<SYCLExecutionProvider>(info);
  EXPECT_TRUE(provider != nullptr);

  auto allocator_manager_ = std::make_shared<onnxruntime::AllocatorManager>();
  provider->RegisterAllocator(allocator_manager_);

  ASSERT_STREQ(provider->GetAllocator(0, OrtMemTypeDefault)->Info().name, "sycl");
}

}  // namespace test
}  // namespace onnxruntime
