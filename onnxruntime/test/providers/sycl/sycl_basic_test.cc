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

TEST(SYCLExecutionProviderTest, MetadataTestCPU) {
  SYCLExecutionProviderInfo info;
  info.device_selector = false;
  auto provider = std::make_unique<SYCLExecutionProvider>(info);
  EXPECT_TRUE(provider != nullptr);

  auto allocator_manager_ = std::make_shared<onnxruntime::AllocatorManager>();
  provider->RegisterAllocator(allocator_manager_);

  ASSERT_STREQ(provider->GetAllocator(0, OrtMemTypeDefault)->Info().name, "sycl");
}

TEST(SYCLExecutionProviderTest, MetadataTestGPU) {
  SYCLExecutionProviderInfo info;
  info.device_selector = true;
  auto provider = std::make_unique<SYCLExecutionProvider>(info);
  EXPECT_TRUE(provider != nullptr);

  auto allocator_manager_ = std::make_shared<onnxruntime::AllocatorManager>();
  provider->RegisterAllocator(allocator_manager_);

  ASSERT_STREQ(provider->GetAllocator(0, OrtMemTypeDefault)->Info().name, "sycl");
}

TEST(SYCLExecutionProviderTest, FunctionTest_1) {
  const ORTCHAR_T* model_file_name = ORT_TSTR("sycl_execution_provider_test_graph.onnx");

  {  // Create a model with a single Add node
    onnxruntime::Model model("singleNodeAddGraph", false, DefaultLoggingManager().DefaultLogger());
    auto& graph = model.MainGraph();
    std::vector<onnxruntime::NodeArg*> inputs;
    std::vector<onnxruntime::NodeArg*> outputs;

    // FLOAT tensor.
    ONNX_NAMESPACE::TypeProto float_tensor;
    float_tensor.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    // float_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
    float_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(2);
    float_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(3);

    auto& input_arg_1 = graph.GetOrCreateNodeArg("X", &float_tensor);
    auto& input_arg_2 = graph.GetOrCreateNodeArg("Y", &float_tensor);
    inputs.push_back(&input_arg_1);
    inputs.push_back(&input_arg_2);
    auto& output_arg = graph.GetOrCreateNodeArg("Z", &float_tensor);
    outputs.push_back(&output_arg);
    graph.AddNode("nodeAddFloat", "Add", "Single Add Node Example", inputs, outputs);

    ASSERT_STATUS_OK(graph.Resolve());
    ASSERT_STATUS_OK(onnxruntime::Model::Save(model, model_file_name));
  }

  std::vector<int64_t> dims_mul_x = {2, 3};
  std::vector<float> values_mul_x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  OrtValue ml_value_x;

  // Allocator manager needed in SYCL EP Case to register its allocator
  auto allocator_manager_ = std::make_shared<onnxruntime::AllocatorManager>();
  TestSyclExecutionProvider()->RegisterAllocator(allocator_manager_);

  CreateMLValue<float>(TestSyclExecutionProvider()->GetAllocator(0, OrtMemTypeDefault),
                       dims_mul_x, values_mul_x, &ml_value_x);
  OrtValue ml_value_y;
  CreateMLValue<float>(TestSyclExecutionProvider()->GetAllocator(0, OrtMemTypeDefault),
                       dims_mul_x, values_mul_x, &ml_value_y);

  NameMLValMap feeds;
  feeds.insert(std::make_pair("X", ml_value_x));
  feeds.insert(std::make_pair("Y", ml_value_y));

  // Verification is conducted by forcing the placement of at least one node to SYCL EP
  // and comparing with default CPU EP output
  SYCLExecutionProviderInfo info;
  info.device_selector = true;  // true : GPU Selector
  RunAndVerifyOutputsWithEP(model_file_name, "SYCLExecutionProviderTest.FunctionTest_1",
                            std::make_unique<SYCLExecutionProvider>(info),
                            feeds);
}

}  // namespace test
}  // namespace onnxruntime
