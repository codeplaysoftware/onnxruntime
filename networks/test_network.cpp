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

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <chrono>

#ifdef SYCL_EP
#include "sycl_provider_factory.h"
#endif

#include <onnxruntime_cxx_api.h>

#include "arg_parser.hpp"
#include "utils.hpp"

int main(int argc, char** argv) {
  const std::string epOpt = "--ep";
  const std::string modelOpt = "--model";
  const std::string imageOpt = "--image";
#ifdef SYCL_EP
  const std::string deviceOpt = "--device";
  const std::string vendorOpt = "--vendor";
#endif
  const std::string logLevelOpt = "--log_level";
  const std::string graphOptLevelOpt = "--graph_opt_level";

  ArgParser argParser(argc, argv);
  argParser.addOptWithSupportedValues(epOpt, "Execution provider to use",
                                      {"SYCL", "CUDA"}, /*required*/ true);
  argParser.addOpt(modelOpt, "Path to ONNX model file");
  argParser.addOpt(imageOpt, "Path to image file");

#ifdef SYCL_EP
  argParser.addOptWithSupportedValues(deviceOpt, "Device selector to use",
                                      {"CPU", "cpu", "GPU", "gpu", "ACC", "acc",
                                       "HOST", "host", "DEFAULT", "default"},
                                      /*required*/ false);  // "" By default
  argParser.addOpt(vendorOpt, "Device vendor name to specify",
                   /*required*/ false);  // "" By default
#endif 

  argParser.addOptWithSupportedValues(
      logLevelOpt,
      "Logging level to use (0=verbose, 1=info, 2=warning, 3=error, 4=fatal)",
      {"0", "1", "2", "3", "4"},
      /*required*/ false);
  argParser.addOptWithSupportedValues(
      graphOptLevelOpt, "The graph optimization level to use",
      {"disable_all", "enable_basic", "enable_extended", "enable_all"},
      /*required*/ false);  // "disable_all" by default

  if (!argParser.parseArgs()) {
    argParser.printHelp();
    std::exit(1);
  }
  // Retrieve custom execution provider (if passed) from CL
  // Samples should be built with the given EP enabled first
  std::string customEP = "SYCL";
  if (argParser.hasValueForOpt(epOpt)) {
    customEP = argParser.getValueForOpt(epOpt);
  }

  std::vector<std::string> imagePaths = {"../10.txt"};
  if (argParser.hasValueForOpt(imageOpt)) {
    imagePaths = {argParser.getValueForOpt(imageOpt)};
    const std::string expectedExt = "txt";
    for (const auto& imPath : imagePaths) {
      if (!utils::hasExtension(imPath, expectedExt)) {
        std::cerr << "One or more input image files don't have the expected "
                     "extension ."
                  << expectedExt << "\n";
        return 1;
      }
    }
  }

  // Specify path to the model
  std::string model = "../models/vgg16-12.onnx";
  if (argParser.hasValueForOpt(modelOpt)) {
    model = argParser.getValueForOpt(modelOpt);
  }
#ifdef SYCL_EP
  std::string device = "default";
  if (argParser.hasValueForOpt(deviceOpt)) {
    device = argParser.getValueForOpt(deviceOpt, /*warn*/ false);
  }

  std::string vendor = "";
  if (argParser.hasValueForOpt(vendorOpt)) {
    vendor = argParser.getValueForOpt(vendorOpt, /*warn*/ false);
  }
#endif

  std::string logLevelStr = "2";
  if (argParser.hasValueForOpt(logLevelOpt)) {
    logLevelStr = argParser.getValueForOpt(logLevelOpt, /*warn*/ false);
  }

  std::string graphOptLevelStr = "disable_all";
  if (argParser.hasValueForOpt(graphOptLevelOpt)) {
    graphOptLevelStr =
        argParser.getValueForOpt(graphOptLevelOpt, /*warn*/ false);
  }

  const std::string image = imagePaths[0];
  std::vector<std::vector<float>> inputData = {
      utils::readImage(image, {1, 3, 224, 224})};

  // Set up logging
  const OrtLoggingLevel logLevel = utils::stringToLogLevel(logLevelStr);
  Ort::Env env(logLevel, model.c_str());

#ifdef SYCL_EP
  // Set up SYCL execution provider options
  // By default its device type is empty "" which points to default selector.
  OrtSYCLProviderOptions syclOptions;
  // Device selector : [""]->Default, ["cpu","CPU",..]->CPU, ["gpu", "GPU",
  // ..]->GPU selectors
  syclOptions.device_selector =
      (utils::str_tolower(device) == "default" ? "" : device.c_str());
  // Device vendor : Upper or Lower case vendor name (e.g. intel, arm, Nvidia
  // etc..). Empty "" by default.
  syclOptions.device_vendor = vendor.c_str();
#endif
#ifdef CUDA_EP
  // Custom options to be specified for CUDA EP
  OrtCUDAProviderOptions cudaOptions{};
  cudaOptions.device_id = 0;
  cudaOptions.gpu_mem_limit = std::numeric_limits<size_t>::max();
  cudaOptions.arena_extend_strategy = 0;
  cudaOptions.do_copy_in_default_stream = true;
  cudaOptions.has_user_compute_stream = 0;
  cudaOptions.user_compute_stream = nullptr;
  cudaOptions.default_memory_arena_cfg = nullptr;
#endif

  // Set up session options
  Ort::SessionOptions sessionOptions;
#ifdef SYCL_EP
  sessionOptions.AppendExecutionProvider_SYCL(syclOptions);
#endif 
#ifdef CUDA_EP
  sessionOptions.AppendExecutionProvider_CUDA(cudaOptions);  
#endif
  sessionOptions.SetGraphOptimizationLevel(
      utils::strToGraphOptLevel(graphOptLevelStr));

  // Create inference session
  Ort::Session session(env, model.c_str(), sessionOptions);

  // Create allocator & memory info
  Ort::AllocatorWithDefaultOptions allocator;
  const Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(
      OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeCPU);

  // Query session for number of inputs and outputs of the model
  const size_t inputCount = session.GetInputCount();
  const size_t outputCount = session.GetOutputCount();

  // Create input `Ort::Value`s
  std::vector<char*> inputNames;
  std::vector<std::vector<int64_t>> inputShapes;
  std::vector<Ort::Value> inputTensors;
  inputNames.reserve(inputCount);
  inputShapes.reserve(inputCount);
  inputTensors.reserve(inputCount);
  for (size_t i = 0; i < inputCount; ++i) {
    // Query session for input names
    inputNames.emplace_back(session.GetInputName(i, allocator));
    // Query session for input shapes
    inputShapes.emplace_back(
        session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
    // Query session for input type
    const ONNXTensorElementDataType inputType = session.GetInputTypeInfo(i)
                                                    .GetTensorTypeAndShapeInfo()
                                                    .GetElementType();
    // Calculate input data size
    const size_t inputDataSize = utils::product(inputShapes[i]);
    // Get rank of input
    const size_t rank = inputShapes[i].size();
    // Create input tensor
    inputTensors.emplace_back(Ort::Value::CreateTensor<float>(
        memInfo, inputData[i].data(), inputDataSize, inputShapes[i].data(),
        rank));
  }

  // Create output `Ort::Value`s
  std::vector<char*> outputNames;
  std::vector<std::vector<int64_t>> outputShapes;
  std::vector<Ort::Value> outputTensors;
  std::vector<std::vector<float>> outputData;
  outputNames.reserve(outputCount);
  outputShapes.reserve(outputCount);
  outputTensors.reserve(outputCount);
  outputData.reserve(outputCount);
  for (size_t i = 0; i < outputCount; ++i) {
    // Query session for output names
    outputNames.emplace_back(session.GetOutputName(i, allocator));
    // Query session for output shapes
    outputShapes.emplace_back(
        session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
    // Query session for output type
    const ONNXTensorElementDataType outputType =
        session.GetOutputTypeInfo(i)
            .GetTensorTypeAndShapeInfo()
            .GetElementType();
    // Calculate output data size
    const size_t outputDataSize = utils::product(outputShapes[i]);
    // Allocate enough memory for output
    outputData.emplace_back(outputDataSize);
    // Get rank of output
    const size_t rank = outputShapes[i].size();
    // Create output tensor
    outputTensors.emplace_back(Ort::Value::CreateTensor<float>(
        memInfo, outputData[i].data(), outputDataSize, outputShapes[i].data(),
        rank));
  }

  // Run inference
  session.Run(Ort::RunOptions{nullptr}, inputNames.data(), inputTensors.data(),
              inputCount, outputNames.data(), outputTensors.data(),
              outputCount);

  // Print output class
  const size_t idx = utils::argmax(outputData[0]);
  std::cout << "Classed as " << idx << " (" << utils::indexToLabel(idx)
            << ")\n";

  int perfPasses = 8;
  long long int avgTime = 0;
  for(int i = 0; i < perfPasses; i++){
    std::cout << "Execution Pass: " << i << "\n";
    auto st = std::chrono::high_resolution_clock::now();
    session.Run(Ort::RunOptions{nullptr}, inputNames.data(), inputTensors.data(),
              inputCount, outputNames.data(), outputTensors.data(),
              outputCount);
  
    auto end = std::chrono::high_resolution_clock::now();  
    auto currTime = (end - st).count();
    avgTime+=currTime;
    std::cout << "Time elapsed : " << currTime << " ns\n";
  }

  std::cout<<"\nAverage Elapsed time = "<< avgTime/perfPasses<<std::endl;

  // Free memory
  for (size_t i = 0; i < inputNames.size(); ++i) {
    allocator.Free(inputNames[i]);
  }
  for (size_t i = 0; i < outputNames.size(); ++i) {
    allocator.Free(outputNames[i]);
  }
  inputNames.clear();
  outputNames.clear();

  return 0;
}
