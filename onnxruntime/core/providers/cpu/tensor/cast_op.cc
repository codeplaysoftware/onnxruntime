// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstddef>
#include <cstdio>
#include <string>

#include "boost/mp11.hpp"

#include "gsl/gsl"

#include "core/common/common.h"
#include "core/common/type_list.h"
#include "core/framework/data_types.h"
#include "core/framework/data_types_internal.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/providers/op_kernel_type_control.h"
#include "core/providers/op_kernel_type_control_utils.h"
#include "core/util/math_cpuonly.h"

#include "Eigen/src/Core/arch/Default/BFloat16.h"
#include "Eigen/src/Core/arch/Default/Half.h"

#if defined(_M_AMD64)
#include "core/mlas/inc/mlas.h"
#endif

namespace onnxruntime {

namespace op_kernel_type_control {
// we're using one set of types for all opsets of Cast
ORT_SPECIFY_OP_KERNEL_ARG_SUPPORTED_TYPES_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, Cast, Input, 0,
    ORT_OP_KERNEL_TYPE_CTRL_ALL_TENSOR_DATA_TYPES);

ORT_SPECIFY_OP_KERNEL_ARG_SUPPORTED_TYPES_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, Cast, Output, 0,
    ORT_OP_KERNEL_TYPE_CTRL_ALL_TENSOR_DATA_TYPES);
}  // namespace op_kernel_type_control

namespace {
using EnabledSrcTypes = ORT_OP_KERNEL_ARG_ENABLED_TYPE_LIST_ALL_OPSETS(kCpuExecutionProvider, kOnnxDomain,
                                                                       Cast, Input, 0);
using EnabledDstTypes = ORT_OP_KERNEL_ARG_ENABLED_TYPE_LIST_ALL_OPSETS(kCpuExecutionProvider, kOnnxDomain,
                                                                       Cast, Output, 0);

// string cast helpers
// Note: when C++17 is available, use <charconv> functions

// handle floating point output separately
template <typename SrcType>
typename std::enable_if<std::is_floating_point<SrcType>::value, void>::type
CastToString(const SrcType& input, std::string& output) {
  static_assert(sizeof(SrcType) <= sizeof(double),
                "largest supported floating point type is double");
  if (std::isnan(input)) {
    output = "NaN";
  } else if (std::isinf(input)) {
    if (input < std::numeric_limits<SrcType>::lowest()) {
      output = "-INF";
    } else {
      output = "INF";
    }
  } else {
    // set precision to 8 to match numpy default behavior
    constexpr const char* format = "%.8g";
    const double value = static_cast<double>(input);

    char static_buffer[256];
    std::unique_ptr<char[]> dynamic_buffer{};

    gsl::span<char> buffer_span = gsl::make_span(static_buffer);

    auto snprintf_result = std::snprintf(buffer_span.data(), buffer_span.size(), format, value);
    ORT_ENFORCE(snprintf_result > 0, "snprintf() failed with return value: ", snprintf_result);

    // include trailing '\0'
    const size_t required_buffer_size = gsl::narrow_cast<size_t>(snprintf_result) + 1;

    if (required_buffer_size > buffer_span.size()) {
      // didn't get it all, allocate a bigger buffer and retry
      dynamic_buffer = onnxruntime::make_unique<char[]>(required_buffer_size);
      buffer_span = gsl::make_span(dynamic_buffer.get(), required_buffer_size);
      snprintf_result = std::snprintf(buffer_span.data(), buffer_span.size(), format, value);
      ORT_ENFORCE(
          snprintf_result > 0 &&
              gsl::narrow_cast<size_t>(snprintf_result) == buffer_span.size() - 1,
          "Failed to write value with snprintf().");
    }

    output.assign(buffer_span.data(), required_buffer_size - 1);
  }
}

template <typename SrcType>
typename std::enable_if<!std::is_floating_point<SrcType>::value, void>::type
CastToString(const SrcType& input, std::string& output) {
  output = std::to_string(input);
}

// overloads for MLFloat16 and BFloat16
void CastToString(const MLFloat16& input, std::string& output) {
  CastToString(static_cast<float>(input), output);
}

void CastToString(const BFloat16& input, std::string& output) {
  CastToString(static_cast<float>(input), output);
}

template <typename DstType>
typename std::enable_if<std::is_floating_point<DstType>::value, void>::type
CastFromString(const std::string& input, DstType& output) {
  static_assert(sizeof(DstType) <= sizeof(double),
                "largest supported floating point type is double");
  output = gsl::narrow_cast<DstType>(std::stod(input));
}

template <typename DstType>
typename std::enable_if<std::is_integral<DstType>::value && std::is_unsigned<DstType>::value, void>::type
CastFromString(const std::string& input, DstType& output) {
  static_assert(sizeof(DstType) <= sizeof(unsigned long long),
                "largest supported unsigned integral type is unsigned long long");
  output = gsl::narrow_cast<DstType>(std::stoull(input));
}

template <typename DstType>
typename std::enable_if<std::is_integral<DstType>::value && std::is_signed<DstType>::value, void>::type
CastFromString(const std::string& input, DstType& output) {
  static_assert(sizeof(DstType) <= sizeof(long long),
                "largest supported signed integral type is long long");
  output = gsl::narrow_cast<DstType>(std::stoll(input));
}

// overloads for MLFloat16 and BFloat16
void CastFromString(const std::string& input, MLFloat16& output) {
  float intermediate;
  CastFromString(input, intermediate);
  output = static_cast<MLFloat16>(intermediate);
}

void CastFromString(const std::string& input, BFloat16& output) {
  float intermediate;
  CastFromString(input, intermediate);
  output = static_cast<BFloat16>(intermediate);
}

// type that is usable with Eigen cast
template <typename T>
struct EigenCastType {
  using type = T;
};

// ORT float16 types don't support casting, so map them to Eigen ones

template <>
struct EigenCastType<MLFloat16> {
  using type = Eigen::half;
};

template <>
struct EigenCastType<BFloat16> {
  using type = Eigen::bfloat16;
};

// generic tensor X -> Y
template <typename SrcType, typename DstType, typename Enable = void>
struct TensorCaster {
  void Cast(const OpKernelContext&, const Tensor& in, Tensor& out, const TensorShape& shape) const {
    using SrcEigenCastType = typename EigenCastType<SrcType>::type;
    using DstEigenCastType = typename EigenCastType<DstType>::type;

    const std::ptrdiff_t shape_size = gsl::narrow<std::ptrdiff_t>(shape.Size());
    const auto in_vector =
        ConstEigenVectorMap<SrcEigenCastType>(reinterpret_cast<const SrcEigenCastType*>(in.Data<SrcType>()), shape_size);
    auto out_vector =
        EigenVectorMap<DstEigenCastType>(reinterpret_cast<DstEigenCastType*>(out.MutableData<DstType>()), shape_size);
    out_vector = in_vector.template cast<DstEigenCastType>();
  }
};

// tensor X -> string
template <typename SrcType>
struct TensorCaster<SrcType, std::string> {
  void Cast(const OpKernelContext&, const Tensor& in, Tensor& out, const TensorShape& shape) const {
    const std::ptrdiff_t shape_size = gsl::narrow<std::ptrdiff_t>(shape.Size());
    const auto* in_data = in.Data<SrcType>();
    auto* out_data = out.MutableData<std::string>();
    for (std::ptrdiff_t i = 0; i < shape_size; ++i) {
      CastToString(in_data[i], out_data[i]);
    }
  }
};

// tensor string -> X
template <typename DstType>
struct TensorCaster<std::string, DstType> {
  void Cast(const OpKernelContext&, const Tensor& in, Tensor& out, const TensorShape& shape) const {
    const std::ptrdiff_t shape_size = gsl::narrow<std::ptrdiff_t>(shape.Size());
    const auto* in_data = in.Data<std::string>();
    auto* out_data = out.MutableData<DstType>();
    for (std::ptrdiff_t i = 0; i < shape_size; ++i) {
      CastFromString(in_data[i], out_data[i]);
    }
  }
};

#if defined(_M_AMD64)
// specializations to use optimized and Windows x64-specific
// MlasConvertHalfToFloatBuffer() routine for MLFloat16 -> float conversion

template <typename DstType>
void CastMLFloat16ThroughFloat(
    const OpKernelContext& context, const Tensor& in, Tensor& out, const TensorShape& shape) {
  // use optimized MLFloat16 -> float, then float -> DstType
  AllocatorPtr allocator;
  ORT_THROW_IF_ERROR(context.GetTempSpaceAllocator(&allocator));
  auto intermediate_buffer = IAllocator::MakeUniquePtr<float>(allocator, gsl::narrow<size_t>(shape.Size()));
  Tensor intermediate_tensor{DataTypeImpl::GetType<float>(), shape, intermediate_buffer.get(), allocator->Info()};
  TensorCaster<MLFloat16, float>{}.Cast(context, in, intermediate_tensor, shape);
  TensorCaster<float, DstType>{}.Cast(context, intermediate_tensor, out, shape);
}

// tensor MLFloat16 -> X
template <typename DstType>
struct TensorCaster<MLFloat16, DstType> {
  void Cast(const OpKernelContext& context, const Tensor& in, Tensor& out, const TensorShape& shape) const {
    CastMLFloat16ThroughFloat<DstType>(context, in, out, shape);
  }
};

// tensor MLFloat16 -> float
template <>
struct TensorCaster<MLFloat16, float> {
  void Cast(const OpKernelContext&, const Tensor& in, Tensor& out, const TensorShape& shape) const {
    auto out_data = out.MutableData<float>();
    auto in_data = in.Data<MLFloat16>();
    const size_t shape_size = gsl::narrow<size_t>(shape.Size());
    MlasConvertHalfToFloatBuffer(&in_data[0].val, out_data, shape_size);
  }
};

// tensor MLFloat16 -> string
template <>
struct TensorCaster<MLFloat16, std::string> {
  void Cast(const OpKernelContext& context, const Tensor& in, Tensor& out, const TensorShape& shape) const {
    CastMLFloat16ThroughFloat<std::string>(context, in, out, shape);
  }
};
#endif

class Cast final : public OpKernel {
 public:
  Cast(const OpKernelInfo& info) : OpKernel(info) {
    int64_t to;
    Status status = info.GetAttr("to", &to);
    ORT_ENFORCE(status.IsOK(), "Attribute to is not set.");
    to_ = gsl::narrow_cast<ONNX_NAMESPACE::TensorProto_DataType>(to);
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  ONNX_NAMESPACE::TensorProto_DataType to_;
};

template <typename TSrc, typename TDst>
struct Dispatcher {
  void operator()(const OpKernelContext& context, const Tensor& src, Tensor& dst, const TensorShape& shape) {
    TensorCaster<TSrc, TDst>{}.Cast(context, src, dst, shape);
  }
};

template <typename TSrc>
struct SrcDispatcher {
  void operator()(
      int32_t to, const OpKernelContext& context, const Tensor& src, Tensor& dst, const TensorShape& shape) {
    using DstTypes = boost::mp11::mp_remove_if_q<EnabledDstTypes, boost::mp11::mp_bind_front<std::is_same, TSrc>>;
    utils::MLTypeCallDispatcherFromTypeList<DstTypes> dispatcher{to};
    dispatcher.template InvokeWithLeadingTemplateArgs<Dispatcher, TypeList<TSrc>>(context, src, dst, shape);
  }
};

Status Cast::Compute(OpKernelContext* context) const {
  const Tensor* X = context->Input<Tensor>(0);
  const TensorShape& shape = X->Shape();
  Tensor* Y = context->Output(0, shape);

  if (shape.Size() == 0) {
    return Status::OK();
  }

  const auto from = X->GetElementType();

  if (from == to_) {
    // will copy if X and Y have different buffers
    CopyCpuTensor(X, Y);
    return Status::OK();
  }

  utils::MLTypeCallDispatcherFromTypeList<EnabledSrcTypes> dispatcher{from};
  dispatcher.Invoke<SrcDispatcher>(to_, *context, *X, *Y, shape);

  return Status::OK();
}

const std::vector<MLDataType> src_type_constraints = BuildKernelDefConstraintsFunctorFromTypeList<EnabledSrcTypes>{}();
const std::vector<MLDataType> dst_type_constraints = BuildKernelDefConstraintsFunctorFromTypeList<EnabledDstTypes>{}();

}  // namespace

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Cast,
    6,
    12,
    KernelDefBuilder()
        .TypeConstraint("T1", src_type_constraints)
        .TypeConstraint("T2", dst_type_constraints)
        .MayInplace(0, 0),  // allocation planner will check input and output sizes match before inplacing
    Cast);

ONNX_CPU_OPERATOR_KERNEL(
    Cast,
    13,
    KernelDefBuilder()
        .TypeConstraint("T1", src_type_constraints)
        .TypeConstraint("T2", dst_type_constraints)
        .MayInplace(0, 0),  // allocation planner will check input and output sizes match before inplacing
    Cast);

}  // namespace onnxruntime
