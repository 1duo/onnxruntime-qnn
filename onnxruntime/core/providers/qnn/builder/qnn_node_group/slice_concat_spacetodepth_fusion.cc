// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "core/providers/qnn/builder/qnn_node_group/slice_concat_spacetodepth_fusion.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/providers/qnn/builder/qnn_def.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_node_group/utils.h"
#include "core/providers/qnn/builder/qnn_utils.h"

namespace onnxruntime {
namespace qnn {
namespace {

constexpr size_t kNchwRank = 4;
constexpr int64_t kHeightAxis = 2;
constexpr int64_t kWidthAxis = 3;
constexpr int64_t kChannelAxis = 1;
constexpr int64_t kSliceStepTwo = 2;

using NodeToUnitMap = std::unordered_map<const OrtNode*, const OrtNodeUnit*>;
using UnitToGroupMap = std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>;

[[nodiscard]] bool IsSliceUnit(const OrtNodeUnit* node_unit) {
  return node_unit != nullptr && node_unit->OpType() == "Slice";
}

[[nodiscard]] bool IsConcatUnit(const OrtNodeUnit* node_unit) {
  return node_unit != nullptr && node_unit->OpType() == "Concat";
}

// Slice indices use int64 or int32 depending on exporter; accept both.
[[nodiscard]] std::optional<std::vector<int64_t>> ReadSliceIndexInitializer(
    const QnnModelWrapper& model_wrapper, const OrtNodeUnit& slice_unit, size_t input_index) {
  const auto& slice_inputs = slice_unit.Inputs();
  if (input_index >= slice_inputs.size()) {
    return std::nullopt;
  }
  const std::string& initializer_name = slice_inputs[input_index].name;
  if (initializer_name.empty() || !model_wrapper.IsConstantInput(initializer_name)) {
    return std::nullopt;
  }
  const OrtValueInfo* constant = model_wrapper.GetConstantTensor(initializer_name);
  if (constant == nullptr) {
    return std::nullopt;
  }
  Ort::ConstValueInfo constant_info(constant);
  Ort::ConstValue constant_value;
  if (!constant_info.GetInitializer(constant_value).IsOK()) {
    return std::nullopt;
  }
  const auto tensor_info = constant_info.TypeInfo().GetTensorTypeAndShapeInfo();
  const size_t element_count = tensor_info.GetElementCount();
  if (element_count == 0 || element_count > 4) {
    return std::nullopt;
  }
  const auto element_type = tensor_info.GetElementType();
  if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    const int64_t* data = constant_value.GetTensorData<int64_t>();
    if (data == nullptr) {
      return std::nullopt;
    }
    return std::vector<int64_t>(data, data + element_count);
  }
  if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    const int32_t* data = constant_value.GetTensorData<int32_t>();
    if (data == nullptr) {
      return std::nullopt;
    }
    return std::vector<int64_t>(data, data + element_count);
  }
  return std::nullopt;
}

struct StepTwoSlice {
  int64_t axis = -1;
  int64_t start = -1;
};

[[nodiscard]] std::optional<StepTwoSlice> ParseStepTwoSlice(
    const QnnModelWrapper& model_wrapper, const OrtNodeUnit& slice_unit,
    const std::vector<uint32_t>& slice_input_shape) {
  if (!IsSliceUnit(&slice_unit) || slice_input_shape.size() != kNchwRank) {
    return std::nullopt;
  }
  // Slice-10+: data, starts, ends, axes, steps. Axes required; steps defaults to 1.
  if (slice_unit.Inputs().size() <= 3) {
    return std::nullopt;
  }
  const auto starts = ReadSliceIndexInitializer(model_wrapper, slice_unit, 1);
  const auto ends = ReadSliceIndexInitializer(model_wrapper, slice_unit, 2);
  const auto axes = ReadSliceIndexInitializer(model_wrapper, slice_unit, 3);
  const auto steps = slice_unit.Inputs().size() > 4
                         ? ReadSliceIndexInitializer(model_wrapper, slice_unit, 4)
                         : std::optional<std::vector<int64_t>>{std::vector<int64_t>{1}};
  if (!starts.has_value() || !ends.has_value() || !axes.has_value() ||
      starts->size() != 1 || ends->size() != 1 || axes->size() != 1) {
    return std::nullopt;
  }
  int64_t axis = (*axes)[0] < 0 ? (*axes)[0] + 4 : (*axes)[0];
  if (axis != kHeightAxis && axis != kWidthAxis) {
    return std::nullopt;
  }
  const int64_t step = (steps.has_value() && !steps->empty()) ? (*steps)[0] : 1;
  if (step != kSliceStepTwo) {
    return std::nullopt;
  }
  int64_t start = (*starts)[0] < 0
                      ? (*starts)[0] + static_cast<int64_t>(slice_input_shape[static_cast<size_t>(axis)])
                      : (*starts)[0];
  if (start != 0 && start != 1) {
    return std::nullopt;
  }
  return StepTwoSlice{axis, start};
}

[[nodiscard]] bool HasExpectedShape(const std::vector<uint32_t>& actual,
                                    uint32_t batch, uint32_t channels, uint32_t height, uint32_t width) {
  return actual.size() == kNchwRank && actual[0] == batch && actual[1] == channels &&
         actual[2] == height && actual[3] == width;
}

// QDQGroups merge DQ/Q into the Slice/Concat unit, so the Concat parent is the Slice
// group directly. A lone DQ/Q singleton between them carries an unchecked scale.
[[nodiscard]] const OrtNodeUnit* GetParentSliceOrNull(
    const QnnModelWrapper& model_wrapper, const OrtNodeUnit& child_unit,
    const OrtNodeUnitIODef& child_input, const NodeToUnitMap& node_to_unit,
    const UnitToGroupMap& unit_to_group) {
  const OrtNodeUnit* parent =
      GetParentOfInput(model_wrapper, child_unit, child_input, node_to_unit, unit_to_group);
  return IsSliceUnit(parent) ? parent : nullptr;
}

struct PerTensorQuant {
  bool quantized = false;
  float scale = 1.0f;
  int32_t offset = 0;
  uint32_t qnn_dtype = 0;
};

[[nodiscard]] std::optional<PerTensorQuant> GetBoundaryQuant(
    const QnnModelWrapper& model_wrapper, const OrtNodeUnitIODef& tensor_def) {
  TensorInfo tensor_info = {};
  if (!model_wrapper.GetTensorInfo(tensor_def, tensor_info).IsOK()) {
    return std::nullopt;
  }
  PerTensorQuant quant;
  quant.qnn_dtype = tensor_info.qnn_data_type;
  quant.quantized = tensor_info.quant_param.IsQuantized();
  if (quant.quantized) {
    if (!tensor_info.quant_param.IsPerTensor(/*include_bw*/ true)) {
      return std::nullopt;
    }
    if (!tensor_info.quant_param.GetPerTensorScaleOffset(quant.scale, quant.offset).IsOK()) {
      return std::nullopt;
    }
  }
  return quant;
}

[[nodiscard]] bool HasSameQuant(const PerTensorQuant& lhs, const PerTensorQuant& rhs) {
  if (lhs.quantized != rhs.quantized || lhs.qnn_dtype != rhs.qnn_dtype) {
    return false;
  }
  return !lhs.quantized || (lhs.scale == rhs.scale && lhs.offset == rhs.offset);
}

Ort::Status AddLayoutTranspose(QnnModelWrapper& model_wrapper, const OrtNodeUnit& concat_unit,
                               const std::string& transpose_name, const std::string& input_name,
                               const std::string& output_name, std::vector<uint32_t> perm) {
  QnnParamWrapper perm_param(concat_unit.Index(), transpose_name, QNN_OP_TRANSPOSE_PARAM_PERM,
                             {static_cast<uint32_t>(perm.size())}, std::move(perm));
  const std::string perm_param_name = perm_param.GetParamTensorName();
  RETURN_IF_NOT(model_wrapper.AddParamWrapper(std::move(perm_param)), "Failed to add transpose perm.");
  RETURN_IF_NOT(model_wrapper.CreateQnnNode(transpose_name, QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_TRANSPOSE,
                                            {input_name}, {output_name}, {perm_param_name},
                                            /*validate*/ false),
                "Failed to add layout transpose.");
  return Ort::Status();
}

Ort::Status AddCrdSpaceToDepth(QnnModelWrapper& model_wrapper, const OrtNodeUnit& concat_unit,
                               const std::string& s2d_name, const std::string& input_name,
                               const std::string& output_name) {
  std::vector<uint32_t> block_shape{2};
  std::vector<uint32_t> block_data{SliceConcatSpaceToDepthFusion::kBlockHeight,
                                   SliceConcatSpaceToDepthFusion::kBlockWidth};
  QnnParamWrapper block_param(concat_unit.Index(), s2d_name, QNN_OP_SPACE_TO_DEPTH_PARAM_BLOCK_SIZE,
                              std::move(block_shape), std::move(block_data));
  Qnn_Scalar_t mode_scalar = QNN_SCALAR_INIT;
  mode_scalar.dataType = QNN_DATATYPE_UINT_32;
  mode_scalar.uint32Value = QNN_OP_SPACE_TO_DEPTH_MODE_CRD;
  QnnParamWrapper mode_param(concat_unit.Index(), s2d_name, QNN_OP_SPACE_TO_DEPTH_PARAM_MODE, mode_scalar);
  const std::string block_name = block_param.GetParamTensorName();
  const std::string mode_name = mode_param.GetParamTensorName();
  RETURN_IF_NOT(model_wrapper.AddParamWrapper(std::move(block_param)), "Failed to add S2D block param.");
  RETURN_IF_NOT(model_wrapper.AddParamWrapper(std::move(mode_param)), "Failed to add S2D mode param.");
  RETURN_IF_NOT(model_wrapper.CreateQnnNode(s2d_name, QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_SPACE_TO_DEPTH,
                                            {input_name}, {output_name}, {block_name, mode_name},
                                            /*validate*/ false),
                "Failed to add SpaceToDepth.");
  return Ort::Status();
}

}  // namespace

gsl::span<const OrtNodeUnit* const> SliceConcatSpaceToDepthFusion::GetNodeUnits() const {
  return gsl::span<const OrtNodeUnit* const>{node_units_.data(), node_units_.size()};
}

std::unique_ptr<IQnnNodeGroup> SliceConcatSpaceToDepthFusion::TryFusion(
    QnnModelWrapper& model_wrapper, const OrtNodeUnit& concat_unit, const NodeToUnitMap& node_to_unit,
    const UnitToGroupMap& unit_to_group, const Ort::Logger& logger) {
  ORT_UNUSED_PARAMETER(logger);
  if (!IsConcatUnit(&concat_unit) || concat_unit.Inputs().size() != 4) {
    return nullptr;
  }
  if (OrtNodeAttrHelper(concat_unit).Get("axis", static_cast<int64_t>(-1)) != kChannelAxis) {
    return nullptr;
  }
  if (model_wrapper.GetQnnBackendType() != QnnBackendType::HTP &&
      model_wrapper.GetQnnBackendType() != QnnBackendType::HTP_FP16) {
    return nullptr;
  }

  std::array<const OrtNodeUnit*, 4> width_slices{};
  for (size_t i = 0; i < width_slices.size(); ++i) {
    width_slices[i] =
        GetParentSliceOrNull(model_wrapper, concat_unit, concat_unit.Inputs()[i], node_to_unit, unit_to_group);
    if (width_slices[i] == nullptr) {
      return nullptr;
    }
  }

  std::array<const OrtNodeUnit*, 4> height_slice_for_width{};
  for (size_t i = 0; i < height_slice_for_width.size(); ++i) {
    height_slice_for_width[i] = GetParentSliceOrNull(model_wrapper, *width_slices[i],
                                                     width_slices[i]->Inputs()[0], node_to_unit, unit_to_group);
    if (height_slice_for_width[i] == nullptr) {
      return nullptr;
    }
  }
  if (!(height_slice_for_width[0] == height_slice_for_width[2] &&
        height_slice_for_width[1] == height_slice_for_width[3] &&
        height_slice_for_width[0] != height_slice_for_width[1])) {
    return nullptr;
  }

  const OrtNodeUnit* height_slice_a = height_slice_for_width[0];
  const OrtNodeUnit* height_slice_b = height_slice_for_width[1];
  if (height_slice_a->Inputs().empty() || height_slice_b->Inputs().empty() ||
      height_slice_a->Inputs()[0].name != height_slice_b->Inputs()[0].name) {
    return nullptr;
  }

  std::vector<uint32_t> input_shape, concat_shape;
  std::array<std::vector<uint32_t>, 2> height_shapes{};
  std::array<std::vector<uint32_t>, 4> width_shapes{};
  if (!QnnModelWrapper::GetOnnxShape(height_slice_a->Inputs()[0].shape, input_shape) ||
      !QnnModelWrapper::GetOnnxShape(concat_unit.Outputs()[0].shape, concat_shape) ||
      !QnnModelWrapper::GetOnnxShape(height_slice_a->Outputs()[0].shape, height_shapes[0]) ||
      !QnnModelWrapper::GetOnnxShape(height_slice_b->Outputs()[0].shape, height_shapes[1])) {
    return nullptr;
  }
  for (size_t i = 0; i < width_shapes.size(); ++i) {
    if (!QnnModelWrapper::GetOnnxShape(width_slices[i]->Outputs()[0].shape, width_shapes[i])) {
      return nullptr;
    }
  }
  if (input_shape.size() != kNchwRank || input_shape[0] == 0 || input_shape[1] == 0 ||
      input_shape[2] % 2 != 0 || input_shape[3] % 2 != 0 || input_shape[2] == 0 || input_shape[3] == 0) {
    return nullptr;
  }
  const uint32_t batch = input_shape[0];
  const uint32_t channels = input_shape[1];
  const uint32_t height = input_shape[2];
  const uint32_t width = input_shape[3];
  for (const auto& height_shape : height_shapes) {
    if (!HasExpectedShape(height_shape, batch, channels, height / 2, width)) {
      return nullptr;
    }
  }
  for (const auto& width_shape : width_shapes) {
    if (!HasExpectedShape(width_shape, batch, channels, height / 2, width / 2)) {
      return nullptr;
    }
  }
  if (!HasExpectedShape(concat_shape, batch, 4 * channels, height / 2, width / 2)) {
    return nullptr;
  }

  const auto height_spec_a = ParseStepTwoSlice(model_wrapper, *height_slice_a, input_shape);
  const auto height_spec_b = ParseStepTwoSlice(model_wrapper, *height_slice_b, input_shape);
  if (!height_spec_a.has_value() || !height_spec_b.has_value() ||
      height_spec_a->axis != kHeightAxis || height_spec_b->axis != kHeightAxis ||
      height_spec_a->start == height_spec_b->start) {
    return nullptr;
  }
  if (!((height_spec_a->start == 0 && height_spec_b->start == 1) ||
        (height_spec_a->start == 1 && height_spec_b->start == 0))) {
    return nullptr;
  }
  const OrtNodeUnit* height_start_zero = (height_spec_a->start == 0) ? height_slice_a : height_slice_b;
  const OrtNodeUnit* height_start_one = (height_spec_a->start == 0) ? height_slice_b : height_slice_a;
  if (!(height_slice_for_width[0] == height_start_zero && height_slice_for_width[1] == height_start_one &&
        height_slice_for_width[2] == height_start_zero && height_slice_for_width[3] == height_start_one)) {
    return nullptr;
  }
  for (size_t i = 0; i < width_slices.size(); ++i) {
    std::vector<uint32_t> parent_height_shape;
    if (!QnnModelWrapper::GetOnnxShape(height_slice_for_width[i]->Outputs()[0].shape, parent_height_shape)) {
      return nullptr;
    }
    const auto width_spec = ParseStepTwoSlice(model_wrapper, *width_slices[i], parent_height_shape);
    if (!width_spec.has_value() || width_spec->axis != kWidthAxis) {
      return nullptr;
    }
    if (width_spec->start != ((i == 0 || i == 1) ? 0 : 1)) {
      return nullptr;
    }
  }

  // Every boundary tensor (stem in, both H outs, all 4 W outs, concat out) shares one
  // per-tensor quant. Slices only rearrange, so any intermediate requant would be
  // skipped by S2D and change numerics.
  const OrtNodeUnitIODef* boundary_defs[] = {
      &height_start_zero->Inputs()[0],
      &height_start_zero->Outputs()[0],
      &height_start_one->Outputs()[0],
      &width_slices[0]->Outputs()[0],
      &width_slices[1]->Outputs()[0],
      &width_slices[2]->Outputs()[0],
      &width_slices[3]->Outputs()[0],
      &concat_unit.Outputs()[0],
  };
  std::optional<PerTensorQuant> reference_quant;
  for (const OrtNodeUnitIODef* boundary_def : boundary_defs) {
    const auto boundary_quant = GetBoundaryQuant(model_wrapper, *boundary_def);
    if (!boundary_quant.has_value()) {
      return nullptr;
    }
    if (!reference_quant.has_value()) {
      reference_quant = boundary_quant;
    } else if (!HasSameQuant(*reference_quant, *boundary_quant)) {
      return nullptr;
    }
  }

  const std::array<const OrtNodeUnit*, kGroupSize> focus_group = {
      height_start_zero, height_start_one, width_slices[0], width_slices[1],
      width_slices[2], width_slices[3], &concat_unit};
  return std::make_unique<SliceConcatSpaceToDepthFusion>(
      gsl::make_span<const OrtNodeUnit* const>(focus_group.data(), focus_group.size()));
}

Ort::Status SliceConcatSpaceToDepthFusion::IsSupported(QnnModelWrapper& model_wrapper,
                                                       const Ort::Logger& logger) const {
  ORT_UNUSED_PARAMETER(logger);
  if (model_wrapper.GetQnnBackendType() != QnnBackendType::HTP &&
      model_wrapper.GetQnnBackendType() != QnnBackendType::HTP_FP16) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "SliceConcatS2D: HTP only.");
  }
  if (node_units_.size() != kGroupSize || !IsConcatUnit(concat_node_unit_)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "SliceConcatS2D: expected 7 units with Concat target.");
  }
  std::vector<uint32_t> input_shape, output_shape;
  if (!QnnModelWrapper::GetOnnxShape(node_units_[0]->Inputs()[0].shape, input_shape) ||
      !QnnModelWrapper::GetOnnxShape(concat_node_unit_->Outputs()[0].shape, output_shape)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "SliceConcatS2D: unresolved shapes.");
  }
  if (input_shape.size() != kNchwRank || output_shape.size() != kNchwRank ||
      std::any_of(input_shape.begin(), input_shape.end(), [](uint32_t dim) { return dim == 0; }) ||
      std::any_of(output_shape.begin(), output_shape.end(), [](uint32_t dim) { return dim == 0; })) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "SliceConcatS2D: rank-4 non-zero shapes only.");
  }
  return Ort::Status();
}

Ort::Status SliceConcatSpaceToDepthFusion::AddToModelBuilder(QnnModelWrapper& model_wrapper,
                                                             const Ort::Logger& logger) const {
  ORT_UNUSED_PARAMETER(logger);
  const OrtNodeUnitIODef& focus_input_def = node_units_[0]->Inputs()[0];
  const OrtNodeUnitIODef& focus_output_def = concat_node_unit_->Outputs()[0];

  QnnTensorWrapper focus_input_tensor, focus_output_tensor;
  RETURN_IF_ERROR(model_wrapper.MakeTensorWrapper(focus_input_def, focus_input_tensor));
  RETURN_IF_ERROR(model_wrapper.MakeTensorWrapper(focus_output_def, focus_output_tensor));

  TensorInfo focus_input_info = {}, focus_output_info = {};
  RETURN_IF_ERROR(model_wrapper.GetTensorInfo(focus_input_def, focus_input_info));
  RETURN_IF_ERROR(model_wrapper.GetTensorInfo(focus_output_def, focus_output_info));

  std::vector<uint32_t> input_shape, output_shape;
  RETURN_IF_NOT(QnnModelWrapper::GetOnnxShape(focus_input_def.shape, input_shape),
                "SliceConcatS2D: bad input shape.");
  RETURN_IF_NOT(QnnModelWrapper::GetOnnxShape(focus_output_def.shape, output_shape),
                "SliceConcatS2D: bad output shape.");

  const std::string base_name = utils::UniqueNameGenerator().New(*concat_node_unit_, "_focus_s2d");
  const std::string nhwc_input_name = base_name + "_in";
  const std::string nhwc_output_name = base_name + "_out";

  QnnTensorWrapper nhwc_input_tensor(
      nhwc_input_name, QNN_TENSOR_TYPE_NATIVE, focus_input_info.qnn_data_type,
      focus_input_info.quant_param.Copy(),
      std::vector<uint32_t>{input_shape[0], input_shape[2], input_shape[3], input_shape[1]});
  QnnTensorWrapper nhwc_output_tensor(
      nhwc_output_name, QNN_TENSOR_TYPE_NATIVE, focus_output_info.qnn_data_type,
      focus_output_info.quant_param.Copy(),
      std::vector<uint32_t>{output_shape[0], output_shape[2], output_shape[3], output_shape[1]});

  auto add_tensor_once = [&](QnnTensorWrapper&& tensor, const std::string& tensor_name,
                             const char* error_message) -> Ort::Status {
    if (model_wrapper.IsQnnTensorWrapperExist(tensor_name)) {
      return Ort::Status();
    }
    RETURN_IF_NOT(model_wrapper.AddTensorWrapper(std::move(tensor)), error_message);
    return Ort::Status();
  };
  RETURN_IF_ERROR(add_tensor_once(std::move(focus_input_tensor), focus_input_def.name, "Bad focus input."));
  RETURN_IF_ERROR(add_tensor_once(std::move(focus_output_tensor), focus_output_def.name, "Bad focus output."));
  RETURN_IF_NOT(model_wrapper.AddTensorWrapper(std::move(nhwc_input_tensor)), "Bad NHWC input.");
  RETURN_IF_NOT(model_wrapper.AddTensorWrapper(std::move(nhwc_output_tensor)), "Bad NHWC output.");

  RETURN_IF_ERROR(AddLayoutTranspose(model_wrapper, *concat_node_unit_, base_name + "_pre",
                                     focus_input_def.name, nhwc_input_name, {0, 2, 3, 1}));
  RETURN_IF_ERROR(AddCrdSpaceToDepth(model_wrapper, *concat_node_unit_, base_name + "_s2d",
                                     nhwc_input_name, nhwc_output_name));
  return AddLayoutTranspose(model_wrapper, *concat_node_unit_, base_name + "_post", nhwc_output_name,
                            focus_output_def.name, {0, 3, 1, 2});
}

}  // namespace qnn
}  // namespace onnxruntime
