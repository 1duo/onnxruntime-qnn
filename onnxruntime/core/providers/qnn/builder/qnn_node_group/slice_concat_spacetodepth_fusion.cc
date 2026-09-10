// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "core/providers/qnn/builder/qnn_node_group/slice_concat_spacetodepth_fusion.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
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
constexpr int64_t kChannelAxis = 1;
constexpr int64_t kHeightAxis = 2;
constexpr int64_t kWidthAxis = 3;
constexpr int64_t kSliceStepTwo = 2;

using NodeToUnitMap = std::unordered_map<const OrtNode*, const OrtNodeUnit*>;
using UnitToGroupMap = std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>;

struct SlicePhase {
  int64_t height_offset = -1;
  int64_t width_offset = -1;
};

// Canonical ONNX SpaceToDepth (DCR) block order.
constexpr std::array<SlicePhase, 4> kCanonicalDcrPhases = {
    SlicePhase{0, 0}, SlicePhase{0, 1}, SlicePhase{1, 0}, SlicePhase{1, 1}};

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

// Each fused intermediate must have no consumers outside the group; otherwise lowering
// to S2D would silently drop that edge's tensor. Consumers resolve through NodeUnits:
// in QDQ graphs the direct edge target is a DQ node owned by the consumer's group.
[[nodiscard]] bool HasExactlyUnitConsumers(const OrtNodeUnit& producer_unit,
                                           const std::vector<const OrtNodeUnit*>& expected_consumer_units,
                                           const NodeToUnitMap& node_to_unit) {
  const Ort::ConstNode producer_node(&producer_unit.GetNode());
  const std::vector<Ort::ConstValueInfo> producer_outputs = producer_node.GetOutputs();
  if (producer_outputs.size() != 1 || producer_outputs[0].IsGraphOutput()) {
    return false;
  }
  const std::vector<Ort::ValueInfoConsumerProducerInfo> consumers = producer_outputs[0].GetConsumers();
  if (consumers.size() != expected_consumer_units.size()) {
    return false;
  }
  for (const OrtNodeUnit* expected_unit : expected_consumer_units) {
    const bool found = std::any_of(consumers.begin(), consumers.end(),
                                   [&node_to_unit, expected_unit](
                                       const Ort::ValueInfoConsumerProducerInfo& consumer) {
                                     if (consumer.node == nullptr) {
                                       return false;
                                     }
                                     const auto it = node_to_unit.find(consumer.node);
                                     return it != node_to_unit.end() && it->second == expected_unit;
                                   });
    if (!found) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool TryGetDcrPhasePermutation(const std::array<SlicePhase, 4>& actual_phases,
                                             std::array<int64_t, 4>& permutation) {
  std::array<bool, 4> used{false, false, false, false};
  for (size_t i = 0; i < actual_phases.size(); ++i) {
    bool matched = false;
    for (size_t j = 0; j < kCanonicalDcrPhases.size(); ++j) {
      if (!used[j] && actual_phases[i].height_offset == kCanonicalDcrPhases[j].height_offset &&
          actual_phases[i].width_offset == kCanonicalDcrPhases[j].width_offset) {
        permutation[i] = static_cast<int64_t>(j);
        used[j] = true;
        matched = true;
        break;
      }
    }
    if (!matched) {
      return false;
    }
  }
  return true;
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

// Channel indices restoring Concat block order from canonical DCR output.
[[nodiscard]] std::vector<int32_t> BuildDcrGatherIndices(const std::array<int64_t, 4>& dcr_permutation,
                                                         uint32_t channel_count) {
  std::vector<int32_t> gather_indices;
  gather_indices.reserve(static_cast<size_t>(4 * channel_count));
  for (const int64_t source_block : dcr_permutation) {
    for (uint32_t channel = 0; channel < channel_count; ++channel) {
      gather_indices.push_back(static_cast<int32_t>(source_block * channel_count + channel));
    }
  }
  return gather_indices;
}

Ort::Status AddLayoutTranspose(QnnModelWrapper& model_wrapper, const OrtNodeUnit& concat_unit,
                               const std::string& transpose_name, const std::string& input_name,
                               const std::string& output_name, std::vector<uint32_t> perm, bool validate,
                               std::vector<Qnn_Tensor_t> validated_inputs,
                               std::vector<Qnn_Tensor_t> validated_outputs) {
  QnnParamWrapper perm_param(concat_unit.Index(), transpose_name, QNN_OP_TRANSPOSE_PARAM_PERM,
                             {static_cast<uint32_t>(perm.size())}, std::move(perm));
  if (validate) {
    std::vector<Qnn_Param_t> params{perm_param.GetQnnParam()};
    return model_wrapper.ValidateQnnNode(transpose_name, QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_TRANSPOSE,
                                         std::move(validated_inputs), std::move(validated_outputs),
                                         std::move(params));
  }
  const std::string perm_param_name = perm_param.GetParamTensorName();
  RETURN_IF_NOT(model_wrapper.AddParamWrapper(std::move(perm_param)), "Failed to add transpose perm.");
  RETURN_IF_NOT(model_wrapper.CreateQnnNode(transpose_name, QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_TRANSPOSE,
                                            {input_name}, {output_name}, {perm_param_name},
                                            /*validate*/ false),
                "Failed to add layout transpose.");
  return Ort::Status();
}

// Lowering shared by IsSupported (validate=true) and AddToModelBuilder (validate=false):
// PreT(NCHW->NHWC) + S2D(DCR) + [Gather channel reorder] + PostT(NHWC->NCHW).
Ort::Status CreateOrValidateFocusGraph(QnnModelWrapper& model_wrapper, const OrtNodeUnit& common_input_owner,
                                       const OrtNodeUnit& concat_unit,
                                       const std::array<int64_t, 4>& dcr_permutation, uint32_t channel_count,
                                       bool validate, const Ort::Logger& logger) {
  ORT_UNUSED_PARAMETER(logger);
  const OrtNodeUnitIODef& focus_input_def = common_input_owner.Inputs()[0];
  const OrtNodeUnitIODef& focus_output_def = concat_unit.Outputs()[0];

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

  const std::string base_name = utils::UniqueNameGenerator().New(concat_unit, "_focus_s2d");
  const std::string nhwc_input_name = base_name + "_in";
  const std::string nhwc_output_name = base_name + "_s2d_out";
  const std::string pre_name = base_name + "_pre";
  const std::string s2d_name = base_name + "_s2d";
  const std::string post_name = base_name + "_post";

  QnnTensorWrapper nhwc_input_tensor(
      nhwc_input_name, QNN_TENSOR_TYPE_NATIVE, focus_input_info.qnn_data_type,
      focus_input_info.quant_param.Copy(),
      std::vector<uint32_t>{input_shape[0], input_shape[2], input_shape[3], input_shape[1]});
  QnnTensorWrapper nhwc_output_tensor(
      nhwc_output_name, QNN_TENSOR_TYPE_NATIVE, focus_output_info.qnn_data_type,
      focus_output_info.quant_param.Copy(),
      std::vector<uint32_t>{output_shape[0], output_shape[2], output_shape[3], output_shape[1]});

  const bool needs_gather = dcr_permutation != std::array<int64_t, 4>{0, 1, 2, 3};
  const std::string gather_input_name = needs_gather ? base_name + "_gather_in" : nhwc_output_name;
  // Gather runs on NCHW channels after PostT (mirrors upstream S2D-then-Gather order).
  const std::string nchw_s2d_name = needs_gather ? gather_input_name : focus_output_def.name;

  QnnTensorWrapper nchw_s2d_tensor(
      nchw_s2d_name, QNN_TENSOR_TYPE_NATIVE, focus_output_info.qnn_data_type,
      focus_output_info.quant_param.Copy(),
      std::vector<uint32_t>(output_shape));

  std::vector<int32_t> gather_indices;
  if (needs_gather) {
    gather_indices = BuildDcrGatherIndices(dcr_permutation, channel_count);
  }
  std::vector<uint8_t> gather_indices_bytes;
  if (needs_gather) {
    gather_indices_bytes.resize(gather_indices.size() * sizeof(int32_t));
    std::memcpy(gather_indices_bytes.data(), gather_indices.data(), gather_indices_bytes.size());
  }
  // Static int32 indices (QNN Gather has no int64 static path); built only when reordering.
  std::optional<QnnTensorWrapper> gather_indices_tensor;
  if (needs_gather) {
    gather_indices_tensor.emplace(
        base_name + "_gather_idx", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_INT_32, QnnQuantParamsWrapper(),
        std::vector<uint32_t>{static_cast<uint32_t>(gather_indices.size())}, std::move(gather_indices_bytes));
  }

  Qnn_Scalar_t gather_axis_scalar = QNN_SCALAR_INIT;
  gather_axis_scalar.dataType = QNN_DATATYPE_INT_32;
  gather_axis_scalar.int32Value = 1;
  QnnParamWrapper gather_axis_param(concat_unit.Index(), base_name + "_gather",
                                    QNN_OP_GATHER_PARAM_AXIS, gather_axis_scalar);

  std::vector<uint32_t> block_shape{2};
  std::vector<uint32_t> block_data{SliceConcatSpaceToDepthFusion::kBlockHeight,
                                   SliceConcatSpaceToDepthFusion::kBlockWidth};
  QnnParamWrapper block_param(concat_unit.Index(), s2d_name, QNN_OP_SPACE_TO_DEPTH_PARAM_BLOCK_SIZE,
                              std::move(block_shape), std::move(block_data));
  Qnn_Scalar_t mode_scalar = QNN_SCALAR_INIT;
  mode_scalar.dataType = QNN_DATATYPE_UINT_32;
  mode_scalar.uint32Value = QNN_OP_SPACE_TO_DEPTH_MODE_DCR;
  QnnParamWrapper mode_param(concat_unit.Index(), s2d_name, QNN_OP_SPACE_TO_DEPTH_PARAM_MODE, mode_scalar);

  if (validate) {
    RETURN_IF_ERROR(AddLayoutTranspose(model_wrapper, concat_unit, pre_name, focus_input_def.name,
                                       nhwc_input_name, {0, 2, 3, 1}, /*validate*/ true,
                                       {focus_input_tensor.GetQnnTensor()},
                                       {nhwc_input_tensor.GetQnnTensor()}));
    {
      std::vector<Qnn_Param_t> params{block_param.GetQnnParam(), mode_param.GetQnnParam()};
      RETURN_IF_ERROR(model_wrapper.ValidateQnnNode(s2d_name, QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                    QNN_OP_SPACE_TO_DEPTH,
                                                    {nhwc_input_tensor.GetQnnTensor()},
                                                    {nhwc_output_tensor.GetQnnTensor()}, std::move(params)));
    }
    RETURN_IF_ERROR(AddLayoutTranspose(model_wrapper, concat_unit, post_name, nhwc_output_name,
                                       nchw_s2d_name, {0, 3, 1, 2}, /*validate*/ true,
                                       {nhwc_output_tensor.GetQnnTensor()},
                                       {nchw_s2d_tensor.GetQnnTensor()}));
    if (needs_gather) {
      std::vector<Qnn_Param_t> params{gather_axis_param.GetQnnParam()};
      RETURN_IF_ERROR(model_wrapper.ValidateQnnNode(base_name + "_gather", QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                    QNN_OP_GATHER,
                                                    {nchw_s2d_tensor.GetQnnTensor(),
                                                     gather_indices_tensor->GetQnnTensor()},
                                                    {focus_output_tensor.GetQnnTensor()}, std::move(params)));
    }
    return Ort::Status();
  }

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
  RETURN_IF_NOT(model_wrapper.AddTensorWrapper(std::move(nhwc_output_tensor)), "Bad NHWC S2D output.");
  RETURN_IF_NOT(model_wrapper.AddTensorWrapper(std::move(nchw_s2d_tensor)), "Bad NCHW S2D output.");
  if (needs_gather) {
    RETURN_IF_NOT(model_wrapper.AddTensorWrapper(std::move(*gather_indices_tensor)), "Bad Gather indices.");
  }

  RETURN_IF_ERROR(AddLayoutTranspose(model_wrapper, concat_unit, pre_name, focus_input_def.name,
                                     nhwc_input_name, {0, 2, 3, 1}, /*validate*/ false, {}, {}));
  {
    const std::string block_name = block_param.GetParamTensorName();
    const std::string mode_name = mode_param.GetParamTensorName();
    RETURN_IF_NOT(model_wrapper.AddParamWrapper(std::move(block_param)), "Failed to add S2D block param.");
    RETURN_IF_NOT(model_wrapper.AddParamWrapper(std::move(mode_param)), "Failed to add S2D mode param.");
    RETURN_IF_NOT(model_wrapper.CreateQnnNode(s2d_name, QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_SPACE_TO_DEPTH,
                                              {nhwc_input_name}, {nhwc_output_name}, {block_name, mode_name},
                                              /*validate*/ false),
                  "Failed to add SpaceToDepth.");
  }
  RETURN_IF_ERROR(AddLayoutTranspose(model_wrapper, concat_unit, post_name, nhwc_output_name, nchw_s2d_name,
                                     {0, 3, 1, 2}, /*validate*/ false, {}, {}));

  if (needs_gather) {
    const std::string gather_name = base_name + "_gather";
    const std::string axis_name = gather_axis_param.GetParamTensorName();
    RETURN_IF_NOT(model_wrapper.AddParamWrapper(std::move(gather_axis_param)), "Failed to add Gather axis.");
    RETURN_IF_NOT(model_wrapper.CreateQnnNode(gather_name, QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_GATHER,
                                              {nchw_s2d_name, base_name + "_gather_idx"},
                                              {focus_output_def.name}, {axis_name}, /*validate*/ false),
                  "Failed to add channel Gather.");
  }
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
    if (width_slices[i] == nullptr ||
        !HasExactlyUnitConsumers(*width_slices[i], {&concat_unit}, node_to_unit)) {
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
  // The two H slices split the four W slices 2+2; positions are order-free here,
  // exact Concat order is enforced later via the DCR phase permutation.
  const OrtNodeUnit* height_slice_a = height_slice_for_width[0];
  const OrtNodeUnit* height_slice_b = nullptr;
  for (size_t i = 1; i < height_slice_for_width.size(); ++i) {
    if (height_slice_for_width[i] != height_slice_a) {
      height_slice_b = height_slice_for_width[i];
      break;
    }
  }
  if (height_slice_b == nullptr) {
    return nullptr;
  }
  std::vector<const OrtNodeUnit*> height_a_consumers;
  std::vector<const OrtNodeUnit*> height_b_consumers;
  for (size_t i = 0; i < height_slice_for_width.size(); ++i) {
    if (height_slice_for_width[i] == height_slice_a) {
      height_a_consumers.push_back(width_slices[i]);
    } else if (height_slice_for_width[i] == height_slice_b) {
      height_b_consumers.push_back(width_slices[i]);
    } else {
      return nullptr;
    }
  }
  if (height_a_consumers.size() != 2 || height_b_consumers.size() != 2 ||
      !HasExactlyUnitConsumers(*height_slice_a, height_a_consumers, node_to_unit) ||
      !HasExactlyUnitConsumers(*height_slice_b, height_b_consumers, node_to_unit)) {
    return nullptr;
  }

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

  // Effective (h, w) phase per Concat input; any permutation of the four is accepted,
  // with the Gather restoring exact Concat order downstream.
  std::array<SlicePhase, 4> actual_phases{};
  for (size_t i = 0; i < width_slices.size(); ++i) {
    std::vector<uint32_t> parent_height_shape;
    if (!QnnModelWrapper::GetOnnxShape(height_slice_for_width[i]->Outputs()[0].shape, parent_height_shape)) {
      return nullptr;
    }
    const auto width_spec = ParseStepTwoSlice(model_wrapper, *width_slices[i], parent_height_shape);
    if (!width_spec.has_value() || width_spec->axis != kWidthAxis) {
      return nullptr;
    }
    const int64_t parent_h_start = (height_slice_for_width[i] == height_slice_a) ? height_spec_a->start
                                                                                 : height_spec_b->start;
    actual_phases[i] = SlicePhase{parent_h_start, width_spec->start};
  }
  std::array<int64_t, 4> dcr_permutation{0, 1, 2, 3};
  if (!TryGetDcrPhasePermutation(actual_phases, dcr_permutation)) {
    return nullptr;
  }

  // Every boundary tensor shares one per-tensor quant; S2D+Gather only rearrange,
  // so any intermediate requant would be skipped and change numerics.
  const OrtNodeUnitIODef* boundary_defs[] = {
      &height_slice_a->Inputs()[0],
      &height_slice_a->Outputs()[0],
      &height_slice_b->Outputs()[0],
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
      height_slice_a, height_slice_b, width_slices[0], width_slices[1],
      width_slices[2], width_slices[3], &concat_unit};
  return std::make_unique<SliceConcatSpaceToDepthFusion>(
      gsl::make_span<const OrtNodeUnit* const>(focus_group.data(), focus_group.size()),
      dcr_permutation, channels);
}

Ort::Status SliceConcatSpaceToDepthFusion::IsSupported(QnnModelWrapper& model_wrapper,
                                                       const Ort::Logger& logger) const {
  if (model_wrapper.GetQnnBackendType() != QnnBackendType::HTP &&
      model_wrapper.GetQnnBackendType() != QnnBackendType::HTP_FP16) {
    return MAKE_EP_FAIL("SliceConcatS2D: HTP only.");
  }
  if (node_units_.size() != kGroupSize || !IsConcatUnit(concat_node_unit_)) {
    return MAKE_EP_FAIL("SliceConcatS2D: expected 7 units with Concat target.");
  }
  std::vector<uint32_t> input_shape, output_shape;
  if (!QnnModelWrapper::GetOnnxShape(node_units_[0]->Inputs()[0].shape, input_shape) ||
      !QnnModelWrapper::GetOnnxShape(concat_node_unit_->Outputs()[0].shape, output_shape)) {
    return MAKE_EP_FAIL("SliceConcatS2D: unresolved shapes.");
  }
  if (input_shape.size() != kNchwRank || output_shape.size() != kNchwRank ||
      std::any_of(input_shape.begin(), input_shape.end(), [](uint32_t dim) { return dim == 0; }) ||
      std::any_of(output_shape.begin(), output_shape.end(), [](uint32_t dim) { return dim == 0; })) {
    return MAKE_EP_FAIL("SliceConcatS2D: rank-4 non-zero shapes only.");
  }
  ORT_UNUSED_PARAMETER(logger);
  return CreateOrValidateFocusGraph(model_wrapper, *node_units_[0], *concat_node_unit_, phase_permutation_,
                                    channel_count_, /*validate*/ true, logger);
}

Ort::Status SliceConcatSpaceToDepthFusion::AddToModelBuilder(QnnModelWrapper& model_wrapper,
                                                             const Ort::Logger& logger) const {
  return CreateOrValidateFocusGraph(model_wrapper, *node_units_[0], *concat_node_unit_, phase_permutation_,
                                    channel_count_, /*validate*/ false, logger);
}

}  // namespace qnn
}  // namespace onnxruntime
