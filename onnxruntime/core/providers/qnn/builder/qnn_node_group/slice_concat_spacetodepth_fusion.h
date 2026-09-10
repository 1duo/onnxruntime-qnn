// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "core/providers/qnn/builder/qnn_node_group/qnn_node_group.h"
#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {

class QnnModelWrapper;

// YOLOX Focus stem (2 H-Slice + 4 W-Slice + Concat) fused to
// QNN SpaceToDepth (DCR) + channel Gather. Mirrors upstream ORT
// SliceConcatToSpaceToDepthFusion: canonical DCR order needs no Gather,
// any other phase order (e.g. Focus h0w0,h1w0,h0w1,h1w1) is restored
// with a Gather over the 4C channels. Bare S2D alone is wrong here.
class SliceConcatSpaceToDepthFusion : public IQnnNodeGroup {
 public:
  static constexpr uint32_t kBlockHeight = 2;
  static constexpr uint32_t kBlockWidth = 2;
  // Order: H(start0), H(start1), W(h0w0), W(h1w0), W(h0w1), W(h1w1), Concat.
  static constexpr size_t kGroupSize = 7;

  SliceConcatSpaceToDepthFusion(gsl::span<const OrtNodeUnit* const> focus_node_units,
                                std::array<int64_t, 4> dcr_phase_permutation,
                                uint32_t focus_channels)
      : phase_permutation_(dcr_phase_permutation), channel_count_(focus_channels) {
    node_units_.reserve(focus_node_units.size());
    for (const OrtNodeUnit* node_unit : focus_node_units) {
      node_units_.push_back(node_unit);
    }
    concat_node_unit_ = node_units_.empty() ? nullptr : node_units_.back();
  }
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(SliceConcatSpaceToDepthFusion);

  Ort::Status IsSupported(QnnModelWrapper& qnn_model_wrapper, const Ort::Logger& logger) const override;
  Ort::Status AddToModelBuilder(QnnModelWrapper& qnn_model_wrapper, const Ort::Logger& logger) const override;
  gsl::span<const OrtNodeUnit* const> GetNodeUnits() const override;
  const OrtNodeUnit* GetTargetNodeUnit() const override { return concat_node_unit_; }
  std::string_view Type() const override { return "SliceConcatSpaceToDepthFusion"; }

  static std::unique_ptr<IQnnNodeGroup> TryFusion(
      QnnModelWrapper& qnn_model_wrapper,
      const OrtNodeUnit& concat_node_unit,
      const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
      const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
      const Ort::Logger& logger);

 private:
  std::vector<const OrtNodeUnit*> node_units_;
  const OrtNodeUnit* concat_node_unit_ = nullptr;
  // Canonical DCR phase index per Concat input; {0,1,2,3} needs no Gather.
  std::array<int64_t, 4> phase_permutation_{0, 1, 2, 3};
  uint32_t channel_count_ = 0;
};

}  // namespace qnn
}  // namespace onnxruntime
