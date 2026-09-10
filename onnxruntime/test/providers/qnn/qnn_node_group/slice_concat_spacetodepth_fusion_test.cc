// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD)

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "test/providers/qnn/qnn_node_group/qnn_graph_checker.h"
#include "test/providers/qnn/qnn_test_utils.h"
#include "test/unittest_util/qdq_test_utils.h"
#include "gtest/gtest.h"

// Declared in test_main.cc.
extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {
namespace test {

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

namespace {

enum class IndexElementType { kInt64,
                              kInt32 };

// [1,3,8,8] Focus stem; Pair B perf shape [1,3,640,640] shares the pattern with generic N.
GetTestModelFn BuildFocusTestCase(bool use_qdq, bool use_contrib_qdq, bool wrong_order = false,
                                  IndexElementType index_type = IndexElementType::kInt64,
                                  bool mismatched_scales = false) {
  return [=](ModelTestBuilder& builder) -> void {
    builder.graph_->set_name("focus_slice_concat_graph");
    const std::vector<int64_t> input_shape{1, 3, 8, 8};
    const auto input_def = TestInputDef<float>(input_shape, false, -1.0f, 1.0f);
    MakeTestInput<float>(builder, "image", input_def);

    std::string stem_in = "image";
    QuantParams<uint16_t> stem_quant{0.05f, 0};
    if (use_qdq) {
      stem_quant = GetTestInputQuantParams<uint16_t>(input_def);
      stem_in = AddQDQNodePair<uint16_t>(builder, "qdq_in", "image", stem_quant.scale,
                                         stem_quant.zero_point, use_contrib_qdq);
    }

    auto add_slice = [&](const std::string& name, const std::string& data, int64_t axis,
                         int64_t start, int64_t end, const std::string& output) {
      if (index_type == IndexElementType::kInt64) {
        builder.Make1DInitializer<int64_t>(name + "_starts", {start});
        builder.Make1DInitializer<int64_t>(name + "_ends", {end});
        builder.Make1DInitializer<int64_t>(name + "_axes", {axis});
        builder.Make1DInitializer<int64_t>(name + "_steps", {2});
      } else {
        builder.Make1DInitializer<int32_t>(name + "_starts", {static_cast<int32_t>(start)});
        builder.Make1DInitializer<int32_t>(name + "_ends", {static_cast<int32_t>(end)});
        builder.Make1DInitializer<int32_t>(name + "_axes", {static_cast<int32_t>(axis)});
        builder.Make1DInitializer<int32_t>(name + "_steps", {2});
      }
      builder.AddNode(name, "Slice",
                      {data, name + "_starts", name + "_ends", name + "_axes", name + "_steps"},
                      {output}, kOnnxDomain);
    };

    add_slice("SliceH0", stem_in, 2, 0, 8, "h0");
    add_slice("SliceH1", stem_in, 2, 1, 8, "h1");
    add_slice("SliceH0W0", "h0", 3, 0, 8, "h0w0");
    add_slice("SliceH1W0", "h1", 3, 0, 8, "h1w0");
    add_slice("SliceH0W1", "h0", 3, 1, 8, "h0w1");
    add_slice("SliceH1W1", "h1", 3, 1, 8, "h1w1");

    // Optional intermediate requant with a different scale on one branch.
    // Fusion must fail closed here; S2D would skip the requant error.
    std::string h1w1_in = "h1w1";
    if (mismatched_scales) {
      h1w1_in = AddQDQNodePair<uint16_t>(builder, "qdq_mismatch", "h1w1", stem_quant.scale * 2.0f,
                                         stem_quant.zero_point, use_contrib_qdq);
    }

    const std::vector<std::string> concat_inputs =
        wrong_order ? std::vector<std::string>{"h0w0", "h0w1", "h1w0", h1w1_in}
                    : std::vector<std::string>{"h0w0", "h1w0", "h0w1", h1w1_in};
    builder.AddNode("Concat", "Concat", concat_inputs, {"cat_out"}, kOnnxDomain,
                    {test::MakeAttribute("axis", static_cast<int64_t>(1))});

    std::string tail_in = "cat_out";
    if (use_qdq) {
      tail_in = AddQDQNodePair<uint16_t>(builder, "qdq_out", "cat_out", stem_quant.scale,
                                         stem_quant.zero_point, use_contrib_qdq);
    }
    builder.MakeInitializer<float>("stem_w", {12, 12, 1, 1}, -1.f, 1.f);
    builder.AddNode("StemConv", "Conv", {tail_in, "stem_w"}, {"Y"}, kOnnxDomain);
    builder.MakeOutput("Y");
  };
}

ProviderOptions HtpOptions() {
  ProviderOptions options;
  options["backend_type"] = "htp";
  options["offload_graph_io_quantization"] = "0";
  return options;
}

void RunFocusFusionTest(const std::filesystem::path& dir, GetTestModelFn model_fn, bool expect_fused,
                        float tolerance = 1e-2f) {
  std::filesystem::remove_all(dir);
  ASSERT_TRUE(std::filesystem::create_directory(dir));
  auto cleanup = gsl::finally([&dir]() { std::filesystem::remove_all(dir); });

  ProviderOptions options = HtpOptions();
  options["dump_json_qnn_graph"] = "1";
  options["json_qnn_graph_dir"] = dir.string();

  RunQnnModelTest(model_fn, options, /*opset_version=*/13,
                  EPVerificationParams{ExpectedEPNodeAssignment::All,
                                       ElementwiseAbsoluteVerifier(tolerance)},
                  OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE);
  AssertOpInQnnGraph(dir, "SpaceToDepth", expect_fused ? 1 : 0);
}

}  // namespace

TEST_F(QnnHTPBackendTests, FocusSliceConcat_Float_Fused) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  RunFocusFusionTest("FocusSliceConcatFloat_HTP", BuildFocusTestCase(false, false), true);
}

TEST_F(QnnHTPBackendTests, FocusSliceConcat_Float_Int32Indices_Fused) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  RunFocusFusionTest("FocusSliceConcatInt32_HTP",
                     BuildFocusTestCase(false, false, false, IndexElementType::kInt32), true);
}

TEST_F(QnnHTPBackendTests, FocusSliceConcat_QDQ_U16_Fused) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  RunFocusFusionTest("FocusSliceConcatQDQU16_HTP", BuildFocusTestCase(true, true), true, 3e-2f);
}

TEST_F(QnnHTPBackendTests, FocusSliceConcat_WrongOrder_NotFused) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  RunFocusFusionTest("FocusSliceConcatWrongOrder_HTP",
                     BuildFocusTestCase(false, false, /*wrong_order=*/true), false);
}

TEST_F(QnnHTPBackendTests, FocusSliceConcat_MismatchedScales_NotFused) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  RunFocusFusionTest("FocusSliceConcatMismatch_HTP",
                     BuildFocusTestCase(true, true, false, IndexElementType::kInt64,
                                        /*mismatched_scales=*/true),
                     false, 3e-2f);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
