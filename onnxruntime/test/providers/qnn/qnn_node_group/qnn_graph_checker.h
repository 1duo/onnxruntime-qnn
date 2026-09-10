// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#pragma once

#include <filesystem>
#include <string>

namespace onnxruntime {
namespace test {

// Asserts that the given QNN op type appears exactly `count` times in
// the compiled QNN graph JSON (root["graph"]["nodes"][*]["type"]).
// Finds the JSON graph file in `dump_dir`, skipping tensor log files.
void AssertOpInQnnGraph(const std::filesystem::path& dump_dir,
                        const std::string& op,
                        size_t count = 1);

// Asserts that a node with the exact `node_name` does not appear in
// the compiled QNN graph JSON (root["graph"]["nodes"]).
void AssertNodeNotInQnnGraph(const std::filesystem::path& dump_dir,
                             const std::string& node_name);

// Sums FP32 STATIC tensor bytes in the compiled QNN graph JSON
// (root["graph"]["tensors"][*]) and asserts the total is below max_bytes.
// Tensor "type" 4 is QNN_TENSOR_TYPE_STATIC and "data_type" 562 is
// QNN_DATATYPE_FLOAT_32; STATIC float tensors report "params_count".
// Used to prove materialization policy: e.g. a grouped large-weight graph must
// carry ~0 FP32 STATIC bytes, while the folded equivalent carries elems*4.
void AssertFp32StaticBytesBelow(const std::filesystem::path& dump_dir, size_t max_bytes);

// Inverse: asserts the graph carries at least min_bytes of FP32 STATIC tensors.
// Pairs with Below on folded graphs (folding must materialize!): Below alone would
// pass vacuously if lowering ever stopped emitting the tensor at all.
void AssertFp32StaticBytesAbove(const std::filesystem::path& dump_dir, size_t min_bytes);

}  // namespace test
}  // namespace onnxruntime
