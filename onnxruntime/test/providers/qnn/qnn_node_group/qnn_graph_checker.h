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

// Sums FP32 STATIC bytes (type 4 = STATIC, data_type 562 = FLOAT_32, via params_count)
// in the QNN JSON dump; asserts below max_bytes.
void AssertFp32StaticBytesBelow(const std::filesystem::path& dump_dir, size_t max_bytes);

// Inverse of Below: asserts at least min_bytes (proves a fold materialized).
void AssertFp32StaticBytesAbove(const std::filesystem::path& dump_dir, size_t min_bytes);

}  // namespace test
}  // namespace onnxruntime
