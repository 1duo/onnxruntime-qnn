// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "test/providers/qnn/qnn_node_group/qnn_graph_checker.h"

#include <fstream>

#include "nlohmann/json.hpp"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

void AssertOpInQnnGraph(const std::filesystem::path& dump_dir,
                        const std::string& op,
                        size_t count) {
  std::filesystem::path json_path;
  for (const auto& entry : std::filesystem::directory_iterator{dump_dir}) {
    if (entry.is_regular_file() && entry.path().extension() == ".json" &&
        entry.path().filename().string().find("_tensor_log") == std::string::npos) {
      json_path = entry.path();
      break;
    }
  }
  ASSERT_FALSE(json_path.empty()) << "No QNN JSON graph file found in " << dump_dir;

  std::ifstream json_file(json_path);
  ASSERT_TRUE(json_file.is_open()) << "Failed to open QNN JSON graph: " << json_path;

  nlohmann::json root;
  json_file >> root;

  ASSERT_TRUE(root.contains("graph") && root["graph"].contains("nodes"))
      << "JSON missing 'graph.nodes' field in: " << json_path;

  size_t actual_count = 0;
  for (const auto& [node_name, node_json] : root["graph"]["nodes"].items()) {
    if (node_json.value("type", "") == op) {
      ++actual_count;
    }
  }

  EXPECT_EQ(actual_count, count)
      << "QNN op '" << op << "': expected " << count
      << " occurrence(s), found " << actual_count << " in " << json_path;
}

void AssertNodeNotInQnnGraph(const std::filesystem::path& dump_dir,
                             const std::string& node_name) {
  std::filesystem::path json_path;
  for (const auto& entry : std::filesystem::directory_iterator{dump_dir}) {
    if (entry.is_regular_file() && entry.path().extension() == ".json" &&
        entry.path().filename().string().find("_tensor_log") == std::string::npos) {
      json_path = entry.path();
      break;
    }
  }
  ASSERT_FALSE(json_path.empty()) << "No QNN JSON graph file found in " << dump_dir;

  std::ifstream json_file(json_path);
  ASSERT_TRUE(json_file.is_open()) << "Failed to open QNN JSON graph: " << json_path;

  nlohmann::json root;
  json_file >> root;

  ASSERT_TRUE(root.contains("graph") && root["graph"].contains("nodes"))
      << "JSON missing 'graph.nodes' field in: " << json_path;

  EXPECT_FALSE(root["graph"]["nodes"].contains(node_name))
      << "Unexpected QNN node found: '" << node_name << "' in " << json_path;
}

namespace {

size_t SumFp32StaticBytes(const nlohmann::json& root) {
  size_t total_fp32_static_bytes = 0;
  if (root.contains("graph") && root["graph"].contains("tensors")) {
    for (const auto& [name, tensor_json] : root["graph"]["tensors"].items()) {
      if (tensor_json.value("type", -1) == 4 && tensor_json.value("data_type", -1) == 562) {
        const auto it = tensor_json.find("params_count");
        if (it == tensor_json.end()) {
          continue;
        }
        size_t elems = 0;
        if (it->is_string()) {
          elems = static_cast<size_t>(std::stoul(it->get<std::string>()));
        } else if (it->is_number_unsigned()) {
          elems = it->get<size_t>();
        } else if (it->is_number_integer()) {
          elems = static_cast<size_t>(it->get<int64_t>());
        }
        total_fp32_static_bytes += elems * 4;
      }
    }
  }
  return total_fp32_static_bytes;
}

std::filesystem::path FindQnnGraphJson(const std::filesystem::path& dump_dir) {
  for (const auto& entry : std::filesystem::directory_iterator{dump_dir}) {
    if (entry.is_regular_file() && entry.path().extension() == ".json" &&
        entry.path().filename().string().find("_tensor_log") == std::string::npos) {
      return entry.path();
    }
  }
  return {};
}

}  // namespace

void AssertFp32StaticBytesBelow(const std::filesystem::path& dump_dir, size_t max_bytes) {
  const std::filesystem::path json_path = FindQnnGraphJson(dump_dir);
  ASSERT_FALSE(json_path.empty()) << "No QNN JSON graph file found in " << dump_dir;

  std::ifstream json_file(json_path);
  ASSERT_TRUE(json_file.is_open()) << "Failed to open QNN JSON graph: " << json_path;

  nlohmann::json root;
  json_file >> root;

  EXPECT_LE(SumFp32StaticBytes(root), max_bytes)
      << "FP32 STATIC bytes in " << json_path << " exceed budget (weight folded instead of grouped?)";
}

void AssertFp32StaticBytesAbove(const std::filesystem::path& dump_dir, size_t min_bytes) {
  const std::filesystem::path json_path = FindQnnGraphJson(dump_dir);
  ASSERT_FALSE(json_path.empty()) << "No QNN JSON graph file found in " << dump_dir;

  std::ifstream json_file(json_path);
  ASSERT_TRUE(json_file.is_open()) << "Failed to open QNN JSON graph: " << json_path;

  nlohmann::json root;
  json_file >> root;

  EXPECT_GT(SumFp32StaticBytes(root), min_bytes)
      << "FP32 STATIC bytes in " << json_path << " below floor (folded weight missing?)";
}

}  // namespace test
}  // namespace onnxruntime
