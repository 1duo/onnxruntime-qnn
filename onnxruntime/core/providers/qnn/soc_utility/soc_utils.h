// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace onnxruntime {
namespace qnn {
namespace soc {

int GetSocId();

// Returns true on Android arm64 if ro.soc.manufacturer is "QTI" (case-insensitive).
// Returns true on Linux arm64 if /dev/fastrpc-cdsp* is present. Always false elsewhere.
bool HasFastRpcCdspDevice();

// Resolves a SoC model name (e.g. "SM8750", case-insensitive) to the corresponding
// Qnn_SocModel_t integer value. Returns 0 (QNN_SOC_MODEL_UNKNOWN) if not found.
uint32_t MapSocModelFromSocName(std::string_view soc_name);

// Resolves a SoC model to the corresponding QnnHtpDevice_Arch_t integer value.
// Returns 0 (QNN_HTP_DEVICE_ARCH_NONE) if not found.
uint32_t MapHtpArchFromSocModel(uint32_t soc_model);

// Parses the htp_arch session option. Empty/"0" means automatic architecture;
// unsupported or non-canonical values return nullopt.
std::optional<uint32_t> ParseHtpArchitecture(std::string_view architecture);

// EP metadata uses a concrete target (e.g. "htp-v73"), so automatic architecture
// is not accepted. These wrappers reuse the session-option architecture parser.
std::optional<uint32_t> ParseHtpTarget(std::string_view target);
std::string MakeHtpTarget(uint32_t architecture);

}  // namespace soc
}  // namespace qnn
}  // namespace onnxruntime
