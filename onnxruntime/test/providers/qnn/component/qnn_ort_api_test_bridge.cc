// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// Windows internal-symbol unit-test bridge.
//
// The ORT C++ wrapper stores its OrtApi pointer in a header-only function-local
// static. On Windows, onnxruntime_provider_test.exe and
// onnxruntime_providers_qnn.dll each get their own copy. Fake OrtNode/FakeGraph
// tests therefore need a small entry point compiled into the provider DLL so the
// DLL's copy can be pointed at the fake OrtApi table for the duration of a test.
// The SharedContext singleton has the same per-module behavior, so tests that
// exercise shared backend state must also update the provider DLL's copy.

#include "core/providers/qnn/ort_api.h"

#include "test/providers/qnn/infra/qnn_test_windows_compat.h"

#include "core/providers/qnn/builder/qnn_backend_manager.h"
#include "core/providers/qnn/shared_context.h"

extern "C" const OrtApi* ORT_API_CALL QnnUnit_SetOrtApiForTesting(const OrtApi* api) noexcept {
  const OrtApi* previous = Ort::detail::Global::Api();
  Ort::detail::Global::Api(api);
  return previous;
}

extern "C" void ORT_API_CALL QnnUnit_RestoreOrtApiForTesting(const OrtApi* previous) noexcept {
  Ort::detail::Global::Api(previous);
}

extern "C" bool ORT_API_CALL QnnUnit_SetSharedBackendManagerAllocatorForTesting(
    uint32_t allocator_type) noexcept {
  auto manager = onnxruntime::SharedContext::GetInstance().GetSharedQnnBackendManager();
  if (manager == nullptr) {
    return false;
  }

  manager->SetQnnAllocatorType(static_cast<onnxruntime::qnn::QnnAllocatorType>(allocator_type));
  return true;
}

extern "C" void ORT_API_CALL QnnUnit_ResetSharedBackendManagerForTesting() noexcept {
  onnxruntime::SharedContext::GetInstance().ResetSharedQnnBackendManager();
}
