#!/usr/bin/env bash
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
#
# Run a test binary under AddressSanitizer and classify the result:
#
#   - Any ASan heap error (overflow, UAF, double-free, ...)     → exit 1
#   - Any gtest assertion failure ([  FAILED  ])                → exit 1
#   - Any Direct leak (post-suppression)                        → exit 1
#   - Only Indirect leaks remain (any source)                   → exit 0
#   - LSan tracer crash during the post-test scan, all tests OK → exit 0
#   - Anything else (segfault before gtest XML, ...)            → exit 1
#
# Rationale: Indirect leaks frequently have <unknown module> stacks because
# the QAIRT SDK backend libraries are shipped stripped, so they cannot be
# reliably attributed to a root and cannot be matched by `leak:` suppression
# patterns. Rather than chase that, we trust the Direct-leak signal and
# the suppression list in tools/ci_build/lsan_suppressions.txt to cover
# known stripped backend libs. ORT-side leaks would show as Direct.
#
# LSan tracer crash: QAIRT backend libraries use guard pages (mprotect
# PROT_NONE) for device memory. If a reachable chunk holds a pointer into a
# guard page, LSan's tracer takes SIGSEGV while scanning that chunk and exits
# with a fatal error before it can report anything (google/sanitizers#723).
# That interaction is out of our control and is not an ORT defect, so accept
# it only after gtest completed with every test passing. Real ASan errors,
# gtest failures, and Direct leaks still fail above.
#
# Usage:
#   asan_filter_leaks.sh <test_binary> [args...]

set -u  # NOTE: -e and pipefail are intentionally omitted.
# The core pipeline below must be allowed to "fail" (non-zero binary exit)
# so that PIPESTATUS[0] can be captured and the leak type classified.
# Adding pipefail/errexit would abort the script before the classification
# logic runs, defeating the purpose of this filter.

if [ "$#" -lt 1 ]; then
    echo "usage: $0 <test_binary> [args...]" >&2
    exit 2
fi

binary="$1"
shift

log_file=$(mktemp -t asan-filter.XXXXXX.log)
trap 'rm -f "$log_file"' EXIT

"$binary" "$@" 2>&1 | tee "$log_file"
binary_rc=${PIPESTATUS[0]}

# Clean run - no leaks, no ASan errors, all tests passed
if [ "$binary_rc" -eq 0 ]; then
    exit 0
fi

# Binary returned non-zero - classify the cause
echo "" >&2
echo "===== [asan-filter] binary exited ${binary_rc}, classifying =====" >&2

# 1. Any non-leak ASan error wins
if grep -q "ERROR: AddressSanitizer:" "$log_file"; then
    echo "[asan-filter] ASan heap error detected -> FAIL" >&2
    exit 1
fi

# 2. Any gtest assertion failure wins
if grep -q "\[  FAILED  \]" "$log_file"; then
    echo "[asan-filter] gtest assertion failed -> FAIL" >&2
    exit 1
fi

# 3. Any Direct leak originates in our (or test) code
if grep -q "^Direct leak" "$log_file"; then
    direct_count=$(grep -c "^Direct leak" "$log_file")
    echo "[asan-filter] ${direct_count} Direct leak(s) from ORT/test code -> FAIL" >&2
    exit 1
fi

# 4. Only Indirect leaks left -> not policed, pass
if grep -q "^Indirect leak" "$log_file"; then
    indirect_count=$(grep -c "^Indirect leak" "$log_file")
    summary=$(grep -E "SUMMARY: AddressSanitizer:.*leaked" "$log_file" | tail -1 || true)
    echo "[asan-filter] Only ${indirect_count} Indirect leak(s) (${summary})" >&2
    echo "[asan-filter] No Direct leaks; Indirect leaks are not policed -> PASS" >&2
    exit 0
fi

# 5. LSan died during the post-test scan before it could report a leak.
#    See the header comment: this is a known guard-page interaction in the
#    QAIRT backend libraries, not an ORT/test defect. Only accept it when the
#    gtest run completed with all tests passing; otherwise fall through and
#    fail as an unrecognized mode.
if grep -q "LeakSanitizer has encountered a fatal error" "$log_file"; then
    if grep -q "^\[  PASSED  \]" "$log_file" &&
        ! grep -q "^\[  FAILED  \]" "$log_file"; then
        echo "[asan-filter] WARNING: LSan tracer crashed during the post-test scan" >&2
        echo "[asan-filter] (known QAIRT guard-page interaction); no ASan error or Direct leak -> PASS" >&2
        exit 0
    fi
fi

# 6. Anything else (segfault, etc.) - fail
echo "[asan-filter] Unrecognized failure mode (binary exit=${binary_rc}) -> FAIL" >&2
echo "[asan-filter] Last 30 lines of binary output:" >&2
tail -n 30 "$log_file" >&2
exit 1
