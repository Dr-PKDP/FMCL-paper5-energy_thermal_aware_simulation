#!/usr/bin/env bash
# Downloads and verifies both real FedScale data files needed for the
# Section 7.7 and 7.9 empirical-grounding checks. Neither file belongs to
# this paper -- both are FedScale's, and are downloaded fresh rather than
# bundled in this repository. This script exists because a previous
# attempt at this step failed silently: the download either did not run
# or was not verified, and the resulting gap was only caught when a
# reviewer asked where the data came from. This script is deliberately
# loud on failure so that cannot happen again unnoticed.
#
# Usage: bash tracesim/setup_data.sh
# Run from the repository root, or from tracesim/ -- both work.

set -euo pipefail

# Resolve paths regardless of where this script is invoked from
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/data"
mkdir -p "$DATA_DIR"

# --- File 1: client behaviour / availability trace (Sections 7.7, 7.9) ---
BEHAVE_URL="https://raw.githubusercontent.com/SymbioticLab/FedScale/master/benchmark/dataset/data/device_info/client_behave_trace"
BEHAVE_PATH="$DATA_DIR/client_behave_trace"
BEHAVE_SHA256="d0b6f81a01f0ea5f7ec432583f5afe93b1e72424db4c1affa88c484f15901662"
BEHAVE_SIZE=25640150

# --- File 2: client device computation/communication capacity (Section 7.7's
#     third check, the 500,000-measurement compute-heterogeneity comparison) ---
CAPACITY_URL="https://raw.githubusercontent.com/SymbioticLab/FedScale/master/benchmark/dataset/data/device_info/client_device_capacity"
CAPACITY_PATH="$DATA_DIR/client_device_capacity"
CAPACITY_SHA256="477f61049443987318fc5667cafa059b3b420eb0e03ccb5ceb3262d20e059e53"
CAPACITY_SIZE=39369071

download_and_verify () {
    local url="$1" path="$2" expected_sha="$3" expected_size="$4" label="$5"

    if [ -f "$path" ]; then
        actual_sha=$(sha256sum "$path" | awk '{print $1}')
        if [ "$actual_sha" = "$expected_sha" ]; then
            echo "[OK] $label already present and verified at $path"
            return 0
        else
            echo "[WARN] $label exists at $path but checksum does not match -- re-downloading"
            rm -f "$path"
        fi
    fi

    echo "[..] Downloading $label ..."
    curl -L --fail -o "$path" "$url"

    actual_size=$(stat -c%s "$path" 2>/dev/null || stat -f%z "$path")
    if [ "$actual_size" != "$expected_size" ]; then
        echo "[FAIL] $label: expected $expected_size bytes, got $actual_size bytes."
        echo "       FedScale may have reorganised its repository. Check"
        echo "       https://github.com/SymbioticLab/FedScale/tree/master/benchmark/dataset"
        echo "       and update this script's URL if so. Refusing to proceed with a"
        echo "       file that does not match the expected size."
        rm -f "$path"
        exit 1
    fi

    actual_sha=$(sha256sum "$path" | awk '{print $1}')
    if [ "$actual_sha" != "$expected_sha" ]; then
        echo "[FAIL] $label: checksum mismatch after download."
        echo "       Expected: $expected_sha"
        echo "       Got:      $actual_sha"
        echo "       Do not use this file for reproduction until this is resolved."
        exit 1
    fi

    echo "[OK] $label downloaded and verified ($actual_size bytes, sha256 $actual_sha)"
}

download_and_verify "$BEHAVE_URL" "$BEHAVE_PATH" "$BEHAVE_SHA256" "$BEHAVE_SIZE" \
    "client_behave_trace (107,749-device availability trace)"

download_and_verify "$CAPACITY_URL" "$CAPACITY_PATH" "$CAPACITY_SHA256" "$CAPACITY_SIZE" \
    "client_device_capacity (500,000-entry compute/communication capacity trace)"

echo
echo "Both files verified. You can now run:"
echo "  cd $SCRIPT_DIR && python3 trace_run.py          # Section 7.9"
echo "  cd $SCRIPT_DIR && python3 trace_summary_stats.py # Section 7.7"
