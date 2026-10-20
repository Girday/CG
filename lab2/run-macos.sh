#!/usr/bin/env bash
set -euo pipefail

lab_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${lab_dir}/.." && pwd)"
sdk_root="${CG_VULKAN_SDK_ROOT:-${HOME}/VulkanSDK/1.4.321.0}"
if [[ ! -f "${sdk_root}/setup-env.sh" ]]; then
    echo "Vulkan SDK not found: ${sdk_root}/setup-env.sh" >&2
    exit 1
fi
export PKG_CONFIG_PATH="${PKG_CONFIG_PATH:-}"
source "${sdk_root}/setup-env.sh" >/dev/null
cmake -S "${repo_dir}" -B "${repo_dir}/build-debug" \
    -DCMAKE_BUILD_TYPE=Debug -DFETCHCONTENT_UPDATES_DISCONNECTED=ON
cmake --build "${repo_dir}/build-debug" --target cg-lab2 --parallel
cd "${lab_dir}"
exec "${repo_dir}/build-debug/lab2/cg-lab2"
