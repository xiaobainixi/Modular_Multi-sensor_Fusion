#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${1:-${REPO_ROOT}/data/euroc}"
mkdir -p "${DATA_ROOT}"

download_one() {
  local name="$1"
  local url="$2"
  local archive="${DATA_ROOT}/${name}.zip"
  local extract_dir="${DATA_ROOT}/${name}"

  if [[ -f "${extract_dir}/mav0/imu0/data.csv" ]]; then
    echo "Dataset ${name} already exists at ${extract_dir}"
    return
  fi

  echo "Downloading ${name}..."
  curl -L --fail --retry 3 --retry-delay 5 "${url}" -o "${archive}"
  mkdir -p "${extract_dir}"
  unzip -o "${archive}" -d "${DATA_ROOT}"

  if [[ ! -d "${extract_dir}" ]]; then
    local moved_dir
    moved_dir="$(find "${DATA_ROOT}" -maxdepth 1 -type d -name "${name}*" | head -n 1 || true)"
    if [[ -n "${moved_dir}" && "${moved_dir}" != "${extract_dir}" ]]; then
      mv "${moved_dir}" "${extract_dir}"
    fi
  fi
}

download_one "MH_01_easy" "https://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_01_easy/MH_01_easy.zip"
download_one "V2_02_medium" "https://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room2/V2_02_medium/V2_02_medium.zip"
