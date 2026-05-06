#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${1:-${REPO_ROOT}/data/euroc}"
mkdir -p "${DATA_ROOT}"

SOURCE_BASES=()
if [[ -n "${EUROC_MIRROR_BASE:-}" ]]; then
  SOURCE_BASES+=("${EUROC_MIRROR_BASE%/}")
fi
SOURCE_BASES+=(
  "https://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset"
  "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset"
)

download_one() {
  local name="$1"
  local archive="${DATA_ROOT}/${name}.zip"
  local extract_dir="${DATA_ROOT}/${name}"
  shift
  local relative_path="$1"

  if [[ -f "${extract_dir}/mav0/imu0/data.csv" ]]; then
    echo "Dataset ${name} already exists at ${extract_dir}"
    return
  fi

  local downloaded=0
  for base in "${SOURCE_BASES[@]}"; do
    local url="${base}/${relative_path}"
    echo "Downloading ${name} from ${url} ..."
    if curl -L --fail --retry 3 --retry-delay 5 "${url}" -o "${archive}"; then
      downloaded=1
      break
    fi
  done

  if [[ "${downloaded}" -ne 1 ]]; then
    echo "Failed to download ${name}. If you already have the sequence, place it under ${extract_dir}."
    echo "You can also export EUROC_MIRROR_BASE to point at another mirror root."
    return 1
  fi

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

download_status=0
download_one "MH_01_easy" "machine_hall/MH_01_easy/MH_01_easy.zip" || download_status=1
download_one "V2_02_medium" "vicon_room2/V2_02_medium/V2_02_medium.zip" || download_status=1
exit "${download_status}"
