#!/usr/bin/env bash
# Download Stage 1 inputs and lay out the expected directory structure.
#
# Usage:
#   bash setup_stage1_inputs.sh "https://drive.google.com/file/d/FILE_ID/view"

set -euo pipefail

DRIVE_URL="${1:-${DRIVE_URL:-}}"
SKIP_DRIVE="${SKIP_DRIVE:-0}"

DATA_ROOT="../../data"
TS8_DIR="${DATA_ROOT}/TS8"
UCSC_DIR="${DATA_ROOT}/UCSC/hg19"
FC_DIR="${DATA_ROOT}/fold_change"

TS8_BASE="https://www.targetscan.org/vert_80/vert_80_data_download"
TS8_ZIPS=(
    "Conserved_Site_Context_Scores.txt.zip"
    "Nonconserved_Site_Context_Scores.txt.zip"
    "Gene_info.txt.zip"
)

declare -A DRIVE_FILE_MAP=(
    ["3utr_sequences_hg19.txt"]="${UCSC_DIR}/3utr_sequences_hg19.txt"
    ["id_map"]="${UCSC_DIR}/id_map"
    ["mirna_fcs.csv"]="${FC_DIR}/mirna_fcs.csv"
)

mkdir -p "${UCSC_DIR}" "${FC_DIR}" "${TS8_DIR}" "${TS8_DIR}/processed"

# --- Drive zip ---------------------------------------------------------------
if [[ "${SKIP_DRIVE}" != "1" ]]; then
    if [[ -z "${DRIVE_URL}" ]]; then
        echo "Pass a Drive URL as argument (or set SKIP_DRIVE=1)." >&2
        exit 1
    fi

    if ! command -v gdown >/dev/null 2>&1; then
        python3 -m pip install --quiet --user gdown
        export PATH="${HOME}/.local/bin:${PATH}"
    fi

    ZIP_PATH="$(mktemp -t stage1.XXXXXX.zip)"
    STAGING="$(mktemp -d -t stage1_XXXXXX)"

    gdown --fuzzy "${DRIVE_URL}" -O "${ZIP_PATH}"
    unzip -j -o "${ZIP_PATH}" -d "${STAGING}" >/dev/null

    for src in "${!DRIVE_FILE_MAP[@]}"; do
        dst="${DRIVE_FILE_MAP[$src]}"
        if [[ -f "${STAGING}/${src}" ]]; then
            mv -f "${STAGING}/${src}" "${dst}"
            echo "  ${src} -> ${dst}"
        else
            echo "  WARNING: ${src} not found in zip" >&2
        fi
    done

    rm -rf "${STAGING}" "${ZIP_PATH}"
fi

# --- TargetScan 8 ------------------------------------------------------------
for zname in "${TS8_ZIPS[@]}"; do
    txtname="${zname%.zip}"
    dst="${TS8_DIR}/${txtname}"
    [[ -f "${dst}" ]] && { echo "  ${txtname} already present"; continue; }

    tmpzip="$(mktemp -t ts8.XXXXXX.zip)"
    echo "  downloading ${zname}"
    curl -L --fail --progress-bar -o "${tmpzip}" "${TS8_BASE}/${zname}"
    unzip -j -o "${tmpzip}" -d "${TS8_DIR}" >/dev/null
    rm -f "${tmpzip}"
    echo "  ${txtname} done"
done

# --- Verify ------------------------------------------------------------------
EXPECTED=(
    "${UCSC_DIR}/3utr_sequences_hg19.txt"
    "${UCSC_DIR}/id_map"
    "${FC_DIR}/mirna_fcs.csv"
    "${TS8_DIR}/Conserved_Site_Context_Scores.txt"
    "${TS8_DIR}/Nonconserved_Site_Context_Scores.txt"
    "${TS8_DIR}/Gene_info.txt"
)

missing=0
for p in "${EXPECTED[@]}"; do
    if [[ -e "${p}" ]]; then
        printf '  OK    %s\n' "${p}"
    else
        printf '  MISS  %s\n' "${p}"
        missing=1
    fi
done

[[ "${missing}" == "0" ]] || exit 1
echo "Done."
