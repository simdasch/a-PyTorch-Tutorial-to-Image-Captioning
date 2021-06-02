#!/usr/bin/env bash

# shellcheck disable=SC2164
SCRIPT_DIR="$(cd "$(dirname "$0")"; pwd -P)"
ROOT_DIR=${SCRIPT_DIR}/../../

if [ -z "$1" ]
then
    echo "No conda environment name supplied."
    exit 1
fi
CONDA_ENV_NAME=$1

# shellcheck disable=SC2164
cd "${ROOT_DIR}"

conda env create \
    -n "${CONDA_ENV_NAME}" \
    -f environment.yaml
