#!/bin/bash
# iampred client launcher: ./run.sh <command> [arguments]   (commands: analyze, static, intake, predict, report, check)
PKG="$(cd "$(dirname "$0")" && pwd)"
export IAM_GIMAP="$PKG/code/gimap" IAM_DIST="$PKG/code/toolkit" IAM_WORK="$PKG/work" IAM_RAW="$PKG/no_raw"
export IAM_DATA="${IAM_DATA:-$PKG/data}" IAMPRED_MODELS="$PKG/models" PYTHONPATH="$PKG/code/iam:$PYTHONPATH"
export TZ=Europe/Berlin OMP_NUM_THREADS=1
cmd="$1"; shift
exec "${IAMPRED_PY:-python3}" -m "iampred.$cmd" "$@"
