#!/bin/bash
#SBATCH --partition=allcpu
#SBATCH --account=hasylab
#SBATCH --array=0-0
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=/data/dust/user/zhaiyufe/MaxwellRuns/v5-unbound-search-%A_%a.out
#SBATCH --error=/data/dust/user/zhaiyufe/MaxwellRuns/v5-unbound-search-%A_%a.err

set -euo pipefail

: "${SLURM_JOB_ID:?frozen search must run through Slurm}"
: "${SLURM_ARRAY_TASK_ID:?submit this wrapper as a Slurm array}"
: "${POSTERIOR_V8_SOURCE_ROOT:?set POSTERIOR_V8_SOURCE_ROOT}"
: "${POSTERIOR_V8_V5_EXPECTED_SOURCE_BUNDLE_SHA256:?set expected source bundle SHA-256}"
: "${POSTERIOR_V8_V5_LAUNCH_PLAN_SHA256:?set launch plan SHA-256}"
: "${POSTERIOR_V8_V5_SPLIT_PLAN:?set POSTERIOR_V8_V5_SPLIT_PLAN}"
: "${POSTERIOR_V8_V5_EXPECTED_SPLIT_PLAN_SHA256:?set expected split-plan SHA-256}"
: "${POSTERIOR_V8_V5_EXPECTED_SPLIT_PLAN_FILE_SHA256:?set expected split-plan file SHA-256}"
: "${POSTERIOR_V8_V5_SOBOL_DESIGN:?set POSTERIOR_V8_V5_SOBOL_DESIGN}"
: "${POSTERIOR_V8_V5_EXPECTED_SOBOL_DESIGN_SHA256:?set expected Sobol-design SHA-256}"
: "${POSTERIOR_V8_V5_EXPECTED_SOBOL_DESIGN_FILE_SHA256:?set expected Sobol-design file SHA-256}"
: "${POSTERIOR_V8_V5_LOCAL_SOBOL_SCHEDULE:?set POSTERIOR_V8_V5_LOCAL_SOBOL_SCHEDULE}"
: "${POSTERIOR_V8_V5_EXPECTED_LOCAL_SCHEDULE_SHA256:?set expected local schedule SHA-256}"
: "${POSTERIOR_V8_V5_EXPECTED_LOCAL_SCHEDULE_ARTIFACT_SHA256:?set expected local schedule artifact SHA-256}"
: "${POSTERIOR_V8_V5_EXPECTED_LOCAL_SCHEDULE_MANIFEST_SHA256:?set expected local schedule manifest SHA-256}"
: "${POSTERIOR_V8_V5_SEARCH_OUTPUT_DIR:?set POSTERIOR_V8_V5_SEARCH_OUTPUT_DIR}"
: "${POSTERIOR_V8_V5_TARGET_SPLIT:?set POSTERIOR_V8_V5_TARGET_SPLIT}"
: "${POSTERIOR_V8_V5_SPLIT_START:?set POSTERIOR_V8_V5_SPLIT_START}"
: "${POSTERIOR_V8_V5_RECIPES_PER_SHARD:?set POSTERIOR_V8_V5_RECIPES_PER_SHARD}"
: "${POSTERIOR_V8_V5_TOTAL_RECIPES:?set POSTERIOR_V8_V5_TOTAL_RECIPES}"
: "${POSTERIOR_V8_V5_VIEW_INDICES:?set POSTERIOR_V8_V5_VIEW_INDICES}"
: "${POSTERIOR_V8_V5_TOPOLOGY_SCHEDULE_ID:?set POSTERIOR_V8_V5_TOPOLOGY_SCHEDULE_ID}"
: "${POSTERIOR_V8_V5_SELECTED_TOPOLOGY_IDS:?set POSTERIOR_V8_V5_SELECTED_TOPOLOGY_IDS}"
: "${POSTERIOR_V8_V5_OPTIMIZER_SCHEDULE_ID:?set POSTERIOR_V8_V5_OPTIMIZER_SCHEDULE_ID}"
: "${POSTERIOR_V8_V5_DIRECT_SCOUT_SEEDS:?set POSTERIOR_V8_V5_DIRECT_SCOUT_SEEDS}"
: "${POSTERIOR_V8_V5_PER_SEED_FORWARD_LIMIT:?set POSTERIOR_V8_V5_PER_SEED_FORWARD_LIMIT}"
: "${POSTERIOR_V8_V5_PROTOCOL_ID:?set POSTERIOR_V8_V5_PROTOCOL_ID}"
: "${POSTERIOR_V8_V5_PROTOCOL_TIER:?set POSTERIOR_V8_V5_PROTOCOL_TIER}"
: "${POSTERIOR_V8_V5_FTOL:?set POSTERIOR_V8_V5_FTOL}"
: "${POSTERIOR_V8_V5_XTOL:?set POSTERIOR_V8_V5_XTOL}"
: "${POSTERIOR_V8_V5_GTOL:?set POSTERIOR_V8_V5_GTOL}"

case "$(hostname -s)" in
    max-wgs*) echo "frozen exact search cannot run on max-wgs" >&2; exit 2 ;;
esac
for dust_path in \
    "$POSTERIOR_V8_SOURCE_ROOT" \
    "$POSTERIOR_V8_V5_SPLIT_PLAN" \
    "$POSTERIOR_V8_V5_SOBOL_DESIGN" \
    "$POSTERIOR_V8_V5_LOCAL_SOBOL_SCHEDULE" \
    "$POSTERIOR_V8_V5_SEARCH_OUTPUT_DIR"; do
    case "$dust_path" in
        /data/dust/user/zhaiyufe/*) ;;
        *) echo "all V5.1 search inputs/outputs must be under /data/dust/user/zhaiyufe/" >&2; exit 2 ;;
    esac
done
for integer_value in \
    "$SLURM_ARRAY_TASK_ID" \
    "$POSTERIOR_V8_V5_SPLIT_START" \
    "$POSTERIOR_V8_V5_RECIPES_PER_SHARD" \
    "$POSTERIOR_V8_V5_TOTAL_RECIPES" \
    "$POSTERIOR_V8_V5_DIRECT_SCOUT_SEEDS" \
    "$POSTERIOR_V8_V5_PER_SEED_FORWARD_LIMIT"; do
    case "$integer_value" in
        ''|*[!0-9]*) echo "array/search count fields must be unsigned integers" >&2; exit 2 ;;
    esac
done
if (( POSTERIOR_V8_V5_RECIPES_PER_SHARD < 1 || POSTERIOR_V8_V5_TOTAL_RECIPES < 1 )); then
    echo "shard size and total recipe count must be positive" >&2
    exit 2
fi
case "$POSTERIOR_V8_V5_TARGET_SPLIT" in
    train|tuning_validation) ;;
    *) echo "search labels are restricted to train/tuning_validation" >&2; exit 2 ;;
esac

protocol_args=()
case "$POSTERIOR_V8_V5_PROTOCOL_TIER" in
    engineering_pilot)
        : "${POSTERIOR_V8_V5_STANDARDIZED_THRESHOLD_NAME:?set standardized threshold name}"
        : "${POSTERIOR_V8_V5_STANDARDIZED_THRESHOLD_VALUE:?set standardized threshold value}"
        : "${POSTERIOR_V8_V5_RAW_THRESHOLD_NAME:?set raw threshold name}"
        : "${POSTERIOR_V8_V5_RAW_THRESHOLD_VALUE:?set raw threshold value}"
        : "${POSTERIOR_V8_V5_THRESHOLD_SOURCE_ID:?set threshold source ID}"
        protocol_args=(
            --standardized-threshold-name "$POSTERIOR_V8_V5_STANDARDIZED_THRESHOLD_NAME"
            --standardized-threshold-value "$POSTERIOR_V8_V5_STANDARDIZED_THRESHOLD_VALUE"
            --raw-threshold-name "$POSTERIOR_V8_V5_RAW_THRESHOLD_NAME"
            --raw-threshold-value "$POSTERIOR_V8_V5_RAW_THRESHOLD_VALUE"
            --threshold-source-id "$POSTERIOR_V8_V5_THRESHOLD_SOURCE_ID"
        )
        ;;
    paper_full_calibrated)
        : "${POSTERIOR_V8_V5_CALIBRATION_ARTIFACT:?set calibration artifact}"
        : "${POSTERIOR_V8_V5_EXPECTED_CALIBRATION_SHA256:?set calibration logical SHA-256}"
        : "${POSTERIOR_V8_V5_EXPECTED_CALIBRATION_FILE_SHA256:?set calibration file SHA-256}"
        case "$POSTERIOR_V8_V5_CALIBRATION_ARTIFACT" in
            /data/dust/user/zhaiyufe/*) ;;
            *) echo "calibration artifact must be under /data/dust/user/zhaiyufe/" >&2; exit 2 ;;
        esac
        protocol_args=(
            --calibration-artifact "$POSTERIOR_V8_V5_CALIBRATION_ARTIFACT"
            --expected-calibration-sha256 "$POSTERIOR_V8_V5_EXPECTED_CALIBRATION_SHA256"
            --expected-calibration-file-sha256 "$POSTERIOR_V8_V5_EXPECTED_CALIBRATION_FILE_SHA256"
        )
        ;;
    *) echo "unsupported frozen-search protocol tier" >&2; exit 2 ;;
esac

relative_start=$((SLURM_ARRAY_TASK_ID * POSTERIOR_V8_V5_RECIPES_PER_SHARD))
if (( relative_start >= POSTERIOR_V8_V5_TOTAL_RECIPES )); then
    echo "array task starts beyond the audited requested recipe total" >&2
    exit 2
fi
remaining=$((POSTERIOR_V8_V5_TOTAL_RECIPES - relative_start))
shard_count=$POSTERIOR_V8_V5_RECIPES_PER_SHARD
if (( remaining < shard_count )); then
    shard_count=$remaining
fi
split_start=$((POSTERIOR_V8_V5_SPLIT_START + relative_start))
printf -v shard_name '%s-shard-%06d' \
    "$POSTERIOR_V8_V5_TARGET_SPLIT" "$SLURM_ARRAY_TASK_ID"
output_root="$POSTERIOR_V8_V5_SEARCH_OUTPUT_DIR/$shard_name"
view_indices=${POSTERIOR_V8_V5_VIEW_INDICES//:/,}
topology_ids=${POSTERIOR_V8_V5_SELECTED_TOPOLOGY_IDS//:/,}

cd "$POSTERIOR_V8_SOURCE_ROOT"
export PYTHONPATH="$POSTERIOR_V8_SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

conda run --no-capture-output -n tf \
    python -m utils.ML_Fitting_1D_GISAXS.PosteriorV8.run_frozen_search_pipeline_v5 \
    --split-plan "$POSTERIOR_V8_V5_SPLIT_PLAN" \
    --source-root "$POSTERIOR_V8_SOURCE_ROOT" \
    --expected-source-bundle-sha256 "$POSTERIOR_V8_V5_EXPECTED_SOURCE_BUNDLE_SHA256" \
    --launch-plan-sha256 "$POSTERIOR_V8_V5_LAUNCH_PLAN_SHA256" \
    --expected-split-plan-sha256 "$POSTERIOR_V8_V5_EXPECTED_SPLIT_PLAN_SHA256" \
    --expected-split-plan-file-sha256 "$POSTERIOR_V8_V5_EXPECTED_SPLIT_PLAN_FILE_SHA256" \
    --sobol-design "$POSTERIOR_V8_V5_SOBOL_DESIGN" \
    --expected-sobol-design-sha256 "$POSTERIOR_V8_V5_EXPECTED_SOBOL_DESIGN_SHA256" \
    --expected-sobol-design-file-sha256 "$POSTERIOR_V8_V5_EXPECTED_SOBOL_DESIGN_FILE_SHA256" \
    --target-split "$POSTERIOR_V8_V5_TARGET_SPLIT" \
    --start "$split_start" \
    --count "$shard_count" \
    --view-indices "$view_indices" \
    --topology-schedule-id "$POSTERIOR_V8_V5_TOPOLOGY_SCHEDULE_ID" \
    --selected-topology-ids "$topology_ids" \
    --local-sobol-schedule "$POSTERIOR_V8_V5_LOCAL_SOBOL_SCHEDULE" \
    --expected-local-schedule-sha256 "$POSTERIOR_V8_V5_EXPECTED_LOCAL_SCHEDULE_SHA256" \
    --expected-local-schedule-artifact-sha256 "$POSTERIOR_V8_V5_EXPECTED_LOCAL_SCHEDULE_ARTIFACT_SHA256" \
    --expected-local-schedule-manifest-sha256 "$POSTERIOR_V8_V5_EXPECTED_LOCAL_SCHEDULE_MANIFEST_SHA256" \
    --optimizer-schedule-id "$POSTERIOR_V8_V5_OPTIMIZER_SCHEDULE_ID" \
    --direct-scout-seed-count "$POSTERIOR_V8_V5_DIRECT_SCOUT_SEEDS" \
    --per-seed-forward-evaluation-limit "$POSTERIOR_V8_V5_PER_SEED_FORWARD_LIMIT" \
    --ftol "$POSTERIOR_V8_V5_FTOL" \
    --xtol "$POSTERIOR_V8_V5_XTOL" \
    --gtol "$POSTERIOR_V8_V5_GTOL" \
    --protocol-id "$POSTERIOR_V8_V5_PROTOCOL_ID" \
    --protocol-tier "$POSTERIOR_V8_V5_PROTOCOL_TIER" \
    "${protocol_args[@]}" \
    --output-root "$output_root" \
    --execute
