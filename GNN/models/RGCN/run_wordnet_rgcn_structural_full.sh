#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-python}}"
DEVICE="${DEVICE:-cuda:0}"
AUDIT_DEVICE="${AUDIT_DEVICE:-cpu}"
SEEDS="${SEEDS:-1566911444 20241017 20251017}"
VARIANTS="${VARIANTS:-no_changes all_inverse_edges universal_edges}"
BUDGETS_TO_RUN="${BUDGETS_TO_RUN:-reduced full}"
MODES_TO_RUN="${MODES_TO_RUN:-intersection}"

RUN_TESTS="${RUN_TESTS:-1}"
PREPARE_SPLITS="${PREPARE_SPLITS:-auto}"
PREPROCESS="${PREPROCESS:-1}"
VALIDATE="${VALIDATE:-1}"
TRAIN="${TRAIN:-1}"
AGGREGATE="${AGGREGATE:-1}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
OVERWRITE_PREPROCESS="${OVERWRITE_PREPROCESS:-0}"

INV_GNN_DATA="${INV_GNN_DATA:-${SCRIPT_DIR}/data}"
DATASET_ROOT="${DATASET_ROOT:-${INV_GNN_DATA}/wordnet_3hops_augmented_full_fixed}"
if [[ ! -d "${DATASET_ROOT}" && -d "${INV_GNN_DATA}/wordnet_3hops_augmented_full" ]]; then
    DATASET_ROOT="${INV_GNN_DATA}/wordnet_3hops_augmented_full"
fi
SPLITS_PATH="${SPLITS_PATH:-${DATASET_ROOT}/wordnet_splits.npz}"
CATALOG_ROOT="${CATALOG_ROOT:-${SCRIPT_DIR}/wordnet_structural_catalogs}"
PREPROCESSED_ROOT="${PREPROCESSED_ROOT:-${INV_GNN_DATA}/preprocessed/RGCN_structural/WORDNET/universal_parity}"
RESULTS_ROOT="${RESULTS_ROOT:-${SCRIPT_DIR}/results/RGCN_structural/WORDNET/universal_parity}"
LOG_ROOT="${LOG_ROOT:-${RESULTS_ROOT}/logs}"

MAX_ENDPOINTS_PER_SOURCE="${MAX_ENDPOINTS_PER_SOURCE:-0}"
SAMPLING_SEED="${SAMPLING_SEED:-1566911444}"
SHORTCUT_AUDIT="${SHORTCUT_AUDIT:-full}"
INCLUDE_BASE_RELATIONS="${INCLUDE_BASE_RELATIONS:-1}"
TRAIN_OBJECTIVE="${TRAIN_OBJECTIVE:-semantic_all}"
RUN_PARITY_DIAGNOSTIC="${RUN_PARITY_DIAGNOSTIC:-1}"

HIDDEN_DIM="${HIDDEN_DIM:-200}"
NUM_BASES="${NUM_BASES:-30}"
EDGE_DROPOUT_OTHER="${EDGE_DROPOUT_OTHER:-0.4}"
ROOT_DROPOUT_LOOP="${ROOT_DROPOUT_LOOP:-0.2}"
LR="${LR:-0.01}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
NEG_PER_POS="${NEG_PER_POS:-1}"
BATCH_SIZE="${BATCH_SIZE:-0}"
EPOCHS="${EPOCHS:-3000}"
PATIENCE="${PATIENCE:-3000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-10}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
BINARY_K="${BINARY_K:-50}"
CANDIDATE_SEED="${CANDIDATE_SEED:-1566911444}"
EDGE_CHUNK_SIZE="${EDGE_CHUNK_SIZE:-200000}"
RESUME_MODE="${RESUME_MODE:-auto}"
ADDITIONAL_EPOCHS="${ADDITIONAL_EPOCHS:-0}"
RESET_PATIENCE_ON_RESUME="${RESET_PATIENCE_ON_RESUME:-0}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-1}"

mkdir -p "${PREPROCESSED_ROOT}" "${RESULTS_ROOT}" "${LOG_ROOT}"

run_logged() {
    local log_path="$1"
    shift
    mkdir -p "$(dirname "${log_path}")"
    printf '\n+ '
    printf '%q ' "$@"
    printf '\n'
    set +e
    "$@" 2>&1 | tee "${log_path}"
    local status=${PIPESTATUS[0]}
    set -e
    if [[ ${status} -ne 0 ]]; then
        echo >&2 "ERROR: command failed with status ${status}."
        echo >&2 "  command: $*"
        exit "${status}"
    fi
}

if [[ ! -x "${PYTHON_BIN}" ]] && ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo >&2 "Python executable not found: ${PYTHON_BIN}"
    exit 2
fi
if [[ ! -d "${DATASET_ROOT}" ]]; then
    echo >&2 "WordNet dataset root not found: ${DATASET_ROOT}"
    exit 2
fi
if [[ ! -f "${CATALOG_ROOT}/reduced_selection.json" || ! -f "${CATALOG_ROOT}/full_selection.json" ]]; then
    echo >&2 "Missing reduced/full catalogs under ${CATALOG_ROOT}"
    exit 2
fi

cat <<EOF
============================================================
WordNet conditional RGCN link-prediction suite
script root:       ${SCRIPT_DIR}
python:            ${PYTHON_BIN}
device:            ${DEVICE}
audit device:      ${AUDIT_DEVICE}
dataset root:      ${DATASET_ROOT}
splits:            ${SPLITS_PATH}
catalogs:          ${BUDGETS_TO_RUN}
modes:             ${MODES_TO_RUN}
variants:          ${VARIANTS}
seeds:             ${SEEDS}
preprocessed root: ${PREPROCESSED_ROOT}
results root:      ${RESULTS_ROOT}
endpoint cap:      ${MAX_ENDPOINTS_PER_SOURCE}
train objective:   ${TRAIN_OBJECTIVE}
parity diagnostic: ${RUN_PARITY_DIAGNOSTIC}
resume:            ${RESUME_MODE}
additional epochs: ${ADDITIONAL_EPOCHS}
============================================================
EOF

if [[ "${RUN_TESTS}" == "1" ]]; then
    run_logged "${LOG_ROOT}/00_wordnet_unit_tests.log" \
        "${PYTHON_BIN}" -m unittest discover \
        -s "${SCRIPT_DIR}/tests" -p 'test_wordnet_rgcn_structural.py' -v
fi

if [[ "${PREPARE_SPLITS}" == "1" || ( "${PREPARE_SPLITS}" == "auto" && ! -f "${SPLITS_PATH}" ) ]]; then
    prepare=(
        "${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_WORDNET_rgcn_structural_splits.py"
        --data-root "${DATASET_ROOT}"
        --output "${SPLITS_PATH}"
    )
    [[ -f "${SPLITS_PATH}" ]] && prepare+=(--overwrite)
    run_logged "${LOG_ROOT}/01_prepare_splits.log" "${prepare[@]}"
fi
if [[ ! -f "${SPLITS_PATH}" ]]; then
    echo >&2 "Missing split archive: ${SPLITS_PATH}"
    exit 2
fi

for budget in ${BUDGETS_TO_RUN}; do
    for mode in ${MODES_TO_RUN}; do
        config_root="${PREPROCESSED_ROOT}/${budget}/${mode}"
        all_present=1
        for variant in ${VARIANTS}; do
            [[ -f "${config_root}/${variant}/graph.pt" ]] || all_present=0
        done
        if [[ "${PREPROCESS}" == "1" ]]; then
            if [[ ${all_present} -eq 1 && "${OVERWRITE_PREPROCESS}" != "1" ]]; then
                echo "Skipping existing preprocessing: ${budget}/${mode}"
            else
                cmd=(
                    "${PYTHON_BIN}" "${SCRIPT_DIR}/preprocess_WORDNET_rgcn_structural.py"
                    --splits-path "${SPLITS_PATH}"
                    --catalog-root "${CATALOG_ROOT}"
                    --output-root "${PREPROCESSED_ROOT}"
                    --budget "${budget}"
                    --mode "${mode}"
                    --variants "${VARIANTS// /,}"
                    --max-endpoints-per-source "${MAX_ENDPOINTS_PER_SOURCE}"
                    --sampling-seed "${SAMPLING_SEED}"
                    --include-base-relations "${INCLUDE_BASE_RELATIONS}"
                    --shortcut-audit "${SHORTCUT_AUDIT}"
                )
                [[ "${OVERWRITE_PREPROCESS}" == "1" ]] && cmd+=(--overwrite)
                run_logged "${LOG_ROOT}/preprocess_${budget}_${mode}.log" "${cmd[@]}"
            fi
        fi

        if [[ "${VALIDATE}" == "1" ]]; then
            run_logged "${LOG_ROOT}/audit_${budget}_${mode}.log" \
                "${PYTHON_BIN}" "${SCRIPT_DIR}/validate_WORDNET_rgcn_structural.py" \
                --data-root "${PREPROCESSED_ROOT}" \
                --budget "${budget}" \
                --mode "${mode}" \
                --variants "${VARIANTS// /,}" \
                --seed "${SAMPLING_SEED}" \
                --device "${AUDIT_DEVICE}" \
                --hidden-dim "${HIDDEN_DIM}" \
                --num-bases "${NUM_BASES}" \
                --edge-dropout-other "${EDGE_DROPOUT_OTHER}" \
                --root-dropout-loop "${ROOT_DROPOUT_LOOP}" \
                --lr "${LR}" \
                --weight-decay "${WEIGHT_DECAY}" \
                --neg-per-pos "${NEG_PER_POS}" \
                --batch-size "${BATCH_SIZE}" \
                --edge-chunk-size "${EDGE_CHUNK_SIZE}" \
                --train-objective "${TRAIN_OBJECTIVE}" \
                --report "${RESULTS_ROOT}/audit_${budget}_${mode}.json"
        fi

        if [[ "${RUN_PARITY_DIAGNOSTIC}" == "1" && "${budget}" == "full" && "${mode}" == "intersection" ]]; then
            run_logged "${LOG_ROOT}/parity_${budget}_${mode}.log" \
                "${PYTHON_BIN}" "${SCRIPT_DIR}/diagnose_WORDNET_rgcn_universal_parity.py" \
                --graph "${PREPROCESSED_ROOT}/${budget}/${mode}/universal_edges/graph.pt" \
                --selection "${CATALOG_ROOT}/${budget}_selection.json" \
                --report "${RESULTS_ROOT}/parity_${budget}_${mode}.json" \
                --require-exact
        fi

        if [[ "${TRAIN}" == "1" ]]; then
            for seed in ${SEEDS}; do
                for variant in ${VARIANTS}; do
                    run_dir="${RESULTS_ROOT}/${budget}/${mode}/${variant}/seed_${seed}"
                    if [[ "${SKIP_COMPLETED}" == "1" && -s "${run_dir}/metrics.json" ]]; then
                        echo "Skipping completed run: ${budget}/${mode}/${variant}/seed_${seed}"
                        continue
                    fi
                    train_cmd=(
                        "${PYTHON_BIN}" "${SCRIPT_DIR}/run_WORDNET_rgcn_structural.py"
                        --data-root "${PREPROCESSED_ROOT}"
                        --output-root "${RESULTS_ROOT}"
                        --budget "${budget}"
                        --mode "${mode}"
                        --variant "${variant}"
                        --seed "${seed}"
                        --device "${DEVICE}"
                        --hidden-dim "${HIDDEN_DIM}"
                        --num-bases "${NUM_BASES}"
                        --edge-dropout-other "${EDGE_DROPOUT_OTHER}"
                        --root-dropout-loop "${ROOT_DROPOUT_LOOP}"
                        --lr "${LR}"
                        --weight-decay "${WEIGHT_DECAY}"
                        --neg-per-pos "${NEG_PER_POS}"
                        --batch-size "${BATCH_SIZE}"
                        --epochs "${EPOCHS}"
                        --patience "${PATIENCE}"
                        --eval-interval "${EVAL_INTERVAL}"
                        --eval-batch-size "${EVAL_BATCH_SIZE}"
                        --binary-k "${BINARY_K}"
                        --candidate-seed "${CANDIDATE_SEED}"
                        --edge-chunk-size "${EDGE_CHUNK_SIZE}"
                        --checkpoint-every "${CHECKPOINT_EVERY}"
                        --train-objective "${TRAIN_OBJECTIVE}"
                        --resume "${RESUME_MODE}"
                    )
                    if [[ "${ADDITIONAL_EPOCHS}" -gt 0 ]]; then
                        train_cmd+=(--additional-epochs "${ADDITIONAL_EPOCHS}")
                    fi
                    if [[ "${RESET_PATIENCE_ON_RESUME}" == "1" ]]; then
                        train_cmd+=(--reset-patience-on-resume)
                    fi
                    run_logged \
                        "${LOG_ROOT}/train_${budget}_${mode}_${variant}_seed_${seed}.log" \
                        "${train_cmd[@]}"
                done
            done
        fi
    done
done

if [[ "${AGGREGATE}" == "1" ]]; then
    run_logged "${LOG_ROOT}/aggregate.log" \
        "${PYTHON_BIN}" "${SCRIPT_DIR}/aggregate_WORDNET_rgcn_structural.py" \
        --results-root "${RESULTS_ROOT}" \
        --output-dir "${RESULTS_ROOT}/aggregate"
fi

echo "WordNet conditional RGCN suite complete."
