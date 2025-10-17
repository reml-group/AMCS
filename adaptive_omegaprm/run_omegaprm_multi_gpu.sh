#!/bin/bash



MODEL_NAME="path"
MODEL_TYPE="vllm"
DEVICE_INFO="cuda"
MAX_NEW_TOKENS=2048
TEMPERATURE=0.7
TOP_K=30
TOP_P=0.9
GPU_MEMORY_UTILIZATION=0.95

C_PUCT=0.125
ALPHA=0.5
BETA=0.9
L_Q_LEN_PENALTY=500
K_FIXED_ROLLOUTS=16
MAX_SEARCH_COUNT=20
MAX_ROLLOUT_BUDGET=150
SAVE_DATA_TREE=True

# The initial number of rollouts (k0) to perform when a new node is first evaluated.
K0_ADAPTIVE_MC=6
# The minimum number of additional rollouts to perform in a single adaptive step.
MIN_DYNAMIC_K_STEP=1
# The maximum number of additional rollouts to perform in a single adaptive step.
MAX_DYNAMIC_K_STEP=6
# A scaling factor that converts a cluster's confidence interval half-width (HW) into a number of new rollouts. (k_step ≈ HW * factor).
HW_SCALING_FACTOR=12.0

# --- AMC Clustering Settings ---
K_MIN_CLUSTER_MC=3
EPSILON_CLUSTER_MC=0.2
K_MAX_NODE_MC=20
NUM_CLUSTERS_K_MC=3
FEATURE_NAMES_MC="nll,log_length"
EPSILON_NODE_CONFIDENCE_STOP=0.2



NLL_FILTER_THRESHOLD=100.0
GARBLED_FILTER_LEVEL=3

USE_FILTER=False

NUM_FILTER_ROLLOUTS=32

MAX_QUESTIONS_PER_SPLIT=-1

NUM_GPUS_TO_RUN_ON=2

TARGET_SINGLE_GPU_ID=1

SPLIT_DIR="output_directory"


TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

RUN_TAG="k0-${K0_ADAPTIVE_MC}_k-dyn-${MIN_DYNAMIC_K_STEP}-${MAX_DYNAMIC_K_STEP}_hw-${HW_SCALING_FACTOR}_temp-${TEMPERATURE}"


MAIN_PROJECT_DIR="OmegaPRM_Runs"
RUN_OUTPUT_DIR="$MAIN_PROJECT_DIR/${TIMESTAMP}_${RUN_TAG}/results"
RUN_LOG_DIR="$MAIN_PROJECT_DIR/${TIMESTAMP}_${RUN_TAG}/logs"


mkdir -p "$RUN_OUTPUT_DIR"; mkdir -p "$RUN_LOG_DIR"


echo "Starting OmegaPRM batch processing with DYNAMIC K_STEP..."
echo "Run Tag: $RUN_TAG"
echo "Output will be saved in: $RUN_OUTPUT_DIR"
echo "Logs will be saved in: $RUN_LOG_DIR"
echo "Dynamic k_step params: Min=$MIN_DYNAMIC_K_STEP, Max=$MAX_DYNAMIC_K_STEP, HWScaleFactor=$HW_SCALING_FACTOR"
echo "Filter params: NLL_Threshold=$NLL_FILTER_THRESHOLD, Garbled_Level=$GARBLED_FILTER_LEVEL"

declare -a effective_gpu_ids_arr; declare -a task_slot_ids_arr

if [ -n "$CUSTOM_GPU_IDS" ]; then
    echo "Custom GPU mode: Targeting specific GPU IDs: $CUSTOM_GPU_IDS"
    effective_gpu_ids_arr=($CUSTOM_GPU_IDS)
    num_gpus=${#effective_gpu_ids_arr[@]}
    task_slot_ids_arr=($(seq 1 $num_gpus))
    echo "Task slots for file reading (e.g., part_X): ${task_slot_ids_arr[*]}"

elif [ "$NUM_GPUS_TO_RUN_ON" -eq 1 ]; then
    echo "Single GPU mode selected: Targeting specific GPU ID $TARGET_SINGLE_GPU_ID."
    effective_gpu_ids_arr=("$TARGET_SINGLE_GPU_ID")
    task_slot_ids_arr=(1)
    echo "Task slot for file reading (e.g., part_X): ${task_slot_ids_arr[0]}"
else
    echo "Multi-GPU mode: Using $NUM_GPUS_TO_RUN_ON GPUs (IDs 0 to $((NUM_GPUS_TO_RUN_ON-1)))."
    effective_gpu_ids_arr=($(seq 0 $((NUM_GPUS_TO_RUN_ON-1)) ))
    task_slot_ids_arr=($(seq 1 $NUM_GPUS_TO_RUN_ON))
fi


for (( enum_idx=0; enum_idx<${#task_slot_ids_arr[@]}; enum_idx++ )); do
    task_slot_for_file_naming=${task_slot_ids_arr[$enum_idx]}
    gpu_to_use=${effective_gpu_ids_arr[$enum_idx]}

    QUESTION_FILE_TO_PROCESS="$SPLIT_DIR/questions_part_${task_slot_for_file_naming}.json"

    OUTPUT_FILE_TARGET="$RUN_OUTPUT_DIR/results_gpu_${gpu_to_use}_part_${task_slot_for_file_naming}.jsonl"
    LOG_FILE_TARGET_PREFIX="$RUN_LOG_DIR/omegaprm_gpu_${gpu_to_use}_part_${task_slot_for_file_naming}"

    if [ ! -f "$QUESTION_FILE_TO_PROCESS" ]; then
        echo "Warning: File $QUESTION_FILE_TO_PROCESS DNE. Skipping GPU ${gpu_to_use}."
        continue
    fi

    echo "Launching on GPU $gpu_to_use: File $QUESTION_FILE_TO_PROCESS -> Output $OUTPUT_FILE_TARGET, LogPrefix $LOG_FILE_TARGET_PREFIX"

    COMMAND_TO_RUN="CUDA_VISIBLE_DEVICES=$gpu_to_use python3 run_omegaprm.py \
        --question_file \"$QUESTION_FILE_TO_PROCESS\" \
        --output_file_path \"$OUTPUT_FILE_TARGET\" \
        --log_file_prefix \"$LOG_FILE_TARGET_PREFIX\" \
        --model_name \"$MODEL_NAME\" --model_type \"$MODEL_TYPE\" --device \"$DEVICE_INFO\" \
        --max_new_tokens \"$MAX_NEW_TOKENS\" --temperature \"$TEMPERATURE\" --top_k \"$TOP_K\" --top_p \"$TOP_P\" \
        --gpu_memory_utilization \"$GPU_MEMORY_UTILIZATION\" \
        --c_puct \"$C_PUCT\" --alpha \"$ALPHA\" --beta \"$BETA\" --L_q_len_penalty \"$L_Q_LEN_PENALTY\" \
        --k_fixed_rollouts \"$K_FIXED_ROLLOUTS\" --max_search_count \"$MAX_SEARCH_COUNT\" \
        --max_rollout_budget \"$MAX_ROLLOUT_BUDGET\" --save_data_tree \"$SAVE_DATA_TREE\" \
        --k0_adaptive_mc \"$K0_ADAPTIVE_MC\" \
        --min_dynamic_k_step \"$MIN_DYNAMIC_K_STEP\" \
        --max_dynamic_k_step \"$MAX_DYNAMIC_K_STEP\" \
        --hw_to_k_step_scaling_factor \"$HW_SCALING_FACTOR\" \
        --k_min_cluster_mc \"$K_MIN_CLUSTER_MC\" --epsilon_cluster_mc \"$EPSILON_CLUSTER_MC\" \
        --k_max_node_mc \"$K_MAX_NODE_MC\" --num_clusters_k_mc \"$NUM_CLUSTERS_K_MC\" \
        --feature_names_mc \"$FEATURE_NAMES_MC\" \
        --epsilon_node_confidence_stop \"$EPSILON_NODE_CONFIDENCE_STOP\" \
        --nll_filter_threshold \"$NLL_FILTER_THRESHOLD\" \
        --garbled_text_filter_level \"$GARBLED_FILTER_LEVEL\" \
        --use_filter \"$USE_FILTER\" --num_filter_rollouts \"$NUM_FILTER_ROLLOUTS\" \
        --max_questions_to_process \"$MAX_QUESTIONS_PER_SPLIT\""

    echo "Executing: $COMMAND_TO_RUN"
    eval "$COMMAND_TO_RUN" &
done

wait

echo "All OmegaPRM processes launched. Monitor logs in $RUN_LOG_DIR and results in $RUN_OUTPUT_DIR."
