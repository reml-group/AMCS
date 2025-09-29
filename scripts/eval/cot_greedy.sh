python reason/evaluation/evaluate.py \
    --LM GLM-4-9B-0414 \
    --task_name bench \
    --temperature 0.0 \
    --max_new_tokens 2048 \
    --save_dir cot_greedy \
    --method cot \
    --num_worker 32 \
    --controller_addr http://0.0.0.0:28777