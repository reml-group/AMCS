# using a "Best-of-N" reranking strategy with a Reward Model.

python reason/evaluation/evaluate.py \
    # --LM: Specifies the Language Model (Generator) to be used for generating solutions.
    --LM Qwen3-8B \
    \
    # --RM: Specifies the Reward Model (Reranker) used to score and select the best solution.
    --RM Qwen2.5-Math-7B-PRM800K \
    \
    # --task_name: The name of the evaluation dataset. AIME is a well-known math competition.
    --task_name AIME \
    --temperature 0.7 \
    --num_sequence 4 \
    --max_new_tokens 2048 \
    --save_dir best_of_n \
    \
    # --method: The evaluation methodology being used, explicitly set to "best_of_n".
    --method best_of_n \
    --num_worker 60 \
    --controller_addr http://0.0.0.0:28777


