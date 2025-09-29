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
    \
    # --temperature: The sampling temperature for the Language Model to control randomness.
    --temperature 0.7 \
    \
    # --num_sequence: The number of candidate solutions (N in Best-of-N) to generate for each problem.
    --num_sequence 4 \
    \
    # --max_new_tokens: The maximum length of each generated solution.
    --max_new_tokens 2048 \
    \
    # --save_dir: The directory where the evaluation results and outputs will be saved.
    --save_dir best_of_n \
    \
    # --method: The evaluation methodology being used, explicitly set to "best_of_n".
    --method best_of_n \
    \
    # --num_worker: The number of parallel workers to use for the evaluation process.
    --num_worker 60 \
    \
    # --controller_addr: The network address for a controller, likely used to coordinate the distributed workers.
    --controller_addr http://0.0.0.0:28777


