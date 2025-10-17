# FROM STATIC TO DYNAMIC: ADAPTIVE MONTECARLO SEARCH FOR MATHEMATICAL PROCESS SUPERVISION

This project is the official implementation of the paper **"FROM STATIC TO DYNAMIC: ADAPTIVE MONTE CARLO SEARCH FOR MATHEMATICAL PROCESS SUPERVISION"** .

Large scale language models (LLMs) still face challenges when dealing with complex multi-step mathematical reasoning problems. Process Reward Models (PRMs) have been proven to be an effective way to enhance the reasoning ability of models by supervising each step of the reasoning process. However, obtaining high-quality process supervision data is the main bottleneck for training PRM. Existing methods typically rely on fixed budget sampling strategies, which are inefficient and lack flexibility in large search spaces.

To address these issues, we propose the Adaptive Monte Carlo Search (AMCS) framework. AMCS fundamentally improves the process of generating process supervision data:
1.  **Uncertainty driven adaptive sampling:** AMCS can dynamically allocate more computing resources (samples) to inference steps with high uncertainty, while reducing sampling of simple and high certainty steps, thereby significantly improving the efficiency of data annotation.
2.  **Dynamic Exploration and Utilization Strategy:** AMCS uses Monte Carlo Tree Search (MCTS) to explore inference paths, which smoothly transitions from extensive exploration in the early stages to deep utilization in the later stages, thereby more effectively discovering high-quality inference paths and locating erroneous steps.

Based on AMCS, we constructed a high-quality process supervision dataset containing approximately 200000 samples and trained AMCS-PRM. Experimental results have shown that our method achieves the current best performance on multiple mathematical inference benchmarks such as MATH, AIME, Olympiad Bench, etc. This demonstrates the significant value of high-quality process supervision in enhancing model capabilities.

## Environment Configuration

2. Create a virtual environment:
   ```bash
   conda create -n amcs python=3.10
   conda activate amcs
   ```

3. Install the required dependency packages.
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Generation of MathSearch-200K Datasets

Use `adaptive_omegaprm/run_omegaprm.py`  to generate process supervision data.

```bash
# Run data generation on multiple GPUs
bash adaptive_omegaprm/run_omegaprm_multi_gpu.sh
```

### 2. Process Reward Model Training

Use `gen_rm/fine_tuning.py` to train a process reward model.

```
python gen_rm/fine_tuning.py \
--model_name_or_path "Qwen/Qwen2.5-Math-7B-Instruct" \ 
--train_file "path/to/your/generated_data.jsonl" \
--output_dir "./models/amcs-prm" \
--num_train_epochs 3 \
--per_device_train_batch_size 4 \
--gradient_accumulation_steps 8 \# ... other training parameters
```



### 3. Inference-time Verification

This section aims to use AMCS-PRM as a validator to improve the performance of generative models in mathematical problems through different search strategies. Use MCTS for evaluation. First, start the model service, and then run the evaluation script.

`bash scripts/eval/vanila_mcts.sh [ACTOR_MODEL] [PRM_MODEL] [DATASET] [OUTPUT_DIR]`

## Reference

If you find our work helpful for your research, please consider citing our paper: