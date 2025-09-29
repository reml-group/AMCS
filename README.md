# FROM STATIC TO DYNAMIC: ADAPTIVE MONTECARLO SEARCH FOR MATHEMATICAL PROCESS SUPERVISION

This project is the official implementation of the paper **"FROM STATIC TO DYNAMIC: ADAPTIVE MONTE CARLO SEARCH FOR MATHEMATICAL PROCESS SUPERVISION"** .

We propose the **Adaptive Monte Carlo Search (AMCS**) framework. AMCS fundamentally improves the process of generating process supervision data.Based on AMCs, we constructed a high-quality process supervision dataset **MathSearch-200K** containing approximately 200000 samples and trained **AMCS-PRM**. This demonstrates the significant value of high-quality process supervision in enhancing model capabilities.

# Installation

1. Create a virtual environment:

   ```bash
   conda create -n amcs python=3.10
   conda activate amcs
   ```

2. Install the required dependency packages.

   ```bash
   pip install -r requirements.txt
   ```

# Usage

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

# Reference

If you find our work helpful for your research, please consider citing our paper:
