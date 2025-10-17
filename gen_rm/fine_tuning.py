from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model

def preprocess_function(examples, tokenizer):
    inputs = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=1024)
    
    inputs["labels"] = inputs["input_ids"].copy()
    return inputs

def train_genrm(model_name, tokenizer, output_dir):

    dataset_path = "./dataset/organized_Llama-3.1-8B-Instruct_prm800k_train.json"
    dataset = load_dataset("json", data_files=dataset_path, split="train[0:1000]")

    standard_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        cache_dir="/shares/mxq6904/models",
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    model = get_peft_model(standard_model, lora_config)

    # Set up the data collator.
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        # The directory where model checkpoints and logs will be saved.
        output_dir=output_dir,
        # The total number of training epochs.
        num_train_epochs=1,
        # The batch size for training on each device (GPU).
        per_device_train_batch_size=1,
        logging_dir="./logs",
        logging_steps=10,
        save_steps=100,
        save_total_limit=5,
        fp16=True,  
    )

    # Apply the preprocessing function to the entire dataset.
    dataset = dataset.map(lambda examples: preprocess_function(examples, tokenizer), batched=True, num_proc=4, remove_columns=dataset.column_names)

    # Initialize the Trainer.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    # Start the training process.
    trainer.train()

if __name__ == "__main__":
    model_name = "meta-llama/Llama-3.1-8B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
    )
    tokenizer.pad_token = tokenizer.eos_token  

    train_genrm(model_name, tokenizer, "./output")
