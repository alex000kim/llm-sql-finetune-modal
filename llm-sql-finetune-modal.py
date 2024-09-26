import modal
import os

app = modal.App("llm-sql-finetune")

vol = modal.Volume.from_name("llm-sql-finetune-volume", create_if_missing=True)

cpu_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "datasets",
    )
)

gpu_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.2.0-devel-ubuntu22.04", add_python="3.10"
    )
    .pip_install(
        "torch",
        "transformers",
        "datasets",
        "peft",
        "trl",
        # needed for flash-attn
        "ninja",
        "packaging",
        "wheel",
    )
    .apt_install("git")
    .pip_install("flash-attn", 
                 extra_options="--no-build-isolation")
)

@app.function(image=cpu_image,
              volumes={"/data": vol}, 
              cpu=1.0)
def setup_and_preprocess():
    from pathlib import Path
    from datasets import load_dataset, load_from_disk

    CACHE_DIR = Path("/data/llm-finetune")
    CACHE_DIR.mkdir(exist_ok=True)
    HF_DATASETS_CACHE = CACHE_DIR/'datasets'
    HF_DATASETS_CACHE.mkdir(exist_ok=True)
    OUTPUT_DIR = CACHE_DIR/'output_dir'
    OUTPUT_DIR.mkdir(exist_ok=True)

    train_dataset_path = CACHE_DIR / "train_dataset"
    test_dataset_path = CACHE_DIR / "test_dataset"

    if train_dataset_path.exists() and test_dataset_path.exists():
        print("Datasets already exist. Skipping download and preprocessing.")
        return CACHE_DIR

    # Load and preprocess dataset
    dataset = load_dataset("b-mc2/sql-create-context", 
                           split="train",
                           cache_dir=str(HF_DATASETS_CACHE))
    
    N = 10000  # get a subset of the dataset
    dataset = dataset.select(range(N))
    train_test = dataset.train_test_split(test_size=0.2)

    train_dataset = train_test["train"]
    test_dataset = train_test["test"]

    # Save datasets to volume
    train_dataset.save_to_disk(train_dataset_path)
    test_dataset.save_to_disk(test_dataset_path)

    vol.commit()
    return CACHE_DIR


@app.function(image=gpu_image, 
              volumes={"/data": vol}, 
              secrets=[modal.Secret.from_name("my-huggingface-secret")],
              gpu="A100",
              timeout=2*60*60 # 2 hours
              )
def train_model(cache_dir: str):
    from pathlib import Path
    from datasets import load_from_disk
    from peft import LoraConfig
    from trl import SFTTrainer
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

    CACHE_DIR = Path(cache_dir)
    OUTPUT_DIR = CACHE_DIR / 'output_dir'

    # Load datasets
    train_dataset = load_from_disk(dataset_path=str(CACHE_DIR / "train_dataset"))
    test_dataset = load_from_disk(dataset_path=str(CACHE_DIR / "test_dataset"))
    system_message = """
    You are a text to SQL query translator. Users will ask you questions in English and you will generate a SQL query based on the provided SCHEMA:
    ```
    {prompt}
    ```
    """

    def apply_chat_template(sample, tokenizer):
        messages = [
                {"role": "user", "content": f'{system_message.format(prompt=sample["context"])}\n{sample["question"]}'},
                {"role": "assistant", "content": sample["answer"]}
            ]
        return {'text': tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)}

    column_names = list(train_dataset.features)

    # Define training config
    training_config = {
        "bf16": True,
        "learning_rate": 2e-4,
        "num_train_epochs": 1,
        "max_steps": -1,
        "output_dir": str(OUTPUT_DIR),
        "per_device_train_batch_size": 4,
        "gradient_checkpointing": True,
    }

    peft_config = {
        "r": 256,
        "lora_alpha": 128,
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "target_modules": "all-linear",
    }

    train_conf = TrainingArguments(**training_config)
    peft_conf = LoraConfig(**peft_config)

    # Load model and tokenizer
    model_id = "microsoft/Phi-3-mini-4k-instruct"
    model = AutoModelForCausalLM.from_pretrained(model_id, 
                                                 token=os.environ['HF_TOKEN'],
                                                 trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.unk_token

    processed_train_dataset = train_dataset.map(
        apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer},
        num_proc=10,
        remove_columns=column_names,
    )

    processed_test_dataset = test_dataset.map(
        apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer},
        num_proc=10,
        remove_columns=column_names,
    )
    # Set up trainer
    trainer = SFTTrainer(
        model=model,
        args=train_conf,
        peft_config=peft_conf,
        train_dataset=processed_train_dataset,
        eval_dataset=processed_test_dataset,
        max_seq_length=2048,
        dataset_text_field="text",
        tokenizer=tokenizer,
        packing=True
    )

    # Train the model
    train_result = trainer.train()

    # Save the model
    trainer.save_model(OUTPUT_DIR / "final_model")

    vol.commit()
    return OUTPUT_DIR / "final_model"

@app.function(image=gpu_image, 
              volumes={"/data": vol}, 
              gpu="A10G",
              timeout=2*60*60 # 2 hours
              )
def evaluate_model(model_path: str, cache_dir: str):
    from pathlib import Path
    from datasets import load_from_disk
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    import torch
    from tqdm import tqdm

    CACHE_DIR = Path(cache_dir)
    test_dataset = load_from_disk(dataset_path=str(CACHE_DIR / "test_dataset"))

    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Create a text-generation pipeline
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device="cuda")

    def prepare_input(sample):
        system_message = "You are an text to SQL query translator. Users will ask you questions in English and you will generate a SQL query based on the provided SCHEMA:\n```\n{prompt}\n```"
        messages = [
            {"role": "user", "content": f'{system_message.format(prompt=sample["context"])}\n{sample["question"]}'}
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    number_of_eval_samples = 100
    eval_dataset = test_dataset.shuffle().select(range(number_of_eval_samples))
    
    inputs = [prepare_input(sample) for sample in eval_dataset]
    targets = [sample['answer'].lower() for sample in eval_dataset]

    batch_size = 8  # Adjust based on your GPU memory
    success_rate = []

    # Use batch generation
    for i in tqdm(range(0, len(inputs), batch_size)):
        batch_inputs = inputs[i:i+batch_size]
        batch_targets = targets[i:i+batch_size]
        
        outputs = pipe(batch_inputs, 
                            max_new_tokens=512, 
                            do_sample=True, 
                            temperature=0.1, 
                            top_k=50, 
                            top_p=0.95,
                            num_return_sequences=1,
                            batch_size=batch_size)
        
        for output, input_text, target in zip(outputs, batch_inputs, batch_targets):
            predicted_answer = output[0]['generated_text'][len(input_text):].strip().lower()
            print("-"*100)
            print(f"Input:\n{input_text}")
            print(f"Predicted:\n{predicted_answer}") 
            print(f"Target:\n{target}")
            print("-"*100)
            success_rate.append(predicted_answer == target)

    accuracy = sum(success_rate) / len(success_rate)
    print(f"Accuracy: {accuracy*100:.2f}%")

    return accuracy


@app.local_entrypoint()
def main():
    cache_dir = setup_and_preprocess.remote()
    model_path = train_model.remote(cache_dir)
    accuracy = evaluate_model.remote(model_path, cache_dir)
    print(f"Final model accuracy: {accuracy*100:.2f}%")
