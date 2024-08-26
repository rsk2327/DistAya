import logging
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer, SFTConfig
from accelerate import PartialState
import torch

from huggingface_hub import login

login('xxxxxxxxxxxxxxxxxxxxxxxxx')

logging.basicConfig(level=logging.INFO)

model_name = "Qwen/Qwen2-0.5B"
dataset_name = "AhmadMustafa/dsftaya"

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['prompt'])):
        text = f"""<|im_start|> system
        You are DistAYA, a helpful small multlingual model.<|im_start|>
        <|im_start|>user
        {example['prompt'][i]}<|im_end|>
        <|im_start|>assistant
        {example['completion'][i]}<|im_end|>"""
        output_texts.append(text)
    return output_texts

def main():
    state = PartialState()

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    print(f"EOS Token: {tokenizer.eos_token}")
    print(f"Pad Token: {tokenizer.pad_token}")

    dataset = load_dataset(dataset_name)
    dataset = dataset.rename_column("INSTRUCTION", "prompt")
    dataset = dataset.rename_column("RESPONSE", "completion")
    if "dataset" in dataset.column_names:
        dataset = dataset.remove_columns("dataset")

    train_dataset = dataset["train"]

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map={'': state.process_index},
        torch_dtype=torch.bfloat16
    )

    training_args = TrainingArguments(
        output_dir='/home/ahmadanis/distaya/model',
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,
        learning_rate=2e-5,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="no",  # Changed from evaluation_strategy to eval_strategy
        fp16=True,
        ddp_find_unused_parameters=False,
        fsdp="full_shard auto_wrap",
        fsdp_config={
            "fsdp_offload_params": "auto",
            "fsdp_state_dict_type": "FULL_STATE_DICT",
            "fsdp_transformer_layer_cls_to_wrap": "Qwen2Model",
            "activation_checkpointing": True,  # Added activation checkpointing here
        },
        push_to_hub=True,
    )

    sft_config = SFTConfig(
        output_dir='/home/ahmadanis/distaya/model_SFTConfig',
        max_seq_length=2048,  # Moved max_seq_length to SFTConfig
        push_to_hub=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        formatting_func=formatting_prompts_func,
        tokenizer=tokenizer,
        max_seq_length=2048
        # config=sft_config,  # Added SFTConfig here
    )

    trainer.train()
    
    if state.is_main_process:
        trainer.save_model('/home/ahmadanis/distaya/model')

if __name__ == "__main__":
    main()
