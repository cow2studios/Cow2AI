import os
import glob
import torch
from datasets import load_dataset
from transformers import (
    PreTrainedTokenizerFast,
    LlamaConfig,
    LlamaForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)


def build_model():
    config = LlamaConfig(
        vocab_size=32000,
        hidden_size=1536,
        intermediate_size=4096,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_key_value_heads=4,
        max_position_embeddings=8192,
        bos_token_id=0,
        eos_token_id=2,
        pad_token_id=1,
    )

    print("Initializing model architecture...")
    model = LlamaForCausalLM(config)

    model.gradient_checkpointing_enable()
    return model


def prepare_dataset(tokenizer, max_length):
    dataset = load_dataset(
        "bigcode/the-stack-smol",
        split="train",
        data_dir="data/python",
        streaming=True,
        token=True,
    )

    def tokenize_function(examples):
        return tokenizer(
            examples["content"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["content", "repository_name", "path", "size", "license"],
    )
    return tokenized_dataset


def main():
    tokenizer_path = os.path.join(
        os.path.dirname(__file__), "../models/base_tokenizer/code-bpe-tokenizer.json"
    )
    output_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../models/base_750m")
    )

    print("Loading custom BPE tokenizer...")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    tokenizer.pad_token = "<|pad|>"

    model = build_model()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Model Parameters: {total_params / 1e6:.2f}M")

    train_dataset = prepare_dataset(tokenizer=tokenizer, max_length=4096)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        max_steps=10000,
        learning_rate=3e-4,
        weight_decay=0.1,
        bf16=True,
        optim="adamw_8bit",
        logging_steps=50,
        save_steps=500,
        max_grad_norm=1.0,
        dataloader_num_workers=0,
        report_to="none",
        ignore_data_skip=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))

    if checkpoints:
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[-1]))
        print(f"Resuming training from {latest_checkpoint}...")
        trainer.train(resume_from_checkpoint=latest_checkpoint)
    else:
        print("Starting fresh pre-training run...")
        trainer.train()

    print("Training complete. Saving final base model...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
