
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

    model = LlamaForCausalLM(config)
    model.gradient_checkpointing_enable()
    return model


def prepare_dataset(tokenizer, max_length=4096):
    dataset = load_dataset(
        "codeparrot/codeparrot-clean",
        split="train",
        streaming=True,
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
    )

    return tokenized_dataset


def main():
    tokenizer_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../models/base_tokenizer/code-bpe-tokenizer.json",
        )
    )
    output_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../models/phase1_base_750m")
    )

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    tokenizer.pad_token = "<|pad|>"

    model = build_model()

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
        trainer.train(resume_from_checkpoint=latest_checkpoint)
    else:
        trainer.train()

    final_model_path = os.path.join(output_dir, "final_model")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)


if __name__ == "__main__":
    main()
