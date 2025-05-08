from datasets import load_dataset
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
)
import torch

def train_model():

    # 1. Load dataset
    dataset = load_dataset("json", data_files={"data": "econ_qa.jsonl"})["data"]

    # 2. Split dataset into train/val/test
    data = dataset.train_test_split(test_size=0.2, seed=42)
    valid_split = data["test"].train_test_split(test_size=0.5, seed=42)
    
    train_dataset = data["train"] # 80% of dataset
    valid_dataset = valid_split["train"] # 10% of dataset
    test_dataset = valid_split["test"] # 10% of dataset

    # 3. Load model and tokenizer
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # 4. Tokenize dataset
    def tokenize(example):
        return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

    train_dataset = train_dataset.map(tokenize)
    valid_dataset = valid_dataset.map(tokenize)
    test_dataset = test_dataset.map(tokenize)

    # 5. Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 6. Define training arguments and trainer
    training_args = TrainingArguments(
        output_dir="./def-gpt2",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        learning_rate=5e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_total_limit=2,
        logging_steps=100,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # 7. Train the model
    trainer.train()

    # 8. Save the model and tokenizer
    model.save_pretrained("./def-gpt2")
    tokenizer.save_pretrained("./def-gpt2")

    # 9. Return the test dataset
    return test_dataset