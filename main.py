from train_model import train_model
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import pandas as pd
import torch

test_dataset = train_model()

# Load baseline and fine-tuned models
baseline_model = GPT2LMHeadModel.from_pretrained("gpt2").to("cpu")
baseline_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
baseline_tokenizer.pad_token = baseline_tokenizer.eos_token

finetuned_model = GPT2LMHeadModel.from_pretrained("./def-gpt2").to("cpu")
finetuned_tokenizer = GPT2Tokenizer.from_pretrained("./def-gpt2")
finetuned_tokenizer.pad_token = finetuned_tokenizer.eos_token

# Create DataFrame
df = pd.DataFrame(columns=["input", "baseline", "fine_tuned"])

for row in test_dataset:
    question = row["text"].split("\n")[0]
    prompt = f"{question}\nA:"

    # Baseline model output
    baseline_input = baseline_tokenizer(prompt, return_tensors="pt").input_ids.to("cpu")
    baseline_output = baseline_model.generate(
        baseline_input,
        max_length=50,
        do_sample=False,
        temperature=0.3,
        pad_token_id=baseline_tokenizer.eos_token_id,
    )
    baseline_answer = baseline_tokenizer.decode(baseline_output[0], skip_special_tokens=True).split("A:")[-1].strip().split("\n")[0]

    # Finetuned model output
    finetuned_input = finetuned_tokenizer(prompt, return_tensors="pt").input_ids.to("cpu")
    finetuned_output = finetuned_model.generate(
        finetuned_input,
        max_length=50,
        do_sample=False,
        temperature=0.3,
        pad_token_id=finetuned_tokenizer.eos_token_id,
    )
    finetuned_answer = finetuned_tokenizer.decode(finetuned_output[0], skip_special_tokens=True).split("A:")[-1].strip().split("\n")[0]

    df.loc[len(df)] = {
        "input": prompt,
        "baseline": baseline_answer,
        "fine_tuned": finetuned_answer
    }

# Save to CSV
df.to_csv("gpt2_output.csv", index=False)