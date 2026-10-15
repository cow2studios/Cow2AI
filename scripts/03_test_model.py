import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

model_path = "models\\phase1_base_750m\\checkpoint-6000"

print("Loading trained 750M base model...")
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path, dtype=torch.bfloat16, device_map="cuda"
)

prompt = """def binary_search(arr, target):"""
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

print(f"\nPrompt: {prompt}\nGenerating output...\n")

with torch.no_grad():
    output_tokens = model.generate(
        **inputs,
        max_new_tokens=128,
        temperature=0.2,
        top_p=0.95,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
    )

generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

clean_text = generated_text.replace("Ġ", " ").replace("Ċ", "\n")

print("--- Generated Code ---")
print(clean_text)
