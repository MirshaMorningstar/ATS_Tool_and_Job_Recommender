from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

model_name = "meta-llama/Llama-2-7b-hf"
local_model_dir = "./llama2-7b"

# Load tokenizer and model directly on the GPU
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # Use FP16 to reduce memory usage and increase speed
    device_map="auto",  # This ensures the model is mapped across available devices (GPU or CPU)
)

# Save model locally
model.save_pretrained(local_model_dir)
tokenizer.save_pretrained(local_model_dir)

# Set up the pipeline for text generation
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0,  # Ensures the model uses the first GPU (if available)
)

# Test the pipeline
prompt = "Write a poem about the beauty of nature."
result = generator(prompt, max_length=100, num_return_sequences=1)
print(result)
