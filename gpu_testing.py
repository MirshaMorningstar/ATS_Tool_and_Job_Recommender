import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login

# Authenticate with Hugging Face
HUGGINGFACE_TOKEN = "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"  # Replace with your token
login(HUGGINGFACE_TOKEN)

# Check CUDA availability
print("Is CUDA available?", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))

# Quantization configuration with CPU offloading
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True  # Enables CPU offloading
)

# Load model with CPU offloading and quantization
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto"
)

print("Model loaded successfully!")
print("Model device:", next(model.parameters()).device)

# Save the model locally
model.save_pretrained("./local_llama2_7b")
tokenizer.save_pretrained("./local_llama2_7b")

print("Model and tokenizer saved locally.")
