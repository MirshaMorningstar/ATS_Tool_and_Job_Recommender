from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Step 1: Define model name and local directory
model_name = "meta-llama/Llama-2-7b-hf"  # Official LLaMA 2 7B model
local_model_dir = "./llama2-7b"  # Directory to save the model

# Step 2: Download and save the model locally
print("Downloading and saving the model locally...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",  # Automatically maps to GPU/CPU as available
    offload_folder="./offload",  # Offload layers to CPU if GPU memory is full
    torch_dtype=torch.float16,  # Use FP16 for faster performance on GPU
    low_cpu_mem_usage=True  # Memory optimization for loading
)
model.save_pretrained(local_model_dir)
tokenizer.save_pretrained(local_model_dir)
print("Model downloaded and saved locally!")

# Step 3: Load the saved model
print("Loading the saved model...")
tokenizer = AutoTokenizer.from_pretrained(local_model_dir)
model = AutoModelForCausalLM.from_pretrained(
    local_model_dir,
    device_map="auto",  # Use GPU if available; otherwise CPU
    offload_folder="./offload",  # Offload layers to CPU
    torch_dtype=torch.float16,  # Efficient precision for GPU
    low_cpu_mem_usage=True  # Optimize RAM usage
)

