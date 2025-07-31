from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM
from peft import PeftModel, PeftConfig
import os
from dotenv import load_dotenv
import torch

def load_model_and_tokenizer():

    adapter_path = "models/fine-tune-llama-lora"

    load_dotenv()  # Loads variables from .env into environment

    hf_token = os.getenv("HF_TOKEN")

    base_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-1B-Instruct",
        trust_remote_code=True,
        use_auth_token=hf_token,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.2-1B-Instruct", 
        trust_remote_code=True,
        use_auth_token=hf_token)
    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, adapter_path)

    return model, tokenizer
