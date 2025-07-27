from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model_and_tokenizer():
    model_path = 'models/fine-tuned-llama-domain-generator'
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer
