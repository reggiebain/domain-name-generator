import torch

def generate_domain(model, tokenizer, description, max_new_tokens=20):
    prompt = f"Business: {description} -> Domain:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text.split("->")[-1].strip()
