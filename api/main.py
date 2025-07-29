from fastapi import FastAPI
from pydantic import BaseModel
from api.model import load_model_and_tokenizer
from api.inference import generate_domain

app = FastAPI()

# Load model/tokenizer
model, tokenizer = load_model_and_tokenizer()

class DomainRequest(BaseModel):
    business_description: str

@app.post("/generate")
def generate(request: DomainRequest):
    output = generate_domain(model, tokenizer, request.business_description)
    return {"domain_name": output}
