# Set defaults
MODEL_DIR=models/fine-tuned-llama
APP_DIR=app
TEST_DIR=test
PYTHON=python

.PHONY: help train eval serve test api_call clean

help:
	@echo "Makefile commands:"
	@echo "  make train        - Run model training script"
	@echo "  make eval         - Run evaluation script (metrics + LLM-as-a-judge)"
	@echo "  make serve        - Start FastAPI server"
	@echo "  make test         - Run test edge cases script"
	@echo "  make api_call     - Make a sample API request"
	@echo "  make clean        - Remove logs and outputs"

train:
	$(PYTHON) train_model.py

eval:
	$(PYTHON) evaluate_model.py

serve:
	uvicorn $(APP_DIR).main:app --reload

test:
	$(PYTHON) $(TEST_DIR)/test_edge_cases.py

api_call:
	curl -X POST http://localhost:8000/generate \
		-H "Content-Type: application/json" \
		-d '{"business_description": "AI for personalized fitness coaching"}'

clean:
	rm -rf outputs/ logs/ __pycache__/
