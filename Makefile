# Set defaults
MODEL_DIR=models/fine-tuned-llama
APP_DIR=api
TEST_DIR=test
PYTHON=python

.PHONY: help train eval serve test api_call clean

help:
	@echo "Makefile commands:"
	@echo "  make run          - Start FastAPI server"
	@echo "  make test         - Run test edge cases script"
	@echo "  make api_call     - Make a sample API request"

install:
	python3 -m pip install -r requirements.txt

run:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

test:
	$(PYTHON) $(TEST_DIR)/test_edge_cases.py

api_call:
	curl -X POST http://localhost:8000/generate \
		-H "Content-Type: application/json" \
		-d '{"business_description": "AI for personalized fitness coaching"}'
