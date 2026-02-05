.PHONY: install test lint format check clean docker-build docker-run start stop

# Install dependencies
install:
	pip install --upgrade pip
	pip install -r requirements.txt
	pre-commit install

# Run tests
test:
	python -m pytest tests/ -v

# Lint code
lint:
	flake8 src/
	black --check src/
	isort --check-only src/

# Format code
format:
	black src/
	isort src/

# Run type checking
check:
	mypy src/

# Clean cache files
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/

# Build Docker image
docker-build:
	docker build -t mvanet-segmentation-api .

# Run Docker container
docker-run:
	docker run -p 8000:8000 mvanet-segmentation-api

# Start the application
start:
	uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000

# Stop is handled by Ctrl+C in terminal

# Run benchmarking
benchmark:
	python -m scripts.stress_test

# Run all checks
ci-check: lint check test