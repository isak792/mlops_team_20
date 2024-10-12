.PHONY: help setup update format lint test clean

help:
	@echo "Available commands:"
	@echo "  make setup    - Set up the development environment"
	@echo "  make update   - Update dependencies"
	@echo "  make format   - Run code formatters"
	@echo "  make lint     - Run linters"
	@echo "  make test     - Run tests"
	@echo "  make clean    - Remove build artifacts"

setup:
	pip install -r requirements.txt
	pre-commit install

update:
	pip install -U -r requirements.txt

format:
	black .
	isort .

lint:
	flake8 .

test:
	pytest

clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
