.PHONY: help install test quickstart all clean

help:
	@echo "Available commands:"
	@echo "  make install     Install dependencies and local package in editable mode"
	@echo "  make test        Run all unit tests via unittest discovery"
	@echo "  make quickstart  Run the quickstart 10-second simulation demo"
	@echo "  make all         Run the complete research reproducibility suite"
	@echo "  make clean       Remove temporary Python cache files and logs"

install:
	pip install -r requirements.txt
	pip install -e .

test:
	python -m unittest discover tests

quickstart:
	python examples/quickstart.py

all:
	python run_all_experiments.py --all

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
