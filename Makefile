.PHONY: install data test clean

install:
	pip install -r requirements.txt

data:
	python -m src.data.download
	python -m src.data.clean

test:
	pytest -v

clean:
	rm -rf data/interim/* data/processed/*
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache
