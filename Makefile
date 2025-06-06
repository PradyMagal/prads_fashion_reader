.PHONY: install install-dev install-full test lint format clean train evaluate serve help bumpversion-patch bumpversion-minor bumpversion-major

# Installation commands
install:
	poetry install

install-dev:
	poetry install --with dev

install-full:
	poetry install -E full

install-text:
	poetry install -E text

install-search:
	poetry install -E search

# Development commands
test:
	poetry run pytest tests/ -v

lint:
	poetry run flake8 src/ tests/
	poetry run black --check src/ tests/

format:
	poetry run black src/ tests/
	poetry run isort src/ tests/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +

# Model training and evaluation
train:
	poetry run python -m imagetodescribe.train

evaluate:
	poetry run python -m imagetodescribe.evaluate

# API server
serve:
	poetry run uvicorn src.imagetodescribe.rest_main:app --host 0.0.0.0 --port 8000 --reload

# Version bumping
bumpversion-patch:
	poetry version patch
	git add pyproject.toml
	git commit -m "Bump version to $$(poetry version -s)"
	git tag "v$$(poetry version -s)"

bumpversion-minor:
	poetry version minor
	git add pyproject.toml
	git commit -m "Bump version to $$(poetry version -s)"
	git tag "v$$(poetry version -s)"

bumpversion-major:
	poetry version major
	git add pyproject.toml
	git commit -m "Bump version to $$(poetry version -s)"
	git tag "v$$(poetry version -s)"

# Utility commands
shell:
	poetry shell

show-deps:
	poetry show --tree

update-deps:
	poetry update

build:
	poetry build

publish:
	poetry publish

help:
	@echo "Available commands:"
	@echo "  make install          - Install basic dependencies"
	@echo "  make install-dev      - Install with dev dependencies"
	@echo "  make install-full     - Install with all optional dependencies"
	@echo "  make install-text     - Install with transformers for text processing"
	@echo "  make install-search   - Install with FAISS for similarity search"
	@echo ""
	@echo "  make test            - Run tests"
	@echo "  make lint            - Run linting checks"
	@echo "  make format          - Format code with black and isort"
	@echo "  make clean           - Clean up cache files and build artifacts"
	@echo ""
	@echo "  make train           - Train the fashion classification model"
	@echo "  make evaluate        - Evaluate the trained model"
	@echo "  make serve           - Start the FastAPI server"
	@echo ""
	@echo "  make bumpversion-patch - Bump patch version (0.1.0 -> 0.1.1)"
	@echo "  make bumpversion-minor - Bump minor version (0.1.0 -> 0.2.0)"
	@echo "  make bumpversion-major - Bump major version (0.1.0 -> 1.0.0)"
	@echo ""
	@echo "  make shell           - Open a poetry shell"
	@echo "  make show-deps       - Show dependency tree"
	@echo "  make update-deps     - Update all dependencies"
	@echo "  make build           - Build the package"
	@echo "  make publish         - Publish to PyPI"
	@echo "  make help            - Show this help message"
