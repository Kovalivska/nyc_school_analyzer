# NYC School Analyzer - Production Build System
# Comprehensive Makefile for development, testing, and deployment

.PHONY: help install install-dev clean test test-fast test-coverage lint format type-check \
        build package docker-build docker-run docker-clean docs serve-docs \
        security-check pre-commit setup-dev release deploy

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := python3
PIP := pip3
PACKAGE_NAME := nyc-school-analyzer
SRC_DIR := src/nyc_school_analyzer
TEST_DIR := tests
DOCS_DIR := docs
BUILD_DIR := build
DIST_DIR := dist

# Docker variables
DOCKER_IMAGE := nyc-school-analyzer
DOCKER_TAG := latest
DOCKER_DEV_TAG := dev

# Version detection
VERSION := $(shell python setup.py --version 2>/dev/null || echo "unknown")

## Display help message
help:
	@echo "NYC School Analyzer - Production Build System"
	@echo "============================================="
	@echo
	@echo "Available commands:"
	@echo
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo
	@echo "Environment:"
	@echo "  Python: $(shell $(PYTHON) --version 2>/dev/null || echo 'Not found')"
	@echo "  Package: $(PACKAGE_NAME)"
	@echo "  Version: $(VERSION)"

## Install package in production mode
install:
	@echo "Installing $(PACKAGE_NAME) in production mode..."
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -e .
	@echo "✅ Installation completed"

## Install package in development mode with all dependencies
install-dev:
	@echo "Installing $(PACKAGE_NAME) in development mode..."
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -e ".[dev,docs,test]"
	pre-commit install
	@echo "✅ Development installation completed"

## Set up complete development environment
setup-dev: install-dev
	@echo "Setting up development environment..."
	mkdir -p data outputs logs charts reports
	cp config/config.yaml config/dev_config.yaml
	@echo "✅ Development environment ready"

## Clean build artifacts and cache files
clean:
	@echo "Cleaning build artifacts..."
	rm -rf $(BUILD_DIR)/ $(DIST_DIR)/ *.egg-info/
	rm -rf .pytest_cache/ .coverage htmlcov/
	rm -rf .mypy_cache/ .tox/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*~" -delete
	@echo "✅ Cleanup completed"

## Run full test suite
test:
	@echo "Running full test suite..."
	pytest -v --cov=$(SRC_DIR) --cov-report=html --cov-report=term-missing
	@echo "✅ Tests completed"

## Run fast tests (excluding slow tests)
test-fast:
	@echo "Running fast tests..."
	pytest -v -m "not slow" --cov=$(SRC_DIR) --cov-report=term-missing
	@echo "✅ Fast tests completed"

## Generate test coverage report
test-coverage:
	@echo "Generating coverage report..."
	pytest --cov=$(SRC_DIR) --cov-report=html --cov-report=xml
	@echo "📊 Coverage report generated in htmlcov/"

## Run code linting
lint:
	@echo "Running linting checks..."
	flake8 $(SRC_DIR) $(TEST_DIR)
	@echo "✅ Linting completed"

## Format code with black and isort
format:
	@echo "Formatting code..."
	black $(SRC_DIR) $(TEST_DIR) setup.py
	isort $(SRC_DIR) $(TEST_DIR) setup.py
	@echo "✅ Code formatting completed"

## Run type checking
type-check:
	@echo "Running type checks..."
	mypy $(SRC_DIR)
	@echo "✅ Type checking completed"

## Run all quality checks
quality: format lint type-check test-fast
	@echo "✅ All quality checks completed"

## Build package distributions
build: clean
	@echo "Building package distributions..."
	$(PYTHON) setup.py sdist bdist_wheel
	@echo "📦 Package built in $(DIST_DIR)/"

## Create package for distribution
package: test build
	@echo "Creating package for distribution..."
	twine check $(DIST_DIR)/*
	@echo "✅ Package ready for distribution"

## Build Docker image
docker-build:
	@echo "Building Docker image..."
	docker build \
		--build-arg BUILD_DATE="$(shell date -u +'%Y-%m-%dT%H:%M:%SZ')" \
		--build-arg VCS_REF="$(shell git rev-parse --short HEAD)" \
		--build-arg VERSION="$(VERSION)" \
		-t $(DOCKER_IMAGE):$(DOCKER_TAG) .
	@echo "🐳 Docker image built: $(DOCKER_IMAGE):$(DOCKER_TAG)"

## Build development Docker image
docker-build-dev:
	@echo "Building development Docker image..."
	docker build \
		--target builder \
		--build-arg BUILD_DATE="$(shell date -u +'%Y-%m-%dT%H:%M:%SZ')" \
		--build-arg VCS_REF="$(shell git rev-parse --short HEAD)" \
		--build-arg VERSION="$(VERSION)" \
		-t $(DOCKER_IMAGE):$(DOCKER_DEV_TAG) .
	@echo "🐳 Development Docker image built: $(DOCKER_IMAGE):$(DOCKER_DEV_TAG)"

## Run Docker container with sample data
docker-run:
	@echo "Running Docker container..."
	docker run -it --rm \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/outputs:/app/outputs \
		$(DOCKER_IMAGE):$(DOCKER_TAG) --help
	@echo "🐳 Docker container executed"

## Run analysis in Docker container
docker-analyze:
	@echo "Running analysis in Docker container..."
	@if [ ! -f data/high-school-directory.csv ]; then \
		echo "❌ Data file not found: data/high-school-directory.csv"; \
		exit 1; \
	fi
	docker run -it --rm \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/outputs:/app/outputs \
		$(DOCKER_IMAGE):$(DOCKER_TAG) \
		analyze /app/data/high-school-directory.csv --borough BROOKLYN --grade 9
	@echo "🐳 Analysis completed"

## Clean Docker images and containers
docker-clean:
	@echo "Cleaning Docker images and containers..."
	docker rmi $(DOCKER_IMAGE):$(DOCKER_TAG) $(DOCKER_IMAGE):$(DOCKER_DEV_TAG) 2>/dev/null || true
	docker system prune -f
	@echo "🐳 Docker cleanup completed"

## Build documentation
docs:
	@echo "Building documentation..."
	cd $(DOCS_DIR) && make html
	@echo "📚 Documentation built in $(DOCS_DIR)/_build/html/"

## Serve documentation locally
serve-docs: docs
	@echo "Serving documentation at http://localhost:8000"
	cd $(DOCS_DIR)/_build/html && $(PYTHON) -m http.server 8000

## Run security checks
security-check:
	@echo "Running security checks..."
	safety check
	bandit -r $(SRC_DIR)
	@echo "🔒 Security checks completed"

## Run pre-commit hooks
pre-commit:
	@echo "Running pre-commit hooks..."
	pre-commit run --all-files
	@echo "✅ Pre-commit checks completed"

## Create a new release
release: clean quality test package
	@echo "Creating release $(VERSION)..."
	git tag -a v$(VERSION) -m "Release version $(VERSION)"
	@echo "🚀 Release v$(VERSION) created"
	@echo "Don't forget to: git push origin v$(VERSION)"

## Deploy to staging environment
deploy-staging: docker-build
	@echo "Deploying to staging environment..."
	docker tag $(DOCKER_IMAGE):$(DOCKER_TAG) $(DOCKER_IMAGE):staging
	# Add your staging deployment commands here
	@echo "🚀 Deployed to staging"

## Deploy to production environment
deploy-prod: release docker-build
	@echo "Deploying to production environment..."
	docker tag $(DOCKER_IMAGE):$(DOCKER_TAG) $(DOCKER_IMAGE):v$(VERSION)
	# Add your production deployment commands here
	@echo "🚀 Deployed to production"

## Run sample analysis (requires data file)
sample-analysis:
	@echo "Running sample analysis..."
	@if [ ! -f data/high-school-directory.csv ]; then \
		echo "❌ Sample data file not found: data/high-school-directory.csv"; \
		echo "Please place the NYC school directory CSV file in the data/ directory"; \
		exit 1; \
	fi
	nyc-schools analyze data/high-school-directory.csv \
		--output-dir outputs/sample \
		--borough BROOKLYN \
		--grade 9 \
		--format csv json
	@echo "✅ Sample analysis completed in outputs/sample/"

## Generate project statistics
stats:
	@echo "Project Statistics:"
	@echo "==================="
	@echo "Lines of code:"
	@find $(SRC_DIR) -name "*.py" -exec wc -l {} + | tail -1
	@echo
	@echo "Test files:"
	@find $(TEST_DIR) -name "test_*.py" | wc -l | xargs echo "Test files:"
	@echo
	@echo "Python files:"
	@find $(SRC_DIR) -name "*.py" | wc -l | xargs echo "Source files:"
	@echo
	@echo "Git status:"
	@git status --porcelain | wc -l | xargs echo "Modified files:"

## Check project health
health-check: lint type-check test-fast security-check
	@echo "🏥 Project health check completed"

## Initialize project (first-time setup)
init: setup-dev
	@echo "Initializing project..."
	git init
	git add .
	git commit -m "Initial commit"
	@echo "✅ Project initialized"

## Show available make targets
list:
	@$(MAKE) -pRrq -f $(lastword $(MAKEFILE_LIST)) : 2>/dev/null | awk -v RS= -F: '/^# File/,/^# Finished Make data base/ {if ($$1 !~ "^[#.]") {print $$1}}' | sort | egrep -v -e '^[^[:alnum:]]' -e '^$@$$'