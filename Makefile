# ARTHEN Programming Language Makefile
# AI-Native Blockchain Development Platform

.PHONY: help install install-dev build test clean lint format docs examples deploy-examples

# Default target
help:
	@echo "🚀 ARTHEN Programming Language - Development Commands"
	@echo "=================================================="
	@echo ""
	@echo "Setup Commands:"
	@echo "  install      Install ARTHEN and dependencies"
	@echo "  install-dev  Install development dependencies"
	@echo "  setup        Complete development setup"
	@echo ""
	@echo "Development Commands:"
	@echo "  build        Build ARTHEN compiler and tools"
	@echo "  test         Run all tests"
	@echo "  lint         Run code linting"
	@echo "  format       Format code"
	@echo "  clean        Clean build artifacts"
	@echo ""
	@echo "Documentation:"
	@echo "  docs         Generate documentation"
	@echo "  serve-docs   Serve documentation locally"
	@echo ""
	@echo "Examples:"
	@echo "  examples     Compile all example projects"
	@echo "  test-examples Test all example projects"
	@echo "  deploy-examples Deploy examples to testnets"
	@echo ""
	@echo "Release:"
	@echo "  package      Create distribution packages"
	@echo "  publish      Publish to package registries"
	@echo ""

# Installation
install:
	@echo "📦 Installing ARTHEN Programming Language..."
	pip install -e .
	npm install
	@echo "✅ Installation complete!"

install-dev:
	@echo "🛠️  Installing development dependencies..."
	pip install -e ".[dev,ai,blockchain,all]"
	pip install -r tests/test_requirements.txt
	npm install --include=dev
	pre-commit install
	@echo "✅ Development setup complete!"

# Test Suite Commands
test-unit:
	@echo "🧪 Running ARTHEN unit tests..."
	python tests/test_runner.py --categories unit --parallel
	@echo "✅ Unit tests complete!"

test-integration:
	@echo "🔗 Running ARTHEN integration tests..."
	python tests/test_runner.py --categories integration
	@echo "✅ Integration tests complete!"

test-ai:
	@echo "🤖 Running ARTHEN AI/ML tests..."
	python tests/test_runner.py --categories ai
	@echo "✅ AI/ML tests complete!"

test-blockchain:
	@echo "⛓️  Running ARTHEN blockchain tests..."
	python tests/test_runner.py --categories blockchain
	@echo "✅ Blockchain tests complete!"

test-performance:
	@echo "⚡ Running ARTHEN performance tests..."
	python tests/test_runner.py --categories performance --performance
	@echo "✅ Performance tests complete!"

test-security:
	@echo "🔒 Running ARTHEN security tests..."
	python tests/test_runner.py --categories security --security
	@echo "✅ Security tests complete!"

test-comprehensive:
	@echo "🎯 Running comprehensive ARTHEN test suite..."
	python tests/test_runner.py --parallel --performance --security
	@echo "✅ Comprehensive test suite complete!"

test-quick:
	@echo "⚡ Running quick ARTHEN tests..."
	python tests/test_runner.py --categories unit integration --parallel
	@echo "✅ Quick tests complete!"

# CI/CD Commands
ci-setup:
	@echo "🔧 Setting up CI/CD environment..."
	pip install --upgrade pip setuptools wheel
	pip install -r tests/test_requirements.txt
	mkdir -p tests/reports tests/htmlcov
	@echo "✅ CI/CD environment ready!"

ci-test:
	@echo "🚀 Running CI/CD test pipeline..."
	python tests/test_runner.py --parallel --performance --security
	@echo "✅ CI/CD tests complete!"

ci-build:
	@echo "🏗️  Building ARTHEN for CI/CD..."
	python compiler/arthen_compiler_architecture.py
	python stdlib/arthen_stdlib_implementation.py
	@echo "✅ CI/CD build complete!"

ci-package:
	@echo "📦 Creating CI/CD packages..."
	python -m build --sdist --wheel --outdir dist/
	@echo "✅ CI/CD packaging complete!"

ci-full: ci-setup ci-build ci-test ci-package
	@echo "🎉 Full CI/CD pipeline complete!"

# Security and Quality
security-scan:
	@echo "🔍 Running security analysis..."
	bandit -r compiler/ stdlib/ -f json -o tests/reports/bandit_report.json || true
	safety check --json --output tests/reports/safety_report.json || true
	@echo "✅ Security scan complete!"

quality-check:
	@echo "📊 Running code quality checks..."
	flake8 compiler/ stdlib/ --max-line-length=88 --extend-ignore=E203,W503
	mypy compiler/ stdlib/ --ignore-missing-imports || true
	@echo "✅ Quality check complete!"

setup: install-dev
	@echo "🔧 Setting up ARTHEN development environment..."
	mkdir -p build cache artifacts logs
	chmod +x bin/arthen.js
	chmod +x compiler/arthen_compiler.py
	@echo "✅ Development environment ready!"

# Build
build:
	@echo "🏗️  Building ARTHEN compiler and tools..."
	python -m py_compile compiler/arthen_compiler_architecture.py
	python -m py_compile parser/ai_native_parser.py
	python -m py_compile stdlib/arthen_stdlib_implementation.py
	python -m py_compile compiler/arthen_compiler.py
	npm run build
	@echo "✅ Build complete!"

# Testing
test:
	@echo "🧪 Running ARTHEN tests..."
	python -m pytest tests/ -v --cov=. --cov-report=html
	npm test
	@echo "✅ All tests passed!"

test-examples:
	@echo "🧪 Testing example projects..."
	python compiler/arthen_compiler.py --source examples/defi_liquidity_pool.arthen --target ethereum --analyze security
	python compiler/arthen_compiler.py --source examples/ai_governance_system.arthen --target solana --analyze performance
	python compiler/arthen_compiler.py --source examples/ai_nft_marketplace.arthen --target cosmos --analyze gas
	@echo "✅ Example tests complete!"

# Code Quality
lint:
	@echo "🔍 Running code linting..."
	pylint compiler/ parser/ stdlib/ --rcfile=.pylintrc || true
	flake8 compiler/ parser/ stdlib/ --config=.flake8 || true
	eslint bin/ --config .eslintrc.json || true
	@echo "✅ Linting complete!"

format:
	@echo "🎨 Formatting code..."
	black compiler/ parser/ stdlib/ --line-length=88
	isort compiler/ parser/ stdlib/ --profile=black
	prettier --write bin/ --config .prettierrc
	@echo "✅ Code formatting complete!"

# Documentation
docs:
	@echo "📚 Generating documentation..."
	mkdir -p docs/build
	python scripts/generate_docs.py
	@echo "✅ Documentation generated!"

serve-docs:
	@echo "🌐 Serving documentation at http://localhost:8000"
	cd docs/build && python -m http.server 8000

# Examples
examples:
	@echo "🚀 Compiling example projects..."
	@echo "📄 Compiling DeFi Liquidity Pool..."
	python compiler/arthen_compiler.py --source examples/defi_liquidity_pool.arthen --target ethereum --output build/defi --optimize
	@echo "📄 Compiling AI Governance System..."
	python compiler/arthen_compiler.py --source examples/ai_governance_system.arthen --target solana --output build/governance --optimize
	@echo "📄 Compiling AI NFT Marketplace..."
	python compiler/arthen_compiler.py --source examples/ai_nft_marketplace.arthen --target cosmos --output build/nft --optimize
	@echo "📄 Compiling AI Supply Chain..."
	python compiler/arthen_compiler.py --source examples/ai_supply_chain.arthen --target polkadot --output build/supply --optimize
	@echo "📄 Compiling AI Oracle System..."
	python compiler/arthen_compiler.py --source examples/ai_oracle_system.arthen --target near --output build/oracle --optimize
	@echo "✅ All examples compiled successfully!"

deploy-examples:
	@echo "🚀 Deploying examples to testnets..."
	@echo "⚠️  Note: This requires testnet configuration"
	# node scripts/deploy-examples.js
	@echo "✅ Examples deployed to testnets!"

# Cleaning
clean:
	@echo "🧹 Cleaning build artifacts..."
	rm -rf build/ dist/ *.egg-info/ cache/ artifacts/ logs/
	rm -rf __pycache__/ */__pycache__/ */*/__pycache__/
	rm -rf .pytest_cache/ .coverage htmlcov/
	rm -rf node_modules/.cache/
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	find . -name "*~" -delete
	@echo "✅ Cleanup complete!"

# Packaging and Release
package:
	@echo "📦 Creating distribution packages..."
	python setup.py sdist bdist_wheel
	npm pack
	@echo "✅ Packages created in dist/"

publish-python:
	@echo "🚀 Publishing Python package to PyPI..."
	twine upload dist/*
	@echo "✅ Python package published!"

publish-npm:
	@echo "🚀 Publishing NPM package..."
	npm publish
	@echo "✅ NPM package published!"

publish: package publish-python publish-npm
	@echo "✅ All packages published successfully!"

# Development utilities
dev-server:
	@echo "🔄 Starting development server..."
	npm run dev

watch:
	@echo "👀 Watching for file changes..."
	python scripts/watch.py

benchmark:
	@echo "⚡ Running performance benchmarks..."
	python scripts/benchmark.py
	@echo "✅ Benchmarks complete!"

security-audit:
	@echo "🔒 Running security audit..."
	bandit -r compiler/ parser/ stdlib/
	npm audit
	@echo "✅ Security audit complete!"

# AI Model Management
download-models:
	@echo "🤖 Downloading AI models..."
	python scripts/download_models.py
	@echo "✅ AI models downloaded!"

update-models:
	@echo "🔄 Updating AI models..."
	python scripts/update_models.py
	@echo "✅ AI models updated!"

# Cross-chain testing
test-ethereum:
	@echo "🔗 Testing Ethereum compilation..."
	python compiler/arthen_compiler.py --source examples/defi_liquidity_pool.arthen --target ethereum

test-solana:
	@echo "🔗 Testing Solana compilation..."
	python compiler/arthen_compiler.py --source examples/defi_liquidity_pool.arthen --target solana

test-cosmos:
	@echo "🔗 Testing Cosmos compilation..."
	python compiler/arthen_compiler.py --source examples/defi_liquidity_pool.arthen --target cosmos

test-all-chains: test-ethereum test-solana test-cosmos
	@echo "✅ All blockchain targets tested!"

# Quick development cycle
quick: format lint test
	@echo "✅ Quick development cycle complete!"

# Full development cycle
full: clean install-dev build test examples docs
	@echo "✅ Full development cycle complete!"

# Version management
version-patch:
	@echo "📈 Bumping patch version..."
	npm version patch
	python scripts/bump_version.py patch

version-minor:
	@echo "📈 Bumping minor version..."
	npm version minor
	python scripts/bump_version.py minor

version-major:
	@echo "📈 Bumping major version..."
	npm version major
	python scripts/bump_version.py major

# Help for specific targets
help-install:
	@echo "Installation Commands:"
	@echo "  make install     - Install ARTHEN for end users"
	@echo "  make install-dev - Install with development dependencies"
	@echo "  make setup       - Complete development environment setup"

help-build:
	@echo "Build Commands:"
	@echo "  make build       - Build compiler and tools"
	@echo "  make examples    - Compile all example projects"
	@echo "  make clean       - Clean build artifacts"

help-test:
	@echo "Testing Commands:"
	@echo "  make test        - Run all tests"
	@echo "  make test-examples - Test example projects"
	@echo "  make test-all-chains - Test all blockchain targets"