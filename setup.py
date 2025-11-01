#!/usr/bin/env python3
"""
ARTHEN Programming Language Setup
AI-Native Blockchain Development Platform
"""

from setuptools import setup, find_packages
import os
from pathlib import Path

# Read README file
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path, 'r', encoding='utf-8') as f:
        requirements = [
            line.strip() 
            for line in f.readlines() 
            if line.strip() and not line.startswith('#')
        ]
else:
    requirements = [
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "web3>=6.10.0",
        "numpy>=1.24.0",
        "click>=8.1.0"
    ]

setup(
    name="arthen-lang",
    version="1.0.0",
    author="ARTHEN Team",
    author_email="team@arthen-lang.org",
    description="AI-Native Programming Language for Blockchain Ecosystems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Rafael2022-prog/arthen-lang",
    project_urls={
        "Bug Tracker": "https://github.com/Rafael2022-prog/arthen-lang/issues",
        "Documentation": "https://docs.arthen-lang.org",
        "Source Code": "https://github.com/Rafael2022-prog/arthen-lang",
        "Homepage": "https://arthen-lang.org"
    },
    packages=["compiler", "parser", "stdlib"],
    py_modules=[],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Compilers",
        "Topic :: Software Development :: Code Generators",
        "Topic :: System :: Distributed Computing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Environment :: Console",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Internet :: WWW/HTTP :: Browsers",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    keywords=[
        "blockchain", "ai", "programming-language", "smart-contracts", 
        "cross-chain", "ml-consensus", "defi", "web3", "ethereum", 
        "solana", "cosmos", "polkadot", "compiler", "artificial-intelligence"
    ],
    python_requires=">=3.8",
    install_requires=[
        "click>=8.1.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "ai": [
            "torch>=2.0.0",
            "transformers>=4.30.0",
            "tensorflow>=2.13.0",
            "scikit-learn>=1.3.0",
            "pandas>=2.0.0",
            "huggingface_hub>=0.16.0",
            "safetensors>=0.3.0"
        ],
        "blockchain": [
            "web3>=6.10.0",
            "solana>=0.30.0",
            "cosmpy>=0.9.0",
            "substrate-interface>=1.7.0",
        ],
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.7.0",
            "pylint>=2.17.0",
            "mypy>=1.5.0",
            "pre-commit>=3.4.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
            "bandit>=1.7.5",
            "coverage>=7.3.0",
        ],
        "full": [
            "arthen-lang[ai]",
            "arthen-lang[blockchain]",
        ]
    },
    entry_points={
        "console_scripts": [
            "arthen-compile=compiler.arthen_compiler:main",
            "arthen-stdlib=stdlib.arthen_stdlib_implementation:main",
            "arthen-parse=parser.ai_native_parser:main"
        ]
    },
    include_package_data=True,
    package_data={
        "": [
            "*.arthen",
            "*.json",
            "*.md",
            "*.txt",
            "examples/*.arthen",
            "stdlib/*.arthen"
        ]
    },
    # data_files removed to avoid wheel build issues; rely on package_data and MANIFEST.in if needed

    zip_safe=False,
    platforms=["any"],
    license="MIT",
    test_suite="tests",
    cmdclass={},
    options={
        "build_scripts": {
            "executable": "/usr/bin/env python3"
        }
    }
)

# Post-installation message
def print_post_install_message():
    """Print message after successful installation"""
    print("\n" + "="*60)
    print("🎉 ARTHEN Programming Language installed successfully!")
    print("   AI-Native Blockchain Development Platform")
    print("="*60)
    print("\nQuick Start:")
    print("  1. Create a new project:")
    print("     arthen init my-project")
    print("\n  2. Compile ARTHEN code:")
    print("     arthen-compiler --source contract.arthen --target ethereum")
    print("\n  3. Deploy to blockchain:")
    print("     arthen deploy contract.arthen --network ethereum")
    print("\nDocumentation: https://docs.arthen-lang.org")
    print("Community: https://discord.gg/arthen")
    print("="*60)

import sys

if __name__ == "__main__":
    # Do not call setup() here to avoid double invocation; setup() is already called above.
    if "install" in sys.argv:
        try:
            print_post_install_message()
        except Exception:
            # Avoid encoding errors during non-UTF8 terminals
            pass