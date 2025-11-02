# ARTHEN-LANG v1.0.0 Release Notes

This is the first official release of ARTHEN-LANG, a revolutionary native language for blockchain ecosystem development.

## Highlights
- **AI-Optimized Syntax**: A new paradigm in smart contract development, designed for AI-driven workflows.
- **ML-Driven Consensus**: Unifying consensus mechanisms through machine learning for harmonious blockchain networks.
- **Cross-Platform Compilation**: Write once, deploy anywhere. ARTHEN-LANG can be compiled to major blockchain platforms.
- **Autonomous Development**: Lays the foundation for self-optimizing and evolving smart contracts.

## What’s New
- **Specification Freeze (Alpha)**: A stability addendum has been added to `ARTHEN_SPEC.md`, defining frozen lexical tokens, directives, and a minimal grammar subset for public API stability.
- **Public API & Policies**: New documentation in `docs/API.md`, `docs/VERSIONING.md`, and `docs/DEPRECATION_POLICY.md` establishes request/response contracts, semantic versioning, and deprecation windows.
- **Build-Level Acceptance CI (Phase 2)**: Added `.github/workflows/build_acceptance.yml` with jobs for Ethereum (Solidity, solc + solhint + slither best-effort) and Solana (Rust, cargo) to validate toolchain readiness.
- **CLI & Hardening Guides**: Introduced `docs/CLI.md` and `docs/HARDENING_GUIDE.md` to guide usage and security best practices across major chains.
- **Release & Packaging Policy**: Added `docs/RELEASE_POLICY.md` to formalize release channels and packaging via `pyproject.toml` and GitHub Releases.
- **End-to-End Examples (Draft)**: Created minimal guides in `examples/ethereum/e2e-minimal/README.md` and `examples/solana/e2e-minimal/README.md` for adapting ARTHEN outputs to Sepolia and Devnet deployments.

## Getting Started
To get started with ARTHEN-LANG, clone the repository and install the dependencies:
```bash
git clone https://github.com/Rafael2022-prog/arthen-lang.git
cd arthen-lang
pip install -e .
```

We are excited to see what you will build with ARTHEN-LANG!