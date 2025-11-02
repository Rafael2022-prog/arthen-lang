# Tutorial: Solana (Devnet) — Build from ARTHEN Output

Goal: Use ARTHEN to generate a minimal Rust crate and compile on Solana toolchain.

Prerequisites
- Rust toolchain (stable)
- Python 3.10+ with pip
- Solana CLI (optional for deployment)

1) Install ARTHEN and generate minimal crate
```bash
pip install -e .
python scripts/generate_solana_program.py
# crate path: build_artifacts/solana
```

2) Build crate
```bash
cd build_artifacts/solana
cargo build --quiet
```

3) Optional: integrate with Anchor
- Copy sources into an Anchor workspace (programs/<name>)
- Add lib.rs and Cargo.toml sections according to Anchor template
- Run `anchor build`

Security notes
- Use `cargo audit` for dependency vulnerabilities
- Consider `cargo clippy` and `cargo fmt` for lint & formatting
- See docs/HARDENING_GUIDE.md for chain-specific guidance