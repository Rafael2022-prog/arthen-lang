# Solana E2E Minimal (Draft)

This example outlines how to adapt ARTHEN compiled Rust to an Anchor or plain Solana program and deploy to devnet.

Status: Draft. You will need to shape the generated code into valid program modules.

## Steps (Plain Cargo)
1. Generate minimal crate using:
   ```bash
   python scripts/generate_solana_program.py
   ```
2. Inspect `build_artifacts/solana/src/lib.rs` and refine into your Solana program (entrypoints, accounts, instructions).
3. Build:
   ```bash
   cd build_artifacts/solana
   cargo build
   ```

## Steps (Anchor)
1. Install Anchor: follow official docs.
2. Initialize project:
   ```bash
   anchor init arthen_solana_program
   ```
3. Adapt ARTHEN output into `programs/arthen_solana_program/src/lib.rs`.
4. Build & deploy to devnet:
   ```bash
   anchor build && anchor deploy --provider.cluster devnet
   ```

## Notes
- Use `cargo clippy` and `cargo audit`.
- Validate account signers and PDAs.