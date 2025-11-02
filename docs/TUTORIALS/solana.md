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
- Buat workspace Anchor:
```bash
anchor init arthen-anchor-minimal
cd arthen-anchor-minimal
```
- Salin sumber dari `build_artifacts/solana/src` ke `programs/arthen-anchor-minimal/src`, dan gabungkan isi `Cargo.toml`.
- Update `Anchor.toml` dan `programs/arthen-anchor-minimal/Cargo.toml` sesuai template Anchor.
- Tambahkan test (tests/arthen.test.ts):
```ts
import * as anchor from "@coral-xyz/anchor";
import { Program } from "@coral-xyz/anchor";
describe("arthen-anchor-minimal", () => {
  const provider = anchor.AnchorProvider.env();
  anchor.setProvider(provider);
  it("runs stub function", async () => {
    // Placeholder test: ensure provider works
    const conn = provider.connection;
    const slot = await conn.getSlot();
    console.log("Current slot:", slot);
  });
});
```
- Build & test:
```bash
anchor build
anchor test
```

Security notes
- Gunakan `cargo audit` untuk kerentanan dependency: `cargo install cargo-audit && cargo audit`
- Pertimbangkan `cargo clippy` dan `cargo fmt` untuk lint & formatting
- Lihat docs/HARDENING_GUIDE.md untuk panduan per-chain