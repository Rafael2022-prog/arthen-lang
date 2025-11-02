# ARTHEN Hardening Guide (Initial)

Security best-practices for adapting ARTHEN outputs into production-grade projects. This is an initial guide; more per-chain depth will be added.

## Ethereum (Solidity)
- Use latest stable solc (0.8.x). Enable optimizer with appropriate runs.
- Lint with `solhint` and run static analysis with `slither`.
- Consider property-based testing with `echidna` (Docker recommended).
- Avoid unsafe external calls; use reentrancy guards.
- Validate inputs and enforce access control; minimize `delegatecall`.
- Pin exact dependency versions; run `npm audit` and `cargo audit` where applicable.

## Solana (Rust)
- Prefer Anchor framework for safety and ergonomics.
- Enforce account ownership and signer checks.
- Use `cargo clippy` and `cargo audit`.
- Explicitly handle arithmetic overflows (`checked_*`).
- Validate PDA seeds and instruction data sizes.

## Cosmos (CosmWasm)
- Use latest `wasmd`/`cosmwasm` toolchains.
- Avoid unbounded loops; ensure deterministic behavior.
- Validate message sender and permissions.

## Polkadot (ink!/Substrate)
- Validate environmental constraints for contracts.
- Use `cargo clippy` and static analysis linters.
- Avoid unsafe code; check `env().transferred_balance()` and storage costs.

## NEAR (Rust/AssemblyScript)
- Prefer Rust-based contracts for better tooling.
- Validate env parameters and storage costs; careful with large vectors.

## Move/Aptos
- Use precise resource types; enforce invariants via module tests.
- Lint with `aptos move` tools when available.

## Cardano (Plutus/Haskell)
- Prefer on-chain validation logic isolation.
- Use latest GHC and Plutus toolchain; follow official security best-practices.

## General
- Separate testnet and mainnet configurations.
- Implement observability: logs, metrics, and event tracing.
- Adopt defense-in-depth and principle of least privilege.
- Perform external security audits before production deployments.