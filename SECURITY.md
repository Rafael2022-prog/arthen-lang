# Security Policy and Best Practices

ARTHEN-LANG is currently in Developer Preview. The following practices help improve security prior to production use.

1) Scope and assumptions
- Compiled outputs include AI/ML optimization hooks; they are templates and require further hardening per chain.
- Treat outputs as untrusted until reviewed and tested.

2) Recommended steps before deployment
- Static analysis: run linters and analyzers appropriate to the target (e.g., Slither/Solhint for Solidity, cargo clippy/audit for Rust).
- Fuzz testing: use tools like Echidna (Solidity), proptest/quickcheck (Rust), and chain-specific fuzzers.
- Formal verification: consider Certora, Scribble, or other tools where applicable.
- Code reviews: at least two independent reviewers for critical logic.
- Audits: obtain external security audits before mainnet deployment.

3) Secrets and credentials
- Never commit secrets. Use environment variables or vaults.
- Rotate keys regularly and restrict permissions.

4) Reporting a vulnerability
- Please open a private security issue or email maintainers. Do not disclose details publicly until a fix is coordinated.

5) Roadmap
- Add automated static analysis in CI for generated outputs (Solidity/Rust/Move).
- Provide curated best-practice templates per chain.
- Publish security hardening guides per ecosystem.