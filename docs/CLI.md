# ARTHEN CLI

`bin/arthen.js` provides developer tooling to compile, analyze, and scaffold ARTHEN projects.

Status: Alpha. Commands are subject to change under the deprecation policy.

## Install
- Local: `npm install` (ensures `bin/arthen.js` is executable)
- Use via `node`: `node bin/arthen.js <command>`
- When published: `npm i -g arthen` (planned)

## Commands

### compile
Compile ARTHEN source into target-specific output.

Usage:
```
arthen compile <source> --target <ethereum|solana|cosmos|polkadot|near|move_aptos|cardano> --output <dir> [--optimize] [--ai-level <1-5>]
```
Notes:
- Uses `compiler/arthen_compiler_architecture.py` under the hood.
- Writes artifacts to the output directory.

### deploy
Simulated deploy flow for now; planned integration with real toolchains in examples.
```
arthen deploy <contract> --network <ethereum|solana|...> [--verify]
```

### test
Run tests (placeholder).
```
arthen test [pattern] [--coverage] [--ai-test-gen]
```

### init
Initialize a new ARTHEN project.
```
arthen init [name] --template <basic>
```

### analyze
AI-powered analysis (placeholder outputs).
```
arthen analyze <source> [--security] [--performance] [--gas]
```

### version
Show version info.
```
arthen version
```

## Examples
```
arthen compile examples/ai_governance_system.arthen --target ethereum --output build
```

## Deprecation & Versioning
- See docs/DEPRECATION_POLICY.md and docs/VERSIONING.md.