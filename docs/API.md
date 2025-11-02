# ARTHEN-LANG API (Stable Alpha)

This document describes the stable request/response contracts for the public endpoints. The goal is to provide predictable, versioned responses for early adopters while we continue evolving internals.

Status: Alpha stability. Breaking changes will be announced per Deprecation Policy.

## Base
- Name: ARTHEN-LANG API
- Version: 1.0.0 (server/app version)
- Spec Version: 2.0 (language spec)
- Default Content-Type: application/json

## Endpoints

### GET /api/health
- Response 200:
  - { "status": "ok" }

### GET /api/version
- Response 200:
  - {
      "name": "ARTHEN-LANG",
      "version": "1.0.0",
      "release_url": "https://github.com/Rafael2022-prog/ARTHEN-LANG/releases/tag/v1.0.0"
    }

### POST /api/parse
- Request:
  - {
      "source": "<string: ARTHEN source>",
      "model_mode": "ai|ml|none"  // optional, default "ml"
    }
- Response 200 (Stable Fields):
  - {
      "token_count": <int>,
      "node_count": <int>,
      "ast": <object>,
      "backend_mode": "ai|ml|none"
    }
- Notes:
  - Pydantic protected namespace warnings are suppressed for cross-version compatibility.
  - AST format will be extended, but the top-level keys and types will remain stable.

### POST /api/compile
- Request:
  - {
      "source": "<string: ARTHEN source>",
      "target": "ethereum|solana|cosmos|polkadot|near|move_aptos|cardano",
      "optimization_level": "maximum|..." // default "maximum",
      "ai_enhancement": <bool>,
      "ml_consensus_integration": <bool>,
      "cross_chain_support": <bool>,
      "gas_optimization": <bool>,
      "security_analysis": <bool>
    }
- Response 200 (Stable Fields):
  - {
      "compiled_code": "<string>",
      "target_blockchain": "<string>",
      "ast": <object>,
      "security_report": <object>,
      "optimization_metrics": <object>
    }
- Notes:
  - compiled_code is the primary artifact for build-level validation.
  - security_report is indicative; formal audits are out of scope for this alpha.
  - optimization_metrics may be expanded but will keep the top-level key name stable.

## Deprecation & Versioning
- See docs/DEPRECATION_POLICY.md and docs/VERSIONING.md.
- Any removal or breaking rename of the stable response keys above will follow the deprecation window. New additive fields are allowed with backward compatibility.

## Error Handling
- Errors use HTTP 4xx/5xx with JSON {"detail": "..."}. Internals may include logs; see server/backend/app/main.py.

## CORS
- Allowed Origins: http://localhost:5173, http://127.0.0.1:5173, http://localhost:5174, http://127.0.0.1:5174

## Examples
- Try the compiler from the Playground UI or via curl:

```bash
curl -X POST http://localhost:8000/api/compile \
  -H "Content-Type: application/json" \
  -d '{
        "source": "∆∇⟨blockchain_contract⟩ { ∆compile_target⟨ethereum⟩ {} }",
        "target": "ethereum"
      }'
```