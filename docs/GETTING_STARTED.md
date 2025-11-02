# Getting Started with ARTHEN-LANG (Developer Preview)

This guide helps you run the backend API, Playground UI, and try end-to-end parse/compile across multiple blockchains.

Prerequisites
- Python 3.11+
- Node.js 18+
- PowerShell 7+ (Windows) or bash/zsh (macOS/Linux)

1) Install dependencies
- Backend:
  - pip install -r server/backend/requirements.txt
  - If you plan to run tests using compiler directly: pip install -r requirements.txt
- Frontend:
  - cd server/frontend
  - npm install

2) Run the backend API
- From repository root:
  - cd server/backend
  - uvicorn app.main:app --reload --port 8000
- Health check:
  - curl http://127.0.0.1:8000/api/health → {"status":"ok"}

3) Run the Playground UI
- From repository root:
  - cd server/frontend
  - npm run dev
- Open browser:
  - http://localhost:5173/ (or :5174 if :5173 is busy)

4) Try examples
- List examples:
  - curl http://127.0.0.1:8000/api/examples
- Fetch and compile to Ethereum (PowerShell):
  - $e = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/api/examples/defi_liquidity_pool.arthen' -Method GET
  - $body = @{ source = $e.content; target = 'ethereum' } | ConvertTo-Json
  - Invoke-RestMethod -Uri 'http://127.0.0.1:8000/api/compile' -Method POST -Body $body -ContentType 'application/json' | ConvertTo-Json -Depth 4

5) Playground features
- Multi-target dropdown (ethereum, solana, cosmos, polkadot, near, move_aptos, cardano)
- Auto-target selection via directives in source:
  - ∆compile_target⟨solana⟩
  - ∆target_chains: [ethereum, solana]
- Quick buttons per target for faster switching

6) Notes for deployment
- Compiled outputs are templates/scaffolding with AI/ML hooks. They require chain-specific SDKs and toolchains before production deployment.
- Security reports are indicative. Perform audit and verification per chain.

7) Troubleshooting
- If you see SyntaxError when running REST commands: ensure you are not inside Python REPL (no '>>>'). Use PowerShell or your shell.
- If you encounter Pydantic warnings about protected namespaces, backend already suppresses it, but keep your environment packages updated.

8) Next steps for public readiness
- Follow our roadmap: acceptance tests per chain, CI matrix, documentation, security, packaging, and buildable end-to-end examples.