# ARTHEN-LANG Versioning Policy

Goal: Provide predictable releases for language spec, parser/lexer, compiler, and API, minimizing breakage for consumers.

## Components
- Language Spec: `ARTHEN_SPEC.md` (Spec Version)
- Parser/Lexer: `parser/*`
- Compiler: `compiler/*`
- Server API: `server/backend/app/*`
- CLI: `bin/arthen.js`

## SemVer Strategy
- Overall project version: `MAJOR.MINOR.PATCH`
- Until 1.x is GA, we will use pre-release labels or document alpha stability for APIs.
- Breaking changes bump MAJOR; backward-compatible additions bump MINOR; bugfix bump PATCH.

## Spec Version vs Package Version
- Spec Version is declared in ARTHEN_SPEC.md header (currently 2.0). It indicates the logical language design version.
- Package Version refers to release artifacts (`pyproject.toml`, `package.json`).
- Spec changes that affect syntax/semantics will trigger a MINOR bump at minimum and require deprecation windows if breaking.

## API Stability
- Stable response keys for parse/compile:
  - Parse: `token_count`, `node_count`, `ast`, `backend_mode`
  - Compile: `compiled_code`, `target_blockchain`, `ast`, `security_report`, `optimization_metrics`
- Additive fields are allowed without breaking existing clients.
- Removals/renames require deprecation notice.

## Parser/Lexer Compatibility
- Changes to `NeuralLexer` and `TransformerASTGenerator` constructors must remain backward-compatible (we support optional `model_mode`).
- When new constructor parameters are introduced, retain defaults and provide fallbacks in server code.

## Release Channels
- Developer Preview/Alpha: frequent changes, document in RELEASE_NOTES.md
- Beta: spec freeze window, no breaking changes without deprecation plan
- GA: Formalized acceptance tests and build-level validation across major chains

## Announcements
- Breaking changes and deprecations will be announced via:
  - RELEASE_NOTES.md
  - GitHub Releases
  - docs/DEPRECATION_POLICY.md