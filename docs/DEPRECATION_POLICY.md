# ARTHEN-LANG Deprecation Policy

This policy ensures consumers have time to migrate when APIs or language constructs change.

## Scope
- Server API endpoints `/api/parse`, `/api/compile`
- Response JSON keys documented in docs/API.md
- CLI command options for `bin/arthen.js`
- Parser/Lexer public constructors and their defaults

## Principles
- Backward compatibility first: prefer additive changes.
- Provide clear migration guides for breaking changes.
- Maintain deprecation windows.

## Windows
- Alpha: 1 minor release window (e.g., deprecate in 1.1, remove in 1.2)
- Beta: 2 minor release windows (e.g., deprecate in 1.2, remove in 1.4)
- GA: 1 major release window (e.g., deprecate in 2.1, remove in 3.0)

## Communication
- Document deprecations in RELEASE_NOTES.md and GitHub Releases.
- Optionally emit server logs with `DEPRECATED` tags for affected endpoints.

## Example
- Changing `target` enum values for `/api/compile`:
  - Add new values without removing existing ones.
  - If removal is necessary, mark old value deprecated, accept it with warning for the window, add migration note.