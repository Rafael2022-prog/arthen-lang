# Release & Packaging Policy

This document defines how ARTHEN-LANG versions are released, packaged, and distributed across channels. It complements docs/VERSIONING.md, which defines version semantics and stability levels.

Release channels
- Alpha (aX): Fast iteration, experimental features. Public API and SPEC may change. No PyPI upload. GitHub Releases only.
- Beta (bX): Stabilizing features, deprecation windows begin. GitHub Releases primary. Optional PyPI prerelease.
- GA (Stable): API and SPEC stability per VERSIONING.md. GitHub Releases + PyPI distribution. Security and quality gates enforced.

Version tags
- Use SemVer tags: vMAJOR.MINOR.PATCH
- Optional pre-release identifiers: vMAJOR.MINOR.PATCH-alpha.N, vMAJOR.MINOR.PATCH-beta.N
- Build metadata is discouraged for releases (use commit SHA in release body if needed)

Packaging artifacts
- Python package built via pyproject.toml (PEP 517):
  - Source tarball: dist/arthen_lang-<version>.tar.gz
  - Wheel: dist/arthen_lang-<version>-py3-none-any.whl
- Additional artifacts: CI-generated build-level acceptance proof logs and example outputs may be attached to the GitHub Release.

CI workflow
- .github/workflows/release.yml triggered by tags matching v*:
  - Setup Python 3.11
  - Install build and twine
  - Build package with python -m build
  - Twine check dist/* for metadata correctness
  - Create GitHub Release with body_path pointing to RELEASE_NOTES.md and attaching dist files
- PyPI publish will be added when GA criteria are met. Until then, manual or staged publish to TestPyPI may be performed for verification.

Release readiness checklist
- SPEC Freeze status documented in ARTHEN_SPEC.md
- API stability documented in docs/API.md and docs/DEPRECATION_POLICY.md
- Version consistency per docs/VERSIONING.md
- CI build-level acceptance jobs passing for Ethereum and Solana
- Docs updated: CLI, Hardening Guide, Examples directory
- Security linters integrated (solhint, slither best-effort)

Publishing to PyPI (GA only)
- twine upload dist/* using a repository token with two-person approval in organization secrets
- Use signed tags; release notes must include:
  - Changes summary
  - Breaking changes and migration
  - Deprecations (with effective version and EOL date)
  - Security advisories (if any)

Deprecations
- Follow docs/DEPRECATION_POLICY.md
- For removals, bump MAJOR version if any public API or SPEC change is breaking

Backports & Patch releases
- Patch releases only for critical fixes (no new features)
- Backport branches: release/vX.Y for long-lived minor lines (if needed)

Support policy
- Latest minor under current MAJOR is supported
- N-1 minor may receive security patches depending on severity

Feedback
- Open issues or discussions for proposed changes before GA
- RC builds may be announced in discussions for community testing