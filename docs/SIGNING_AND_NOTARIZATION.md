# Code Signing & Notarization Guide

This document outlines the steps and requirements to sign Windows executables and sign/notarize macOS binaries for ARTHEN CLI releases.

## Overview
- Windows: Authenticode code signing (recommended: OV/EV Code Signing Certificate). Optional timestamping.
- macOS: Developer ID Application certificate + Notarization via Apple (notarytool) + stapling.
- Linux: No native code signing for ELF; optionally provide GPG signatures for artifacts.

## Windows (Authenticode)

### Requirements
- Code Signing Certificate (OV or EV). If EV with hardware token, signing may need to run on a secure host.
- Convert certificate to PFX (PKCS#12) with password.
- Windows SDK (includes `signtool.exe`).
- Timestamp server URL (e.g., Digicert: `http://timestamp.digicert.com`).

### GitHub Actions Secrets
- `WINDOWS_SIGNING_PFX`: Base64-encoded PFX content.
- `WINDOWS_SIGNING_PFX_PASSWORD`: PFX password.

### Signing Steps (Windows Runner)
1. Decode PFX to file:
   ```powershell
   $bytes = [Convert]::FromBase64String($env:WINDOWS_SIGNING_PFX)
   [IO.File]::WriteAllBytes("signing.pfx", $bytes)
   ```
2. Sign binaries using `signtool`:
   ```powershell
   signtool sign `
     /f signing.pfx `
     /p $env:WINDOWS_SIGNING_PFX_PASSWORD `
     /tr http://timestamp.digicert.com `
     /td sha256 `
     /fd sha256 `
     dist\bin\arthen-win-x64.exe
   ```
3. Verify signature:
   ```powershell
   signtool verify /pa dist\bin\arthen-win-x64.exe
   ```

### Alternative (Linux Runner)
- Use `osslsigncode` to sign PE files if Windows runner is not available (less common in production).

### Notes
- EV certificates with hardware tokens typically require Windows runner with connected token.
- Restrict access to secrets; never print secrets in logs.

## macOS (Developer ID + Notarization)

### Requirements
- Apple Developer account (Team ID).
- Developer ID Application certificate (.p12).
- Xcode CLI tools installed (macOS runner).
- Notarization credentials via notarytool:
  - Option A (Apple ID): Apple ID and App-specific password.
  - Option B (API key): App Store Connect API key (`.p8`), Key ID, and Team ID.

### GitHub Actions Secrets
- If using Apple ID:
  - `APPLE_ID`: Apple ID email.
  - `APPLE_APP_PASSWORD`: App-specific password.
  - `APPLE_TEAM_ID`: Team ID.
  - `APPLE_P12_BASE64`: Base64-encoded Developer ID Application certificate (.p12).
  - `APPLE_P12_PASSWORD`: Certificate password.
- If using API key (preferred):
  - `APPLE_API_KEY_BASE64`: Base64-encoded `.p8` content.
  - `APPLE_API_KEY_ID`: Key ID.
  - `APPLE_TEAM_ID`: Team ID.
  - `APPLE_P12_BASE64` / `APPLE_P12_PASSWORD` for codesign.

### Signing + Notarization Steps (macOS Runner)
1. Import Developer ID certificate into keychain:
   ```bash
   mkdir -p signing
   echo "$APPLE_P12_BASE64" | base64 --decode > signing/dev_id_app.p12
   security import signing/dev_id_app.p12 -k ~/Library/Keychains/login.keychain -P "$APPLE_P12_PASSWORD" -T /usr/bin/codesign
   security find-identity -v -p codesigning
   ```
2. Code sign macOS binaries:
   ```bash
   CODESIGN_ID="Developer ID Application: Your Company Name (TEAMID)"
   codesign --force --options runtime --timestamp --sign "$CODESIGN_ID" dist/bin/arthen-macos-x64
   codesign --force --options runtime --timestamp --sign "$CODESIGN_ID" dist/bin/arthen-macos-arm64
   codesign --verify --deep --strict --verbose=2 dist/bin/arthen-macos-x64
   codesign --verify --deep --strict --verbose=2 dist/bin/arthen-macos-arm64
   ```
3. Notarization via notarytool (Apple ID):
   ```bash
   xcrun notarytool submit dist/bin/arthen-macos-x64 \
     --apple-id "$APPLE_ID" --team-id "$APPLE_TEAM_ID" --password "$APPLE_APP_PASSWORD" --wait
   xcrun notarytool submit dist/bin/arthen-macos-arm64 \
     --apple-id "$APPLE_ID" --team-id "$APPLE_TEAM_ID" --password "$APPLE_APP_PASSWORD" --wait
   ```
   Or via API key:
   ```bash
   echo "$APPLE_API_KEY_BASE64" | base64 --decode > signing/AuthKey_$APPLE_API_KEY_ID.p8
   xcrun notarytool submit dist/bin/arthen-macos-x64 \
     --key signing/AuthKey_$APPLE_API_KEY_ID.p8 --key-id "$APPLE_API_KEY_ID" --issuer "$APPLE_TEAM_ID" --wait
   xcrun notarytool submit dist/bin/arthen-macos-arm64 \
     --key signing/AuthKey_$APPLE_API_KEY_ID.p8 --key-id "$APPLE_API_KEY_ID" --issuer "$APPLE_TEAM_ID" --wait
   ```
4. Staple tickets to binaries:
   ```bash
   xcrun stapler staple dist/bin/arthen-macos-x64
   xcrun stapler staple dist/bin/arthen-macos-arm64
   ```
5. Post-check (optional):
   ```bash
   spctl --assess --type execute --verbose dist/bin/arthen-macos-x64
   spctl --assess --type execute --verbose dist/bin/arthen-macos-arm64
   ```

### Notes
- Gatekeeper will trust stapled (notarized) binaries and reduce quarantine prompts.
- If testing locally, remove quarantine: `xattr -dr com.apple.quarantine ./arthen-macos-<arch>`.

## Linux (Optional GPG Signatures)
- Generate and publish detached signatures for transparency:
  ```bash
  gpg --armor --detach-sign --output dist/bin/arthen-linux-x64.asc dist/bin/arthen-linux-x64
  ```
- Publish `.asc` alongside binaries.

## Workflow Integration (Proposal)
- Add signing jobs that depend on packaging and run per OS:
  - Windows job (windows-latest): sign PE using `signtool` with PFX secrets.
  - macOS job (macos-latest): import Developer ID certificate, `codesign`, `notarytool submit`, and `stapler staple`.
- Only run signing/notarization steps if secrets are provided:
  - Example conditional: `if: ${{ secrets.WINDOWS_SIGNING_PFX != '' }}`.
- Upload signed binaries to release after successful signing/notarization.

## Security Best Practices
- Restrict secret access to release workflows and protected branches.
- Avoid echoing sensitive values; mask secrets in logs.
- Rotate certificates and API keys periodically.
- Keep a separate signing account/team with strict MFA.