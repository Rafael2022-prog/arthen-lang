<p align="center"><img src="../ARTHEN-LOGO-TRANSPARANSI.png" width="160" alt="ARTHEN Logo"></p>
# Installation Guide

This guide explains how to install ARTHEN and how file icons for `.arthen`/`.arthenn` are set up automatically at the OS level (Windows/Linux/macOS).

## Prerequisites
- Node.js >= 16
- Python >= 3.8 (for compiler/runtime parts)

## Install
- Local (from repo):
  - `npm install`
  - Use CLI: `node bin/arthen.js <command>`
- Global (when published):
  - `npm i -g arthen`
  - Use CLI: `arthen <command>`
- Direct Binaries (no Node/npm):
  - Download from Releases → Assets `dist/bin`: `arthen-win-x64.exe`, `arthen-linux-x64`, `arthen-macos-x64`, `arthen-macos-arm64`
  - Windows (PowerShell): `./arthen-win-x64.exe --help`
  - Linux/macOS: `chmod +x ./arthen-<linux-x64|macos-x64|macos-arm64>` then run `./arthen-<...> --help`

## Automatic Icon Setup (Postinstall)
ARTHEN attempts to configure OS-level file icons for `.arthen` and `.arthenn` during `npm install` via a postinstall script:

- Runs: `arthen setup-icons --user-level`
- Skipped automatically in CI/headless/container environments.
- If skipped or you want to run manually, use:
  - `npx arthen setup-icons --user-level`

### Windows
- User-level (no admin): `npx arthen setup-icons --os windows --user-level`
- System-level (requires admin): run PowerShell as Administrator, then:
  - `powershell -NoProfile -ExecutionPolicy Bypass -File scripts\setup_windows_arthen_icon.ps1`
- Refresh Explorer if icons are not visible:
  - `Stop-Process -Name explorer; Start-Process explorer`

### Linux
- User-level: `npx arthen setup-icons --os linux --user-level`
- Requires standard desktop tools: `xdg-mime`, `xdg-icon-resource`.
- If icons do not appear:
  - `xdg-mime forceupdate`
  - `gtk-update-icon-cache ~/.local/share/icons/hicolor`

### macOS
- User-level: `npx arthen setup-icons --os macos --user-level`
- The script creates a minimal registrar app bundle and registers UTIs for `.arthen`/`.arthenn`.
- Refresh Finder if icons do not appear:
  - `killall Finder; qlmanage -r`

## Cleanup (Rollback)
- Hapus asosiasi ikon yang sebelumnya dibuat:
  - `npx arthen cleanup-icons --user-level --os <windows|linux|macos>`
  - Binary langsung: `./arthen-<os> cleanup-icons --dry-run --user-level --os <windows|linux|macos>` (preview tanpa mengubah sistem)
- Pada sistem-level, jalankan dengan hak admin/`sudo` sesuai OS.

## CI/Headless Environments
- Skip icon setup by setting an environment variable before install:
  - `ARTHEN_SKIP_POSTINSTALL=1 npm ci`
- Postinstall also auto-skips when `CI=1` or when a container environment is detected.
- For CLI binaries: use `arthen-<os> setup-icons --dry-run --user-level --os <windows|linux|macos>` to verify planned actions without system changes.

## Troubleshooting
- Ensure `assets/icons/` exists and includes `arthen.ico` and PNG sizes.
- If your IDE uses a custom icon theme that overrides OS icons, you may need to disable that theme. ARTHEN also ships a VSCode icon theme as an optional fallback in releases.
- You can always force rerun: `npx arthen setup-icons --user-level`.

## Related
- CLI command docs: see `docs/CLI.md` → `setup-icons`.
- Windows/ Linux/ macOS scripts: `scripts/setup_windows_arthen_icon.ps1`, `scripts/setup_linux_arthen_icon.sh`, `scripts/setup_macos_arthen_icon.sh`, `scripts/cleanup_windows_arthen_icon.ps1`, `scripts/cleanup_linux_arthen_icon.sh`, `scripts/cleanup_macos_arthen_icon.sh`.