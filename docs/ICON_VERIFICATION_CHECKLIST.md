# OS Icon Verification Checklist (Manual)

Use this checklist to manually verify ARTHEN file icon setup and cleanup across Windows, Linux, and macOS.

## Prerequisites
- ARTHEN repo cloned and dependencies installed, or use direct CLI binaries from Releases.
- A few sample files:
  - `examples/empty.arthen`
  - `examples/ai_governance_system.arthen`
  - `examples/ai_nft_marketplace.arthenn` (rename any file to `.arthenn` if needed)
- Ensure assets exist: `assets/icons/arthen.ico` (Windows) and `assets/icons/arthen-*.png` (Linux/macOS).

## General Notes
- Prefer `--user-level` to avoid admin privileges.
- After setup/cleanup, icon caches may need refresh; commands are listed per OS below.
- For CI or preview: use `--dry-run` to inspect planned actions without modifying the system.

---

## Windows (Explorer)

### Setup Icons
- Run (npm):
  - `npx arthen setup-icons --os windows --user-level`
- Run (binary):
  - `./arthen-windows-<x64|arm64>.exe setup-icons --user-level`
- Expected output:
  - PowerShell command shown or executed.
  - No errors printed.

### Verify Icons
- In Explorer, navigate to `examples/`.
- Confirm `.arthen` and `.arthenn` files show ARTHEN icon.
- If icons do not appear:
  - Refresh Explorer process:
    - Open PowerShell as user: `Stop-Process -Name explorer; Start-Process explorer`
  - Or sign out/in.

### Screenshot Guidance
- Capture Explorer window showing at least one `.arthen` and one `.arthenn` file with ARTHEN icon.
- Include path bar showing location (e.g., `...\ARTHEN-LANG\examples`).

### Cleanup Icons
- Run (npm):
  - `npx arthen cleanup-icons --os windows --user-level`
- Run (binary):
  - `./arthen-windows-<x64|arm64>.exe cleanup-icons --user-level`
- Refresh Explorer as above.
- Verify icons revert to default and file association is removed.

---

## Linux (GNOME/Nautilus; KDE Dolphin also noted)

### Setup Icons
- Run (npm):
  - `npx arthen setup-icons --os linux --user-level`
- Run (binary):
  - `./arthen-linux-x64 setup-icons --user-level`
- Expected output:
  - `xdg-mime` and `xdg-icon-resource` operations complete without error.

### Refresh Icon Cache
- GNOME/Nautilus:
  - `xdg-mime forceupdate`
  - `gtk-update-icon-cache ~/.local/share/icons/hicolor`
- KDE (optional if testing on KDE):
  - `kbuildsycoca5 --noincremental`

### Verify Icons
- Open Nautilus (Files) to `examples/`.
- Confirm `.arthen` and `.arthenn` files show ARTHEN icon.

### Screenshot Guidance
- Capture Files/Nautilus window showing `.arthen` and `.arthenn` files with ARTHEN icon.
- Include path breadcrumb (e.g., `examples`).

### Cleanup Icons
- Run (npm):
  - `npx arthen cleanup-icons --os linux --user-level`
- Run (binary):
  - `./arthen-linux-x64 cleanup-icons --user-level`
- Refresh caches again using commands above.
- Verify icons revert to default and association removed.

---

## macOS (Finder)

### Setup Icons
- Run (npm):
  - `npx arthen setup-icons --os macos --user-level`
- Run (binary):
  - `./arthen-macos-<x64|arm64> setup-icons --user-level`
- Expected output:
  - Registrar app bundle is created and UTIs registered.

### Refresh Finder / QuickLook
- Refresh Finder:
  - `killall Finder`
- Refresh QuickLook:
  - `qlmanage -r`
- If icons still do not update, consider a reboot (macOS caches can be persistent).

### Verify Icons
- In Finder, navigate to `examples/`.
- Confirm `.arthen` and `.arthenn` files show ARTHEN icon.

### Screenshot Guidance
- Capture Finder window with `.arthen` and `.arthenn` files showing ARTHEN icon.
- Include path bar or show the folder in the sidebar.

### Cleanup Icons
- Run (npm):
  - `npx arthen cleanup-icons --os macos --user-level`
- Run (binary):
  - `./arthen-macos-<x64|arm64> cleanup-icons --user-level`
- Refresh Finder and QuickLook as above.
- Verify icons revert to default and association removed.

---

## Dry-Run (Preview without System Changes)
- Windows:
  - `arthen setup-icons --dry-run --user-level --os windows`
  - `arthen cleanup-icons --dry-run --user-level --os windows`
- Linux:
  - `arthen setup-icons --dry-run --user-level --os linux`
  - `arthen cleanup-icons --dry-run --user-level --os linux`
- macOS:
  - `arthen setup-icons --dry-run --user-level --os macos`
  - `arthen cleanup-icons --dry-run --user-level --os macos`
- Expected output includes planned commands and materialized script/icon paths.

---

## Reporting Template
- OS and Desktop Environment:
- Tool used (npm CLI or direct binary):
- Steps performed:
- Observations (include whether refresh commands were needed):
- Screenshots attached:
- Any errors or warnings in console:
- Final result (icons present / cleaned up as expected):