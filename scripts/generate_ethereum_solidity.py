"""
Generate Solidity output from ARTHEN compiler for Ethereum build-level acceptance.
Writes artifacts to build_artifacts/ethereum/ai_governance_system.sol
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "ai_governance_system.arthen"
OUT_DIR = ROOT / "build_artifacts" / "ethereum"
OUT_FILE = OUT_DIR / "ai_governance_system.sol"

# Ensure offline-friendly mode for HF models
os.environ.setdefault("ARTHEN_TEST_MODE", "true")

from compiler.arthen_compiler_architecture import ARTHENCompiler, CompilationConfig, BlockchainTarget

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    source = EXAMPLE.read_text(encoding="utf-8")
    config = CompilationConfig(target_blockchain=BlockchainTarget.ETHEREUM)
    compiler = ARTHENCompiler()
    result = compiler.compile(source, config)
    code = result.get("compiled_code", "")
    if not code:
        raise RuntimeError("compiled_code is empty")
    OUT_FILE.write_text(code, encoding="utf-8")
    print(f"Wrote: {OUT_FILE}")

if __name__ == "__main__":
    main()