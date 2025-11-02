import pathlib
import pytest

# Import compiler directly to avoid depending on running API server
from compiler.arthen_compiler_architecture import (
    ARTHENCompiler,
    CompilationConfig,
    BlockchainTarget,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "ai_governance_system.arthen"

TARGETS = [
    ("ethereum", BlockchainTarget.ETHEREUM),
    ("solana", BlockchainTarget.SOLANA),
    ("cosmos", BlockchainTarget.COSMOS),
    ("polkadot", BlockchainTarget.POLKADOT),
    ("near", BlockchainTarget.NEAR),
    ("move_aptos", BlockchainTarget.MOVE_APTOS),
    ("cardano", BlockchainTarget.CARDANO),
]

@pytest.mark.parametrize("name,enum", TARGETS)
def test_compile_acceptance(name, enum):
    source = EXAMPLE.read_text(encoding="utf-8")
    cfg = CompilationConfig(
        target_blockchain=enum,
        optimization_level="maximum",
        ai_enhancement=True,
        ml_consensus_integration=True,
        cross_chain_support=True,
        gas_optimization=True,
        security_analysis=True,
    )
    compiler = ARTHENCompiler()
    result = compiler.compile(source, cfg)
    assert isinstance(result, dict), "compiler should return a dict"
    code = result.get("compiled_code", "")
    assert isinstance(code, str) and len(code) > 10, f"compiled_code should be non-empty for {name}"
    assert "security_report" in result, "security_report should exist"
    assert "optimization_metrics" in result, "optimization_metrics should exist"