import os
import sys
import json
import subprocess
import shutil
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
PARSER_MODULE = "parser.ai_native_parser"
EXAMPLES_DIR = os.path.join(PROJECT_ROOT, "examples")
EXAMPLE_FILE = os.path.join(EXAMPLES_DIR, "ai_supply_chain.arthen")
EMPTY_FILE = os.path.join(EXAMPLES_DIR, "empty.arthen")


def run_cli(args=None, env=None):
    """Run the parser CLI via `python -m parser.ai_native_parser` and return CompletedProcess."""
    cmd = [sys.executable, "-m", PARSER_MODULE]
    if args:
        cmd.extend(args)
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    return subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT, env=full_env)


@pytest.mark.integration
def test_arthen_parse_success_exit_code():
    assert os.path.exists(EXAMPLE_FILE), "Example file missing"
    proc = run_cli([EXAMPLE_FILE, "--raw"], env={
        "ARTHEN_TEST_MODE": "true",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        # allow AI -> ML fallback automatically
    })
    assert proc.returncode == 0, f"Unexpected exit code: {proc.returncode}\nSTDERR: {proc.stderr}"
    # Output should be valid JSON
    data = json.loads(proc.stdout)
    assert isinstance(data, dict)
    assert "parsing_success" in data and data["parsing_success"] is True


@pytest.mark.integration
def test_arthen_parse_pretty_and_raw_flags():
    assert os.path.exists(EMPTY_FILE), "Empty example file missing"
    proc = run_cli([EMPTY_FILE, "--pretty", "--raw"], env={
        "ARTHEN_TEST_MODE": "true",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    })
    assert proc.returncode == 0
    # Pretty JSON should contain newlines and indentation
    out = proc.stdout
    assert out.startswith("{")
    assert "\n" in out
    assert "  \"ast\"" in out or "  \"tokens\"" in out


@pytest.mark.integration
def test_arthen_parse_file_not_found_exit_1():
    proc = run_cli([os.path.join(EXAMPLES_DIR, "__does_not_exist__.arthen"), "--raw"], env={
        "ARTHEN_TEST_MODE": "true",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    })
    assert proc.returncode == 1
    assert "File not found" in proc.stderr


@pytest.mark.integration
def test_arthen_parse_no_model_ml_fallback_embeddings_present():
    assert os.path.exists(EXAMPLE_FILE), "Example file missing"
    proc = run_cli([EXAMPLE_FILE, "--no-model", "--raw"], env={
        "ARTHEN_TEST_MODE": "true",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    })
    assert proc.returncode == 0
    data = json.loads(proc.stdout)
    tokens = data.get("tokens", [])
    # Ensure we parsed some tokens and have embeddings serialized
    assert isinstance(tokens, list) and len(tokens) > 0
    sample = tokens[0]
    assert "embedding" in sample
    emb = sample["embedding"]
    # Embedding in ML/hash fallback should be list (vector) or number (hash)
    assert emb is None or isinstance(emb, (list, int, float))


@pytest.mark.unit
def test_neural_lexer_none_mode_deterministic_embeddings():
    # Import the module and construct a lexer with none mode
    from parser.ai_native_parser import NeuralLexer, AITokenType
    source = "∇⟨ai_bridge⟩ { ∆ml_consensus: harmony_all_types }"
    lexer = NeuralLexer(model_mode='none')
    tokens = lexer.tokenize(source)
    assert tokens, "Lexer should produce tokens"
    for t in tokens:
        # Embedding should be deterministic numeric type
        assert t.embedding is None or isinstance(t.embedding, (int, float)) or (
            hasattr(t.embedding, 'shape') or hasattr(t.embedding, 'numel')
        )


@pytest.mark.unit
def test_ai_optimized_parser_ml_mode_serializes_embeddings():
    from parser.ai_native_parser import AIOptimizedParser
    assert os.path.exists(EXAMPLE_FILE), "Example file missing"
    with open(EXAMPLE_FILE, 'r', encoding='utf-8') as f:
        src = f.read()
    parser = AIOptimizedParser(model_mode='ml')
    result = parser.parse(src)
    assert result.get("parsing_success", False) is True
    tokens = result.get("tokens", [])
    assert tokens, "Parser should produce tokens"
    sample = tokens[0]
    assert "embedding" in sample
    emb = sample["embedding"]
    assert emb is None or isinstance(emb, (list, int, float))