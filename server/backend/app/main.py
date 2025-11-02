import os
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging

# Ensure project root is on PYTHONPATH to import ARTHEN modules
ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

# Force offline/test-friendly mode for HF models
os.environ.setdefault("ARTHEN_TEST_MODE", "true")

# Import ARTHEN components
from parser.ai_native_parser import NeuralLexer, TransformerASTGenerator, AIParsingOptimizer, AIToken  # type: ignore
from compiler.arthen_compiler_architecture import (
    ARTHENCompiler,
    CompilationConfig,
    BlockchainTarget,
)  # type: ignore

# -----------------------------
# Pydantic request/response models
# -----------------------------
try:
    # Pydantic v2
    from pydantic import BaseModel, Field, ConfigDict
    HAS_CONFIGDICT = True
except Exception:
    # Pydantic v1 fallback
    from pydantic import BaseModel, Field
    ConfigDict = None
    HAS_CONFIGDICT = False

class ParseRequest(BaseModel):
    source: str = Field(..., description="ARTHEN source code to parse")
    model_mode: Optional[str] = Field(
        default="ml", description="Backend mode for parser: ai|ml|none"
    )

    # Suppress protected namespace warning across Pydantic versions
    if HAS_CONFIGDICT and ConfigDict is not None:
        model_config = ConfigDict(protected_namespaces=())
    else:
        class Config:
            protected_namespaces = ()

class ParseResponse(BaseModel):
    token_count: int
    node_count: int
    ast: Dict[str, Any]
    backend_mode: str

class CompileRequest(BaseModel):
    source: str = Field(..., description="ARTHEN source code to compile")
    target: str = Field(..., description="Target blockchain (ethereum|solana|cosmos|polkadot|near|move_aptos|cardano)")
    optimization_level: str = Field(default="maximum")
    ai_enhancement: bool = True
    ml_consensus_integration: bool = True
    cross_chain_support: bool = True
    gas_optimization: bool = True
    security_analysis: bool = True

class CompileResponse(BaseModel):
    compiled_code: str
    target_blockchain: str
    ast: Dict[str, Any]
    security_report: Dict[str, Any]
    optimization_metrics: Dict[str, Any]

class ExampleListResponse(BaseModel):
    examples: List[str]

class ExampleContentResponse(BaseModel):
    name: str
    content: str

# -----------------------------
# FastAPI app initialization
# -----------------------------
app = FastAPI(title="ARTHEN-LANG API", version="1.0.0")

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("arthen")

# CORS for local Vite dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Endpoints
# -----------------------------
@app.get("/api/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}

@app.get("/api/version")
def version() -> Dict[str, Any]:
    return {
        "name": "ARTHEN-LANG",
        "version": "1.0.0",
        "release_url": "https://github.com/Rafael2022-prog/ARTHEN-LANG/releases/tag/v1.0.0",
    }

@app.post("/api/parse", response_model=ParseResponse)
def parse(req: ParseRequest) -> ParseResponse:
    try:
        backend_mode = str(req.model_mode or "ml")
        logger.info(f"/api/parse backend_mode={backend_mode} source_len={len(req.source)}")
        # Inisialisasi lexer dengan fallback kompatibilitas versi
        try:
            lexer = NeuralLexer(backend_mode)
        except TypeError:
            lexer = NeuralLexer()
        tokens: List[AIToken] = lexer.tokenize(req.source)
        # Inisialisasi parser dengan fallback kompatibilitas versi
        try:
            parser = TransformerASTGenerator(backend_mode)
        except TypeError:
            parser = TransformerASTGenerator()
        raw_ast = parser.generate_ast(tokens)
        optimizer = AIParsingOptimizer()
        optimized_ast = optimizer.optimize(raw_ast)
        node_count = len(optimized_ast.get("nodes", [])) if isinstance(optimized_ast, dict) else 0
        backend_used = getattr(lexer, "model_mode", backend_mode)
        return ParseResponse(
            token_count=len(tokens),
            node_count=node_count,
            ast=optimized_ast,
            backend_mode=str(backend_used),
        )
    except Exception as e:
        logger.exception("Parse error")
        raise HTTPException(status_code=500, detail=f"Parse error: {str(e)}")

@app.post("/api/compile", response_model=CompileResponse)
def compile(req: CompileRequest) -> CompileResponse:
    try:
        logger.info(f"/api/compile target={req.target}")
        # Map target string to enum
        target_map = {
            "ethereum": BlockchainTarget.ETHEREUM,
            "solana": BlockchainTarget.SOLANA,
            "cosmos": BlockchainTarget.COSMOS,
            "polkadot": BlockchainTarget.POLKADOT,
            "near": BlockchainTarget.NEAR,
            "move_aptos": BlockchainTarget.MOVE_APTOS,
            "cardano": BlockchainTarget.CARDANO,
        }
        target_enum = target_map.get(req.target.lower())
        if target_enum is None:
            raise HTTPException(status_code=400, detail=f"Unsupported target: {req.target}")

        config = CompilationConfig(
            target_blockchain=target_enum,
            optimization_level=req.optimization_level,
            ai_enhancement=req.ai_enhancement,
            ml_consensus_integration=req.ml_consensus_integration,
            cross_chain_support=req.cross_chain_support,
            gas_optimization=req.gas_optimization,
            security_analysis=req.security_analysis,
        )

        compiler = ARTHENCompiler()
        result = compiler.compile(req.source, config)

        return CompileResponse(
            compiled_code=result.get("compiled_code", ""),
            target_blockchain=result.get("target_blockchain", target_enum.value),
            ast=result.get("ast", {}),
            security_report=result.get("security_report", {}),
            optimization_metrics=result.get("optimization_metrics", {}),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Compile error")
        raise HTTPException(status_code=500, detail=f"Compile error: {str(e)}")

@app.get("/api/examples", response_model=ExampleListResponse)
def list_examples() -> ExampleListResponse:
    examples_dir = ROOT_DIR / "examples"
    if not examples_dir.exists():
        return ExampleListResponse(examples=[])
    names = [p.name for p in examples_dir.glob("*.arthen")]
    return ExampleListResponse(examples=sorted(names))

@app.get("/api/examples/{name}", response_model=ExampleContentResponse)
def get_example(name: str) -> ExampleContentResponse:
    examples_dir = ROOT_DIR / "examples"
    file_path = examples_dir / name
    if not file_path.exists() or not file_path.suffix == ".arthen":
        raise HTTPException(status_code=404, detail="Example not found")
    content = file_path.read_text(encoding="utf-8")
    return ExampleContentResponse(name=name, content=content)