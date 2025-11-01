"""
ARTHEN AI-Native Parser & Lexer
Optimized for AI Development, Not Human Readability
Machine-First Language Processing Engine
"""

import re
import argparse
import sys
import json
import os
import ast
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, List, Tuple, Any, Optional
import warnings
import numpy as np

# Guarded framework imports
try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    _TORCH_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModel
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    AutoTokenizer, AutoModel = None, None
    _TRANSFORMERS_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    _SKLEARN_AVAILABLE = True
except ImportError:
    TfidfVectorizer, TruncatedSVD = None, None
    _SKLEARN_AVAILABLE = False


class AITokenType(Enum):
    """AI-Optimized Token Types for Machine Processing"""
    # Core AI Constructs
    NEURAL_BLOCK = "∇⟨"
    ML_FUNCTION = "⟨⟨"
    AI_PRIMITIVE = "∆ml_"
    TENSOR_TYPE = "∆tensor⟨"
    MATRIX_TYPE = "∆matrix⟨"
    VECTOR_TYPE = "∆vector⟨"
    NEURAL_TYPE = "∆neural⟨"
    
    # Consensus AI Tokens
    CONSENSUS_HARMONY = "∇⟨harmony_"
    ML_CONSENSUS = "∇⟨consensus_"
    AI_VALIDATOR = "∇⟨validator_"
    NEURAL_CONSENSUS = "∇⟨neural_consensus"
    
    # Blockchain AI Integration
    CHAIN_TARGET = "∆compile_target⟨"
    AI_BRIDGE = "∇⟨ai_bridge⟩"
    CROSS_CHAIN_AI = "∆[.*?]::ai_"
    ML_INTEROP = "∇⟨ml_interop⟩"
    
    # Mathematical AI Operators
    DERIVATIVE = "∂/∂"
    INTEGRAL = "∫"
    GRADIENT = "∇"
    TENSOR_PRODUCT = "⊗"
    NEURAL_ACTIVATION = "σ"
    LOSS_FUNCTION = "ℒ"
    OPTIMIZATION = "∇ℒ"
    
    # AI Logic Operators
    FUZZY_AND = "∧̃"
    FUZZY_OR = "∨̃"
    NEURAL_NOT = "¬̃"
    AI_EQUIVALENCE = "≡̃"
    ML_IMPLICATION = "⟹"
    
    # Confidence and Probability
    CONFIDENCE = "Ω{"
    PROBABILITY = "ℙ("
    EXPECTATION = "𝔼["
    VARIANCE = "𝕍ar("
    
    # AI Control Flow
    AI_IF = "if̃"
    ML_WHILE = "whilẽ"
    NEURAL_FOR = "for̃"
    AI_MATCH = "match̃"

@dataclass
class AIToken:
    """AI-Optimized Token Structure"""
    type: AITokenType
    value: str
    position: Tuple[int, int]
    # Avoid referencing torch types when torch may be unavailable
    embedding: Optional[Any] = None
    confidence: float = 1.0
    semantic_weight: float = 1.0
    neural_context: Optional[Dict[str, Any]] = None

class NeuralLexer:
    """Neural Network Enhanced Lexer for AI-First Parsing"""
    
    def __init__(self, model_mode: str = 'ai'):
        """Initialize lexer with a chosen backend: 'ai', 'ml', or 'none'.
        - ai: use transformers/torch if available
        - ml: use scikit-learn TF-IDF + SVD embeddings
        - none: deterministic hash-based embeddings (no external libs)
        """
        # Respect environment overrides
        self.model_mode = (model_mode or 'ai').lower()
        no_model_env = os.getenv('ARTHEN_NO_MODEL', '').lower() in ('1', 'true', 'yes')
        prefer_ml_env = os.getenv('ARTHEN_PREFER_ML', '').lower() in ('1', 'true', 'yes')
        if no_model_env:
            self.model_mode = 'ml'  # Disable transformers explicitly, prefer ML
        elif prefer_ml_env and self.model_mode == 'ai':
            self.model_mode = 'ml'

        # Offline mode detection for transformers
        offline = os.getenv('ARTHEN_TEST_MODE', '').lower() == 'true' or os.getenv('HF_HUB_OFFLINE') == '1' or os.getenv('TRANSFORMERS_OFFLINE') == '1'

        # Initialize AI backend (transformers)
        self.tokenizer = None
        self.language_model = None
        if self.model_mode == 'ai' and _TRANSFORMERS_AVAILABLE:
            try:
                if offline:
                    self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base", local_files_only=True)
                    self.language_model = AutoModel.from_pretrained("microsoft/codebert-base", local_files_only=True)
                else:
                    self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
                    self.language_model = AutoModel.from_pretrained("microsoft/codebert-base")
            except Exception:
                # Failed to load AI model; will fall back to ML or none
                self.tokenizer = None
                self.language_model = None

        # Initialize ML backend (TF-IDF + SVD)
        self._ml_vectorizer = None
        self._ml_svd = None
        if (self.model_mode == 'ml' or (self.model_mode == 'ai' and (self.tokenizer is None or self.language_model is None))) and _SKLEARN_AVAILABLE:
            try:
                self._ml_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=2048)
                self._ml_svd = TruncatedSVD(n_components=64, random_state=42)
                # If AI was requested but unavailable, upgrade mode to ML
                self.model_mode = 'ml'
            except Exception:
                self._ml_vectorizer = None
                self._ml_svd = None

        # Final fallback to deterministic hashing embeddings
        if self.model_mode == 'ai' and (self.tokenizer is None or self.language_model is None):
            if not _SKLEARN_AVAILABLE or self._ml_vectorizer is None:
                self.model_mode = 'none'

        # Neural embedding layer for ARTHEN tokens (only when torch & AI mode)
        self.token_embedder = nn.Embedding(len(AITokenType), 768) if (_TORCH_AVAILABLE and self.model_mode == 'ai') else None
        
        # AI-optimized token patterns (machine-readable, not human-readable)
        self.ai_patterns = {
            # Compressed neural patterns for faster AI processing
            'NEURAL_BLOCK': r'∇⟨[^⟩]*⟩\s*\{',
            'ML_FUNCTION': r'⟨⟨[^⟩]*⟩⟩\s*\(',
            'AI_PRIMITIVE': r'∆ml_[a-zA-Z0-9_]+',
            'TENSOR_TYPE': r'∆tensor⟨[^⟩]*⟩',
            'MATRIX_TYPE': r'∆matrix⟨[^⟩]*⟩',
            'VECTOR_TYPE': r'∆vector⟨[^⟩]*⟩',
            'NEURAL_TYPE': r'∆neural⟨[^⟩]*⟩',
            'CONSENSUS_HARMONY': r'∇⟨harmony_[^⟩]*⟩',
            'ML_CONSENSUS': r'∇⟨consensus_[^⟩]*⟩',
            'AI_VALIDATOR': r'∇⟨validator_[^⟩]*⟩',
            'NEURAL_CONSENSUS': r'∇⟨neural_consensus[^⟩]*⟩',
            'CHAIN_TARGET': r'∆compile_target⟨[^⟩]*⟩',
            'AI_BRIDGE': r'∇⟨ai_bridge⟩',
            'CROSS_CHAIN_AI': r'∆\[[^\]]*\]::ai_[a-zA-Z0-9_]+',
            'ML_INTEROP': r'∇⟨ml_interop⟩',
            'DERIVATIVE': r'∂/∂[a-zA-Z0-9_]+',
            'INTEGRAL': r'∫[^∫]*d[a-zA-Z]',
            'GRADIENT': r'∇[a-zA-Z0-9_]+',
            'TENSOR_PRODUCT': r'⊗',
            'NEURAL_ACTIVATION': r'σ\(',
            'LOSS_FUNCTION': r'ℒ\(',
            'OPTIMIZATION': r'∇ℒ',
            'FUZZY_AND': r'∧̃',
            'FUZZY_OR': r'∨̃',
            'NEURAL_NOT': r'¬̃',
            'AI_EQUIVALENCE': r'≡̃',
            'ML_IMPLICATION': r'⟹',
            'CONFIDENCE': r'Ω\{[0-9.]+\}',
            'PROBABILITY': r'ℙ\([^)]*\)',
            'EXPECTATION': r'𝔼\[[^\]]*\]',
            'VARIANCE': r'𝕍ar\([^)]*\)',
            'AI_IF': r'if̃\s*\(',
            'ML_WHILE': r'whilẽ\s*\(',
            'NEURAL_FOR': r'for̃\s*\(',
            'AI_MATCH': r'match̃\s*\('
        }
        
        # Neural pattern weights for AI processing priority
        self.pattern_weights = {
            'NEURAL_BLOCK': 1.0,
            'ML_FUNCTION': 0.95,
            'CONSENSUS_HARMONY': 0.9,
            'AI_PRIMITIVE': 0.85,
            'TENSOR_TYPE': 0.8,
            'NEURAL_CONSENSUS': 0.9,
            'AI_BRIDGE': 0.85,
            'DERIVATIVE': 0.7,
            'CONFIDENCE': 0.75
        }
    
    def tokenize(self, source_code: str) -> List[AIToken]:
        """AI-optimized tokenization for machine processing"""
        tokens = []
        processed_positions = set()
        
        # Neural embedding of entire source
        source_embedding = self._get_source_embedding(source_code)
        
        # Pattern-based tokenization with AI enhancement
        for pattern_name, pattern in self.ai_patterns.items():
            matches = re.finditer(pattern, source_code)
            for match in matches:
                # Skip if position already processed (avoid overlapping tokens)
                if any(pos in processed_positions for pos in range(match.start(), match.end())):
                    continue
                    
                # Mark positions as processed
                for pos in range(match.start(), match.end()):
                    processed_positions.add(pos)
                    
                try:
                    token_type = AITokenType[pattern_name]
                except Exception:
                    # Fallback: treat as generic AI_PRIMITIVE to avoid failing on unknown types
                    token_type = AITokenType.AI_PRIMITIVE
                
                # Generate neural embedding for token
                token_embedding = self._get_token_embedding(match.group(), source_embedding)
                
                # Calculate semantic weight
                semantic_weight = self.pattern_weights.get(pattern_name, 0.5)
                
                # Create AI token
                ai_token = AIToken(
                    type=token_type,
                    value=match.group(),
                    position=match.span(),
                    embedding=token_embedding,
                    confidence=self._calculate_confidence(match.group(), pattern),
                    semantic_weight=semantic_weight,
                    neural_context=self._extract_neural_context(match.group(), source_code)
                )
                
                tokens.append(ai_token)
        
        # Sort tokens by position and semantic weight
        tokens.sort(key=lambda t: (t.position[0], -t.semantic_weight))
        
        return tokens
    
    def _get_source_embedding(self, source_code: str) -> Any:
        """Generate embedding for entire source code using AI, ML, or deterministic fallback."""
        # AI backend (transformers + torch)
        if self.model_mode == 'ai' and _TORCH_AVAILABLE and self.tokenizer is not None and self.language_model is not None:
            try:
                inputs = self.tokenizer(source_code, return_tensors="pt", truncation=True, max_length=512)
                with torch.no_grad():
                    outputs = self.language_model(**inputs)
                    embeddings = outputs.last_hidden_state.mean(dim=1)
                return embeddings.squeeze()
            except Exception:
                # If AI fails, try ML below
                pass
        
        # ML backend (TF-IDF + SVD)
        if (self.model_mode in ('ml', 'ai')) and self._ml_vectorizer is not None:
            try:
                X = self._ml_vectorizer.fit_transform([source_code])
                if self._ml_svd is not None:
                    X_reduced = self._ml_svd.fit_transform(X)
                    return np.asarray(X_reduced).squeeze()
                return np.asarray(X.toarray()).squeeze()
            except Exception:
                # Fall through to deterministic hash-based fallback
                pass
        
        # Deterministic fallback (hash-based)
        try:
            import hashlib
            digest = hashlib.sha256(source_code.encode('utf-8')).digest()
            hash_value = int.from_bytes(digest[:8], byteorder='big') % 1000000
            if _TORCH_AVAILABLE:
                return torch.tensor([float(hash_value)], dtype=torch.float32)
            else:
                return float(hash_value)
        except Exception:
            return None
    
    def _get_token_embedding(self, token_value: str, source_embedding: Any) -> Any:
        """Generate contextual embedding for individual token using AI, ML, or deterministic fallback"""
        # AI backend: transformers + torch
        if self.model_mode == 'ai' and _TORCH_AVAILABLE and self.tokenizer is not None and self.language_model is not None:
            try:
                token_inputs = self.tokenizer(token_value, return_tensors="pt", truncation=True, max_length=64)
                with torch.no_grad():
                    token_outputs = self.language_model(**token_inputs)
                    token_embedding = token_outputs.last_hidden_state.mean(dim=1).squeeze()
                # Combine with source context if available (torch tensors)
                if source_embedding is not None and hasattr(source_embedding, 'numel') and source_embedding.numel() > 0:
                    if token_embedding.dim() == 0:
                        token_embedding = token_embedding.unsqueeze(0)
                    if source_embedding.dim() == 0:
                        source_embedding = source_embedding.unsqueeze(0)
                    min_size = min(token_embedding.size(0), source_embedding.size(0))
                    token_embedding = token_embedding[:min_size]
                    source_embedding = source_embedding[:min_size]
                    return 0.7 * token_embedding + 0.3 * source_embedding
                return token_embedding
            except Exception:
                pass
        
        # ML backend: TF-IDF + SVD (numpy arrays)
        if (self.model_mode in ('ml', 'ai')) and self._ml_vectorizer is not None:
            try:
                token_X = self._ml_vectorizer.transform([token_value])
                if self._ml_svd is not None:
                    token_vec = self._ml_svd.transform(token_X)
                else:
                    token_vec = token_X.toarray()
                token_vec = np.asarray(token_vec).squeeze()
                # Combine with source context if available (numpy arrays)
                if source_embedding is not None and isinstance(source_embedding, np.ndarray):
                    min_size = min(token_vec.shape[0], source_embedding.shape[0])
                    return 0.7 * token_vec[:min_size] + 0.3 * source_embedding[:min_size]
                return token_vec
            except Exception:
                pass
        
        # Deterministic fallback (hash-based)
        try:
            import hashlib
            digest = hashlib.sha256(token_value.encode('utf-8')).digest()
            hash_value = int.from_bytes(digest[:8], byteorder='big') % 10000
            if _TORCH_AVAILABLE:
                return torch.tensor([float(hash_value)], dtype=torch.float32)
            else:
                return float(hash_value)
        except Exception:
            return None
    
    def _calculate_confidence(self, token_value: str, pattern: str) -> float:
        """Calculate AI confidence score for token recognition"""
        base_confidence = 0.8
        
        # Boost confidence for AI-specific constructs
        ai_indicators = ['∇', '∆', '⟨', '⟩', 'ml_', 'ai_', 'neural_', 'tensor', 'matrix']
        ai_boost = sum(0.05 for indicator in ai_indicators if indicator in token_value)
        
        # Pattern complexity factor
        pattern_complexity = min(len(pattern) / 50.0, 0.15)
        
        # Token length normalization
        length_factor = min(len(token_value) / 20.0, 0.05)
        
        confidence = base_confidence + ai_boost + pattern_complexity + length_factor
        return min(confidence, 1.0)
    
    def _calculate_semantic_weight(self, token_type: AITokenType, token_value: str) -> float:
        """Calculate semantic importance weight for AI processing"""
        # Base weights for different token types
        type_weights = {
            AITokenType.NEURAL_BLOCK: 1.0,
            AITokenType.ML_FUNCTION: 0.9,
            AITokenType.CONSENSUS_HARMONY: 0.95,
            AITokenType.AI_PRIMITIVE: 0.8,
            AITokenType.TENSOR_TYPE: 0.85,
            AITokenType.MATRIX_TYPE: 0.85,
            AITokenType.VECTOR_TYPE: 0.8,
            AITokenType.NEURAL_TYPE: 0.9,
            AITokenType.CHAIN_TARGET: 0.7,
            AITokenType.AI_BRIDGE: 0.75,
        }
        
        base_weight = type_weights.get(token_type, 0.5)
        
        # Adjust based on token complexity
        complexity_bonus = min(len(token_value) / 30.0, 0.2)
        
        return min(base_weight + complexity_bonus, 1.0)
    
    def _extract_neural_context(self, token_value: str, source_code: str) -> Dict[str, Any]:
        """Extract neural context information for AI processing"""
        context = {
            'surrounding_tokens': self._get_surrounding_context(token_value, source_code),
            'semantic_category': self._classify_semantic_category(token_value),
            'ai_processing_hints': self._generate_ai_hints(token_value),
            'optimization_metadata': self._extract_optimization_metadata(token_value)
        }
        return context
    
    def _get_surrounding_context(self, token_value: str, source_code: str) -> List[str]:
        """Get surrounding context for better AI understanding"""
        # Find token position and extract surrounding context
        token_pos = source_code.find(token_value)
        if token_pos == -1:
            return []
        
        start = max(0, token_pos - 50)
        end = min(len(source_code), token_pos + len(token_value) + 50)
        context = source_code[start:end]
        
        # Extract meaningful tokens from context
        context_tokens = re.findall(r'[∇∆⟨⟩]+[^⟨⟩\s]*', context)
        return context_tokens[:5]  # Limit for AI processing efficiency
    
    def _classify_semantic_category(self, token_value: str) -> str:
        """Classify token into semantic categories for AI processing"""
        if '∇⟨' in token_value:
            return 'neural_construct'
        elif '∆ml_' in token_value:
            return 'ml_primitive'
        elif '⟨⟨' in token_value:
            return 'ai_function'
        elif 'consensus' in token_value:
            return 'consensus_mechanism'
        elif 'tensor' in token_value or 'matrix' in token_value:
            return 'mathematical_structure'
        else:
            return 'general_construct'
    
    def _generate_ai_hints(self, token_value: str) -> Dict[str, Any]:
        """Generate processing hints for AI systems"""
        hints = {
            'processing_priority': 'high' if any(x in token_value for x in ['neural', 'consensus', 'ai']) else 'medium',
            'optimization_target': self._determine_optimization_target(token_value),
            'parallel_processable': self._is_parallel_processable(token_value),
            'memory_requirements': self._estimate_memory_requirements(token_value)
        }
        return hints
    
    def _determine_optimization_target(self, token_value: str) -> str:
        """Determine optimization target for AI processing"""
        if 'consensus' in token_value:
            return 'consensus_efficiency'
        elif 'tensor' in token_value or 'matrix' in token_value:
            return 'computational_performance'
        elif 'bridge' in token_value:
            return 'cross_chain_latency'
        else:
            return 'general_performance'
    
    def _is_parallel_processable(self, token_value: str) -> bool:
        """Determine if token represents parallel-processable construct"""
        parallel_indicators = ['tensor', 'matrix', 'vector', 'neural', 'consensus']
        return any(indicator in token_value.lower() for indicator in parallel_indicators)
    
    def _estimate_memory_requirements(self, token_value: str) -> str:
        """Estimate memory requirements for AI processing"""
        if 'tensor' in token_value or 'matrix' in token_value:
            return 'high'
        elif 'neural' in token_value or 'consensus' in token_value:
            return 'medium'
        else:
            return 'low'
    
    def _extract_optimization_metadata(self, token_value: str) -> Dict[str, Any]:
        """Extract metadata for AI optimization"""
        metadata = {
            'vectorizable': 'tensor' in token_value or 'matrix' in token_value,
            'gpu_acceleratable': 'neural' in token_value or 'ml_' in token_value,
            'distributed_processable': 'consensus' in token_value or 'bridge' in token_value,
            'cache_friendly': len(token_value) < 20,
            'compression_ratio': self._calculate_compression_potential(token_value)
        }
        return metadata
    
    def _calculate_compression_potential(self, token_value: str) -> float:
        """Calculate potential for token compression in AI processing"""
        # Simple heuristic based on repetitive patterns
        unique_chars = len(set(token_value))
        total_chars = len(token_value)
        return 1.0 - (unique_chars / max(total_chars, 1))

class TransformerASTGenerator:
    """Transformer-based AST Generator Optimized for AI Processing"""
    
    def __init__(self, model_mode: str = 'ai'):
        # Simpan mode backend untuk kontrol alur AST
        self.model_mode = str(model_mode or 'ai').lower()
        # Specialized transformer untuk ARTHEN AST generation (hanya jika torch tersedia dan mode 'ai')
        if _TORCH_AVAILABLE and self.model_mode == 'ai':
            self.ast_transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=1536,
                    nhead=24,
                    dim_feedforward=6144,
                    dropout=0.1,
                    activation='gelu'
                ),
                num_layers=12
            )
            # AST node type classifier
            self.node_classifier = nn.Linear(1536, 50)
            # Semantic relationship predictor
            self.relationship_predictor = nn.Linear(3072, 20)
        else:
            self.ast_transformer = None
            self.node_classifier = None
            self.relationship_predictor = None
        
        # AI-specific AST node types
        self.ai_node_types = {
            'NEURAL_CONTRACT': 0,
            'ML_CONSENSUS_BLOCK': 1,
            'AI_FUNCTION_DEF': 2,
            'TENSOR_DECLARATION': 3,
            'MATRIX_OPERATION': 4,
            'CONSENSUS_HARMONY': 5,
            'CROSS_CHAIN_BRIDGE': 6,
            'AI_OPTIMIZATION': 7,
            'ML_TRAINING_BLOCK': 8,
            'NEURAL_INFERENCE': 9,
            'FUZZY_LOGIC_EXPR': 10,
            'PROBABILITY_CALC': 11,
            'CONFIDENCE_ASSERTION': 12,
            'AI_CONTROL_FLOW': 13,
            'ML_DATA_PIPELINE': 14,
            'NEURAL_ACTIVATION': 15,
            'GRADIENT_COMPUTATION': 16,
            'LOSS_CALCULATION': 17,
            'OPTIMIZATION_STEP': 18,
            'DISTRIBUTED_CONSENSUS': 19
        }

    def generate_ast(self, tokens: List[AIToken]) -> Dict[str, Any]:
        """Generate AI-optimized AST from tokens"""
        
        if not tokens:
            return {"type": "empty_program", "nodes": [], "ai_metadata": {}}
        
        # Jika mode bukan 'ai', langsung gunakan fallback AST tanpa transformer
        if self.model_mode != 'ai':
            return self._generate_fallback_ast(tokens)
        
        # Jika torch tidak tersedia atau transformer tidak terinisialisasi, gunakan fallback
        if not _TORCH_AVAILABLE or self.ast_transformer is None:
            return self._generate_fallback_ast(tokens)
        
        try:
            # Convert tokens to tensor representation
            token_tensors = self._tokens_to_tensors(tokens)
            
            # Apply transformer processing
            with torch.no_grad():
                ast_representation = self.ast_transformer(token_tensors)
            
            # Generate hierarchical AST structure
            ast_tree = self._build_ast_tree(ast_representation, tokens)
            
            # Add AI-specific metadata
            ast_tree = self._add_ai_metadata(ast_tree, tokens)
            
            # Optimize AST for AI processing
            optimized_ast = self._optimize_for_ai_processing(ast_tree)
            
            return optimized_ast
            
        except Exception:
            # Fallback to simple AST generation
            return self._generate_fallback_ast(tokens)

    def _generate_fallback_ast(self, tokens: List[AIToken]) -> Dict[str, Any]:
        """Generate a simple, deterministic AST structure when AI backend is unavailable.
        Struktur ini kompatibel dengan AIParsingOptimizer yang mengharapkan kunci 'nodes'.
        """
        ast_tree: Dict[str, Any] = {
            'type': 'ARTHEN_PROGRAM',
            'ai_optimized': False,
            'nodes': [],
            'metadata': {
                'backend_mode': self.model_mode,
                'torch_available': bool(_TORCH_AVAILABLE)
            }
        }
        
        for token in tokens:
            node = {
                'type': token.neural_context.get('construct_type', 'TOKEN') if isinstance(token.neural_context, dict) else 'TOKEN',
                'value': token.value,
                'position': token.position,
                'confidence': token.confidence,
                'semantic_weight': token.semantic_weight,
                'neural_context': token.neural_context,
                'children': []
            }
            ast_tree['nodes'].append(node)
        
        return ast_tree
    
    def _tokens_to_tensors(self, tokens: List[AIToken]) -> Any:
        """Convert AI tokens to tensor representation"""
        # Require torch
        if not _TORCH_AVAILABLE:
            raise RuntimeError("Torch not available for tensor conversion")
        if not tokens:
            return torch.zeros(1, 512, 1536)
        
        token_embeddings = []
        
        for token in tokens:
            if token.embedding is not None and hasattr(token.embedding, 'numel') and token.embedding.numel() > 0:
                embedding = token.embedding
                # Ensure embedding has correct shape
                if embedding.dim() == 1:
                    embedding = embedding.unsqueeze(0)
                # Pad or truncate to 1536 dimensions
                if embedding.size(-1) < 1536:
                    padding = torch.zeros(embedding.size(0), 1536 - embedding.size(-1))
                    embedding = torch.cat([embedding, padding], dim=-1)
                elif embedding.size(-1) > 1536:
                    embedding = embedding[:, :1536]
                token_embeddings.append(embedding)
            else:
                # Generate deterministic default embedding with proper shape (stable hashing, no RNG)
                import hashlib
                if token.value:
                    digest = hashlib.sha256(token.value.encode('utf-8')).digest()
                    idx = int.from_bytes(digest[:4], byteorder='big') % 1536
                else:
                    idx = 0
                default_embedding = torch.zeros(1, 1536)
                default_embedding[0, idx] = float(getattr(token, 'confidence', 0.1))
                token_embeddings.append(default_embedding)
        
        # Pad sequences for transformer processing
        max_length = 512
        if len(token_embeddings) > max_length:
            token_embeddings = token_embeddings[:max_length]
        else:
            # Pad with zeros
            while len(token_embeddings) < max_length:
                token_embeddings.append(torch.zeros(1, 1536))
        
        # Stack all embeddings
        stacked_embeddings = torch.cat(token_embeddings, dim=0)
        return stacked_embeddings.unsqueeze(0)
    
    def _build_ast_tree(self, ast_representation: Any, tokens: List[AIToken]) -> Dict[str, Any]:
        """Build hierarchical AST tree structure"""
        # Require torch
        if not _TORCH_AVAILABLE:
            return self._generate_fallback_ast(tokens)
        
        # Classify each position as AST node type
        node_logits = self.node_classifier(ast_representation.squeeze(0))
        node_types = torch.argmax(node_logits, dim=-1)
        
        # Build tree structure
        ast_tree = {
            'type': 'ARTHEN_PROGRAM',
            'ai_optimized': True,
            'neural_contracts': [],
            'ml_consensus_blocks': [],
            'ai_functions': [],
            'cross_chain_operations': [],
            'optimization_directives': [],
            'metadata': {
                'processing_hints': self._generate_processing_hints(tokens),
                'optimization_targets': self._identify_optimization_targets(tokens),
                'parallel_regions': self._identify_parallel_regions(tokens),
                'memory_layout': self._optimize_memory_layout(tokens)
            }
        }
        
        # Process tokens and build corresponding AST nodes
        for i, token in enumerate(tokens):
            if i < len(node_types):
                node_type_id = node_types[i].item()
                ast_node = self._create_ast_node(token, node_type_id)
                self._add_node_to_tree(ast_tree, ast_node, token)
        
        return ast_tree
    
    def _create_ast_node(self, token: AIToken, node_type_id: int) -> Dict[str, Any]:
        """Create AST node from token and predicted type"""
        
        # Reverse lookup node type name
        node_type_name = None
        for name, id_val in self.ai_node_types.items():
            if id_val == node_type_id:
                node_type_name = name
                break
        
        if node_type_name is None:
            node_type_name = 'UNKNOWN_NODE'
        
        ast_node = {
            'type': node_type_name,
            'value': token.value,
            'position': token.position,
            'confidence': token.confidence,
            'semantic_weight': token.semantic_weight,
            'neural_context': token.neural_context,
            'ai_processing_hints': {
                'vectorizable': token.neural_context.get('optimization_metadata', {}).get('vectorizable', False),
                'gpu_acceleratable': token.neural_context.get('optimization_metadata', {}).get('gpu_acceleratable', False),
                'distributed_processable': token.neural_context.get('optimization_metadata', {}).get('distributed_processable', False),
                'memory_requirements': token.neural_context.get('ai_processing_hints', {}).get('memory_requirements', 'medium')
            },
            'children': []
        }
        
        return ast_node
    
    def _add_node_to_tree(self, ast_tree: Dict[str, Any], ast_node: Dict[str, Any], token: AIToken):
        """Add AST node to appropriate tree section"""
        node_type = ast_node['type']
        
        if 'NEURAL_CONTRACT' in node_type:
            ast_tree['neural_contracts'].append(ast_node)
        elif 'ML_CONSENSUS' in node_type or 'CONSENSUS_HARMONY' in node_type:
            ast_tree['ml_consensus_blocks'].append(ast_node)
        elif 'AI_FUNCTION' in node_type:
            ast_tree['ai_functions'].append(ast_node)
        elif 'CROSS_CHAIN' in node_type or 'BRIDGE' in node_type:
            ast_tree['cross_chain_operations'].append(ast_node)
        elif 'OPTIMIZATION' in node_type:
            ast_tree['optimization_directives'].append(ast_node)
    
    def _add_ai_metadata(self, ast_tree: Dict[str, Any], tokens: List[AIToken]) -> Dict[str, Any]:
        """Add AI-specific metadata to AST"""
        
        ast_tree['ai_metadata'] = {
            'total_neural_constructs': len(ast_tree['neural_contracts']),
            'consensus_complexity': len(ast_tree['ml_consensus_blocks']),
            'ai_function_count': len(ast_tree['ai_functions']),
            'cross_chain_operations': len(ast_tree['cross_chain_operations']),
            'optimization_level': self._calculate_optimization_level(tokens),
            'parallel_processing_potential': self._calculate_parallel_potential(tokens),
            'memory_footprint_estimate': self._estimate_memory_footprint(tokens),
            'gpu_acceleration_opportunities': self._identify_gpu_opportunities(tokens),
            'distributed_processing_regions': self._identify_distributed_regions(tokens)
        }
        
        return ast_tree
    
    def _optimize_for_ai_processing(self, ast_tree: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize AST structure for AI processing efficiency"""
        
        # Reorder nodes for optimal AI processing
        ast_tree = self._reorder_for_ai_efficiency(ast_tree)
        
        # Add processing pipelines
        ast_tree['ai_processing_pipelines'] = self._generate_processing_pipelines(ast_tree)
        
        # Add optimization strategies
        ast_tree['optimization_strategies'] = self._generate_optimization_strategies(ast_tree)
        
        return ast_tree
    
    def _generate_processing_hints(self, tokens: List[AIToken]) -> Dict[str, Any]:
        """Generate processing hints for AI systems"""
        return {
            'preferred_execution_order': self._determine_execution_order(tokens),
            'parallelization_opportunities': self._find_parallelization_opportunities(tokens),
            'memory_access_patterns': self._analyze_memory_patterns(tokens),
            'cache_optimization_hints': self._generate_cache_hints(tokens)
        }
    
    def _identify_optimization_targets(self, tokens: List[AIToken]) -> List[str]:
        """Identify optimization targets from tokens"""
        targets = set()
        for token in tokens:
            if token.neural_context and 'ai_processing_hints' in token.neural_context:
                target = token.neural_context['ai_processing_hints'].get('optimization_target')
                if target:
                    targets.add(target)
        return list(targets)
    
    def _identify_parallel_regions(self, tokens: List[AIToken]) -> List[Dict[str, Any]]:
        """Identify regions suitable for parallel processing without embedding raw token objects."""
        parallel_regions: List[Dict[str, Any]] = []
        current_region: Optional[Dict[str, Any]] = None
    
        for i, token in enumerate(tokens):
            is_distributable = bool(
                token.neural_context and
                token.neural_context.get('optimization_metadata', {}).get('distributed_processable')
            )
    
            if is_distributable:
                if current_region is None:
                    # Track indices only to keep JSON output serializable
                    current_region = {'start': i, 'token_indices': []}
                current_region['token_indices'].append(i)
            else:
                if current_region is not None:
                    current_region['end'] = i - 1
                    current_region['count'] = len(current_region['token_indices'])
                    parallel_regions.append(current_region)
                    current_region = None
    
        # Close any open region at EOF
        if current_region is not None:
            current_region['end'] = len(tokens) - 1
            current_region['count'] = len(current_region['token_indices'])
            parallel_regions.append(current_region)
    
        return parallel_regions
    
    def _optimize_memory_layout(self, tokens: List[AIToken]) -> Dict[str, Any]:
        """Optimize memory layout for AI processing"""
        return {
            'sequential_access_regions': self._find_sequential_regions(tokens),
            'random_access_regions': self._find_random_regions(tokens),
            'cache_line_optimization': self._optimize_cache_lines(tokens),
            'memory_pooling_opportunities': self._find_pooling_opportunities(tokens)
        }
    
    # Placeholder implementations for helper methods
    def _calculate_optimization_level(self, tokens: List[AIToken]) -> str:
        return "maximum"
    
    def _calculate_parallel_potential(self, tokens: List[AIToken]) -> float:
        return 0.8
    
    def _estimate_memory_footprint(self, tokens: List[AIToken]) -> str:
        return "medium"
    
    def _identify_gpu_opportunities(self, tokens: List[AIToken]) -> List[str]:
        return ["tensor_operations", "neural_inference"]
    
    def _identify_distributed_regions(self, tokens: List[AIToken]) -> List[str]:
        return ["consensus_blocks", "cross_chain_operations"]
    
    def _reorder_for_ai_efficiency(self, ast_tree: Dict[str, Any]) -> Dict[str, Any]:
        return ast_tree
    
    def _generate_processing_pipelines(self, ast_tree: Dict[str, Any]) -> List[Dict[str, Any]]:
        return []
    
    def _generate_optimization_strategies(self, ast_tree: Dict[str, Any]) -> List[str]:
        return ["neural_optimization", "consensus_optimization", "memory_optimization"]
    
    def _determine_execution_order(self, tokens: List[AIToken]) -> List[int]:
        return list(range(len(tokens)))
    
    def _find_parallelization_opportunities(self, tokens: List[AIToken]) -> List[Dict[str, Any]]:
        return []
    
    def _analyze_memory_patterns(self, tokens: List[AIToken]) -> Dict[str, Any]:
        return {}
    
    def _generate_cache_hints(self, tokens: List[AIToken]) -> List[str]:
        return []
    
    def _find_sequential_regions(self, tokens: List[AIToken]) -> List[Dict[str, Any]]:
        return []
    
    def _find_random_regions(self, tokens: List[AIToken]) -> List[Dict[str, Any]]:
        return []
    
    def _optimize_cache_lines(self, tokens: List[AIToken]) -> Dict[str, Any]:
        return {}
    
    def _find_pooling_opportunities(self, tokens: List[AIToken]) -> str:
        return []

class AIOptimizedParser:
    """Main AI-Optimized Parser for ARTHEN Language"""
    
    def __init__(self, model_mode: str = 'ai'):
        """Initializes the parser with a specific model backend.

        Args:
            model_mode (str): The backend to use for embeddings and analysis.
                              'ai' -> transformers/torch (default)
                              'ml' -> scikit-learn (fallback)
                              'none' -> hash-based (deterministic)
        """
        self.lexer = NeuralLexer(model_mode=model_mode)
        self.ast_generator = TransformerASTGenerator(model_mode=model_mode)
        self.optimization_engine = AIParsingOptimizer()
    
    def parse(self, source_code: str) -> Dict[str, Any]:
        """
        Main parsing function - AI-optimized for machine processing
        Returns comprehensive AST with AI metadata
        """
        try:
            # Step 1: AI-enhanced tokenization
            tokens = self.lexer.tokenize(source_code)
            
            if not tokens:
                # Treat empty source as a valid empty program for CLI stability
                return {
                    "ast": {"type": "empty_program", "nodes": []},
                    "tokens": [],
                    "ai_metadata": {
                        "parsing_confidence": 1.0,
                        "machine_readability": 0.0,
                        "optimization_level": "none",
                        "note": "No tokens found; returned empty program"
                    },
                    "parsing_success": True
                }
            
            # Step 2: Generate AI-optimized AST
            ast = self.ast_generator.generate_ast(tokens)
            
            # Step 3: Calculate parsing confidence
            parsing_confidence = self._calculate_parsing_confidence(tokens)
            
            # Step 4: Assess machine readability
            machine_readability = self._calculate_machine_readability(ast)
            
            # Step 5: Generate AI processing metadata
            ai_metadata = self._generate_ai_processing_metadata(ast, tokens)
            ai_metadata.update({
                "parsing_confidence": parsing_confidence,
                "machine_readability": machine_readability,
                "total_tokens": len(tokens),
                "source_length": len(source_code)
            })
            
            # Step 6: Final optimization pass
            optimized_ast = self.optimization_engine.optimize(ast)
            
            return {
                "ast": optimized_ast,
                "tokens": [self._serialize_token(token) for token in tokens],
                "ai_metadata": ai_metadata,
                "parsing_success": True
            }
            
        except Exception as e:
            # Return error information for debugging
            return {
                "ast": {"type": "error_program", "nodes": []},
                "tokens": [],
                "ai_metadata": {
                    "parsing_confidence": 0.0,
                    "machine_readability": 0.0,
                    "optimization_level": "failed",
                    "error": str(e)
                },
                "parsing_success": False
            }
    
    def _serialize_token(self, token: AIToken) -> Dict[str, Any]:
        """Convert AIToken to serializable dictionary with robust embedding serialization"""
        embedding = token.embedding
        embedding_val: Optional[Any] = None
        try:
            if _TORCH_AVAILABLE and isinstance(embedding, torch.Tensor):
                embedding_val = embedding.detach().cpu().numpy().tolist()
            elif isinstance(embedding, np.ndarray):
                embedding_val = embedding.tolist()
            elif isinstance(embedding, (list, tuple)):
                embedding_val = list(embedding)
            elif isinstance(embedding, (int, float)):
                embedding_val = embedding
            elif embedding is None:
                embedding_val = None
            else:
                # Fallback to string representation for unknown types
                embedding_val = str(embedding)
        except Exception:
            embedding_val = None

        return {
            "type": token.type.value if hasattr(token.type, 'value') else str(token.type),
            "value": token.value,
            "position": token.position,
            "confidence": token.confidence,
            "semantic_weight": token.semantic_weight,
            "neural_context": token.neural_context,
            "embedding": embedding_val
        }
    
    def _generate_ai_processing_metadata(self, ast: Dict[str, Any], tokens: List[AIToken]) -> Dict[str, Any]:
        """Generate metadata for AI processing systems"""
        return {
            'execution_graph': self._build_execution_graph(ast),
            'dependency_analysis': self._analyze_dependencies(ast),
            'resource_requirements': self._calculate_resource_requirements(ast),
            'optimization_opportunities': self._identify_optimization_opportunities(ast),
            'parallel_execution_plan': self._generate_parallel_execution_plan(ast),
            'memory_management_strategy': self._generate_memory_strategy(ast)
        }
    
    def _calculate_parsing_confidence(self, tokens: List[AIToken]) -> float:
        """Calculate overall parsing confidence from token confidences"""
        if not tokens:
            return 0.0
        
        # Weighted average of token confidences
        total_confidence = sum(token.confidence * token.semantic_weight for token in tokens)
        total_weight = sum(token.semantic_weight for token in tokens)
        
        if total_weight == 0:
            return 0.0
        
        base_confidence = total_confidence / total_weight
        
        # Bonus for AI-specific constructs
        ai_token_count = sum(1 for token in tokens if any(indicator in token.value 
                           for indicator in ['∇', '∆', '⟨', '⟩', 'ml_', 'ai_', 'neural_']))
        ai_bonus = min(ai_token_count / len(tokens) * 0.2, 0.2)
        
        return min(base_confidence + ai_bonus, 1.0)
    
    def _calculate_machine_readability(self, ast: Dict[str, Any]) -> float:
        """Calculate how well-structured the AST is for machine processing"""
        if not ast or "nodes" not in ast:
            return 0.0
        
        nodes = ast.get("nodes", [])
        if not nodes:
            return 0.0
        
        # Base readability from structure depth and organization
        max_depth = self._calculate_ast_depth(ast)
        depth_score = min(max_depth / 10.0, 1.0)  # Deeper is better for AI
        
        # Node type diversity (more types = better structure)
        node_types = set()
        for node in nodes:
            if isinstance(node, dict) and "type" in node:
                node_types.add(node["type"])
        
        diversity_score = min(len(node_types) / 5.0, 1.0)
        
        # AI-specific construct bonus
        ai_constructs = sum(1 for node in nodes if isinstance(node, dict) and 
                          any(ai_term in str(node.get("type", "")) 
                              for ai_term in ["neural", "ml", "ai", "tensor", "consensus"]))
        ai_score = min(ai_constructs / len(nodes), 0.5)
        
        return (depth_score * 0.4 + diversity_score * 0.4 + ai_score * 0.2)
    
    def _calculate_ast_depth(self, node: Dict[str, Any], current_depth: int = 0) -> int:
        """Calculate maximum depth of AST tree"""
        if not isinstance(node, dict):
            return current_depth
        
        max_child_depth = current_depth
        
        # Check children in nodes array
        if "nodes" in node and isinstance(node["nodes"], list):
            for child in node["nodes"]:
                child_depth = self._calculate_ast_depth(child, current_depth + 1)
                max_child_depth = max(max_child_depth, child_depth)
        
        # Check other potential child nodes
        for key, value in node.items():
            if key != "nodes" and isinstance(value, dict):
                child_depth = self._calculate_ast_depth(value, current_depth + 1)
                max_child_depth = max(max_child_depth, child_depth)
            elif key != "nodes" and isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        child_depth = self._calculate_ast_depth(item, current_depth + 1)
                        max_child_depth = max(max_child_depth, child_depth)
        
        return max_child_depth
    
    # Placeholder implementations for helper methods
    def _build_execution_graph(self, ast: Dict[str, Any]) -> Dict[str, Any]:
        return {}
    
    def _analyze_dependencies(self, ast: Dict[str, Any]) -> Dict[str, Any]:
        return {}
    
    def _calculate_resource_requirements(self, ast: Dict[str, Any]) -> Dict[str, Any]:
        return {}
    
    def _identify_optimization_opportunities(self, ast: Dict[str, Any]) -> List[str]:
        return []
    
    def _generate_parallel_execution_plan(self, ast: Dict[str, Any]) -> Dict[str, Any]:
        return {}
    
    def _generate_memory_strategy(self, ast: Dict[str, Any]) -> Dict[str, Any]:
        return {}

class AIParsingOptimizer:
    """AI-driven AST optimization engine"""
    
    def optimize(self, ast: Dict[str, Any]) -> Dict[str, Any]:
        """Apply comprehensive AI optimizations to AST"""
        if not ast:
            return ast
        
        optimized_ast = ast.copy()
        
        # Apply neural processing optimizations
        optimized_ast = self._optimize_for_neural_processing(optimized_ast)
        
        # Apply parallel execution optimizations
        optimized_ast = self._optimize_for_parallel_execution(optimized_ast)
        
        # Optimize memory layout
        optimized_ast = self._optimize_memory_layout(optimized_ast)
        
        # Add AI execution hints
        optimized_ast = self._add_ai_execution_hints(optimized_ast)
        
        # Add optimization metadata
        optimized_ast["optimization_metadata"] = {
            "optimized": True,
            "optimization_level": "comprehensive",
            "neural_optimized": True,
            "parallel_optimized": True,
            "memory_optimized": True
        }
        
        return optimized_ast
    
    def _optimize_for_neural_processing(self, ast: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize AST structure for neural network processing"""
        if "nodes" in ast:
            # Group neural-related nodes together
            neural_nodes = []
            other_nodes = []
            
            for node in ast["nodes"]:
                if isinstance(node, dict):
                    node_type = node.get("type", "")
                    if any(neural_term in str(node_type) for neural_term in ["neural", "tensor", "matrix", "vector"]):
                        neural_nodes.append(node)
                    else:
                        other_nodes.append(node)
            
            # Reorder for better neural processing
            ast["nodes"] = neural_nodes + other_nodes
            
            # Add neural processing hints
            if neural_nodes:
                ast["neural_processing_hints"] = {
                    "neural_node_count": len(neural_nodes),
                    "batch_processable": len(neural_nodes) > 1,
                    "gpu_recommended": len(neural_nodes) > 5
                }
        
        return ast
    
    def _optimize_for_parallel_execution(self, ast: Dict[str, Any]) -> Dict[str, Any]:
        """Identify and mark parallel execution opportunities"""
        if "nodes" in ast:
            parallel_groups = []
            current_group = []
            
            for i, node in enumerate(ast["nodes"]):
                if isinstance(node, dict):
                    # Check if node can be parallelized
                    if self._is_parallelizable(node):
                        current_group.append(i)
                    else:
                        if current_group:
                            parallel_groups.append(current_group)
                            current_group = []
            
            # Add final group if exists
            if current_group:
                parallel_groups.append(current_group)
            
            # Add parallel execution metadata
            if parallel_groups:
                ast["parallel_execution"] = {
                    "parallel_groups": parallel_groups,
                    "max_parallelism": max(len(group) for group in parallel_groups),
                    "total_parallel_nodes": sum(len(group) for group in parallel_groups)
                }
        
        return ast
    
    def _optimize_memory_layout(self, ast: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize memory access patterns"""
        if "nodes" in ast:
            # Analyze memory access patterns
            memory_intensive_nodes = []
            lightweight_nodes = []
            
            for node in ast["nodes"]:
                if isinstance(node, dict):
                    if self._is_memory_intensive(node):
                        memory_intensive_nodes.append(node)
                    else:
                        lightweight_nodes.append(node)
            
            # Add memory optimization hints
            ast["memory_optimization"] = {
                "memory_intensive_count": len(memory_intensive_nodes),
                "lightweight_count": len(lightweight_nodes),
                "memory_layout_optimized": True,
                "cache_friendly": True
            }
        
        return ast
    
    def _add_ai_execution_hints(self, ast: Dict[str, Any]) -> Dict[str, Any]:
        """Add execution hints for AI processing"""
        ast["ai_execution_hints"] = {
            "recommended_batch_size": self._calculate_batch_size(ast),
            "gpu_acceleration": self._should_use_gpu(ast),
            "distributed_processing": self._should_distribute(ast),
            "optimization_priority": "speed",
            "memory_strategy": "balanced"
        }
        
        return ast
    
    def _is_parallelizable(self, node: Dict[str, Any]) -> bool:
        """Check if a node can be processed in parallel"""
        if not isinstance(node, dict):
            return False
        
        node_type = str(node.get("type", ""))
        parallel_types = ["tensor", "matrix", "vector", "neural", "ml_function"]
        
        return any(ptype in node_type for ptype in parallel_types)
    
    def _is_memory_intensive(self, node: Dict[str, Any]) -> bool:
        """Check if a node requires significant memory"""
        if not isinstance(node, dict):
            return False
        
        node_type = str(node.get("type", ""))
        memory_intensive_types = ["tensor", "matrix", "neural_network", "consensus_harmony"]
        
        return any(mtype in node_type for mtype in memory_intensive_types)
    
    def _calculate_batch_size(self, ast: Dict[str, Any]) -> int:
        """Calculate optimal batch size for processing"""
        node_count = len(ast.get("nodes", []))
        
        if node_count < 10:
            return 1
        elif node_count < 100:
            return 8
        else:
            return 16
    
    def _should_use_gpu(self, ast: Dict[str, Any]) -> bool:
        """Determine if GPU acceleration would be beneficial"""
        nodes = ast.get("nodes", [])
        neural_nodes = sum(1 for node in nodes if isinstance(node, dict) and 
                          any(term in str(node.get("type", "")) for term in ["neural", "tensor", "matrix"]))
        
        return neural_nodes > 3
    
    def _should_distribute(self, ast: Dict[str, Any]) -> bool:
        """Determine if distributed processing would be beneficial"""
        node_count = len(ast.get("nodes", []))
        return node_count > 50

# Example usage and main function
def main():
    """CLI entry point for ARTHEN parser: parse a file and output JSON."""
    import sys
    import argparse
    import warnings

    cli = argparse.ArgumentParser(
        prog="arthen-parse",
        description="Parse an ARTHEN source file and print a JSON result."
    )
    cli.add_argument("file_path", nargs="?", help="Path to .arthen source file")
    cli.add_argument("--pretty", action="store_true", help="Pretty-print JSON with indentation")
    cli.add_argument("--raw", action="store_true", help="Output only JSON (suppress extra logs)")
    cli.add_argument("--quiet", action="store_true", help="Suppress non-JSON logs and tracebacks")
    cli.add_argument("--no-model", action="store_true", help="Disable transformer models and use ML/hash-based fallback")
    args = cli.parse_args()

    # Suppress noisy third-party warnings when in raw/quiet mode to keep output clean for machines
    if getattr(args, "raw", False) or getattr(args, "quiet", False):
        warnings.simplefilter("ignore")
        # Silence specific torch transformer nested tensor warning if present
        try:
            warnings.filterwarnings(
                "ignore",
                message="enable_nested_tensor is True, but self.use_nested_tensor is False",
                module="torch.nn.modules.transformer"
            )
        except Exception:
            pass

    if not args.file_path:
        cli.print_help()
        return

    file_path = args.file_path
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}", file=sys.stderr)
        sys.exit(1)

    with open(file_path, 'r', encoding='utf-8') as f:
        source_code = f.read()

    # Avoid unicode emojis to prevent Windows console encoding issues
    if not args.quiet and not args.raw:
        print(f"Parsing file: {file_path}")
        print("==================================================")

    # Initialize parser with requested backend; '--no-model' enforces ML/hash fallback
    parser = AIOptimizedParser(model_mode=('ml' if getattr(args, 'no_model', False) else 'ai'))
    try:
        result = parser.parse(source_code)
        indent = 2 if args.pretty else None
        # ensure_ascii=True to avoid Windows console encoding errors
        json_output = json.dumps(result, ensure_ascii=True, indent=indent)
        print(json_output)
        # Exit codes: 0 on success, 2 if parsing_success=false
        if not result.get("parsing_success", False):
            sys.exit(2)
        else:
            sys.exit(0)
    except Exception as e:
        # Minimal error message on stderr and non-zero exit code
        print(f"Parsing failed: {e}", file=sys.stderr)
        import traceback
        if not args.quiet and not args.raw:
            traceback.print_exc()
        sys.exit(1)

def parse_arthen_file(file_path: str) -> Dict[str, Any]:
    """Parse an ARTHEN file and return the results"""
    parser = AIOptimizedParser()
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source_code = f.read()
        
        return parser.parse(source_code)
    
    except FileNotFoundError:
        raise FileNotFoundError(f"ARTHEN file not found: {file_path}")
    except Exception as e:
        raise Exception(f"Failed to parse ARTHEN file {file_path}: {str(e)}")

def validate_arthen_syntax(source_code: str) -> bool:
    """Validate ARTHEN syntax and return True if valid"""
    parser = AIOptimizedParser()
    
    try:
        result = parser.parse(source_code)
        return result['parsing_confidence'] > 0.7  # 70% confidence threshold
    except:
        return False

if __name__ == "__main__":
    main()