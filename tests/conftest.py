import os
import random
import sys

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


def pytest_configure(config):
    """Register custom markers in case pytest.ini is not discovered."""
    config.addinivalue_line("markers", "unit: Unit tests for individual components")
    config.addinivalue_line("markers", "integration: Integration tests for component interaction")
    config.addinivalue_line("markers", "performance: Performance and benchmark tests")
    config.addinivalue_line("markers", "ai: AI and ML functionality tests")
    config.addinivalue_line("markers", "blockchain: Blockchain-specific tests")
    config.addinivalue_line("markers", "security: Security and vulnerability tests")
    config.addinivalue_line("markers", "slow: Tests that take a long time to run")
    config.addinivalue_line("markers", "network: Tests that require network access")
    config.addinivalue_line("markers", "gpu: Tests that require GPU acceleration")


def pytest_sessionstart(session):
    """Configure deterministic execution and offline mode before tests start.
    Also ensure source tree is preferred over build artifacts for imports.
    """
    # Prefer project source on sys.path
    project_root = os.path.dirname(os.path.dirname(__file__))
    build_lib = os.path.join(project_root, 'build', 'lib')
    dist_dir = os.path.join(project_root, 'dist')
    # Remove build/dist paths if present to avoid importing stale modules
    sys.path = [p for p in sys.path if p not in (build_lib, dist_dir)]
    # Ensure project root is at the front of sys.path
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    # Also set PYTHONPATH for child processes
    os.environ.setdefault('PYTHONPATH', project_root)

    # Offline/test gating to avoid model/network access
    os.environ.setdefault('ARTHEN_TEST_MODE', 'true')
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    # Ensure child processes inherit deterministic hashing
    os.environ.setdefault('PYTHONHASHSEED', '0')
    # Constrain threads to reduce nondeterminism
    os.environ.setdefault('OMP_NUM_THREADS', '1')

    # Seed Python and NumPy RNG
    random.seed(0)
    if np is not None:
        np.random.seed(0)

    # Seed Torch RNG and set deterministic flags when available
    if torch is not None:
        try:
            torch.manual_seed(0)
            # Prefer CPU determinism; disable benchmarking for cudnn if present
            if hasattr(torch.backends, 'cudnn'):
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            # Some ops require this and may raise warnings if unsupported; keep best-effort
            if hasattr(torch, 'use_deterministic_algorithms'):
                try:
                    torch.use_deterministic_algorithms(True)
                except Exception:
                    pass
        except Exception:
            # If torch seeding fails, continue with Python/NumPy seeds
            pass