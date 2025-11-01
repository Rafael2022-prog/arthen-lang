# Contributing to ARTHEN Programming Language

Terima kasih atas minat Anda untuk berkontribusi pada **ARTHEN** - AI-Native Programming Language for Blockchain Ecosystems! 🚀

## 📋 Daftar Isi

- [Code of Conduct](#code-of-conduct)
- [Cara Berkontribusi](#cara-berkontribusi)
- [Development Setup](#development-setup)
- [Struktur Proyek](#struktur-proyek)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Issue Guidelines](#issue-guidelines)
- [Community](#community)

## 🤝 Code of Conduct

Proyek ini mengikuti [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/). Dengan berpartisipasi, Anda diharapkan untuk menjunjung tinggi kode etik ini.

## 🛠️ Cara Berkontribusi

Ada banyak cara untuk berkontribusi pada ARTHEN:

### 1. 🐛 Melaporkan Bug
- Gunakan GitHub Issues untuk melaporkan bug
- Sertakan informasi detail tentang environment dan langkah reproduksi
- Gunakan template issue yang tersedia

### 2. 💡 Mengusulkan Fitur
- Diskusikan ide fitur baru di GitHub Discussions
- Buat RFC (Request for Comments) untuk fitur besar
- Sertakan use case dan contoh implementasi

### 3. 📝 Dokumentasi
- Perbaiki typo atau kesalahan dalam dokumentasi
- Tambahkan contoh penggunaan
- Terjemahkan dokumentasi ke bahasa lain

### 4. 💻 Kontribusi Kode
- Implementasi fitur baru
- Perbaikan bug
- Optimisasi performance
- Penambahan test coverage

### 5. 🧪 Testing
- Tulis test cases untuk fitur baru
- Perbaiki test yang gagal
- Tambahkan integration tests

## 🚀 Development Setup

### Prerequisites
```bash
# Python 3.8+
python --version

# Node.js 16+
node --version

# Git
git --version
```

### Setup Environment
```bash
# 1. Fork dan clone repository
git clone https://github.com/Rafael2022-prog/arthen-lang.git
cd arthen-lang

# 2. Install dependencies
make install-dev

# 3. Setup development environment
make setup

# 4. Verify installation
make test
```

### Development Workflow
```bash
# 1. Create feature branch
git checkout -b feature/amazing-feature

# 2. Make changes
# ... edit files ...

# 3. Run tests
make test

# 4. Format code
make format

# 5. Lint code
make lint

# 6. Commit changes
git commit -m "feat: add amazing feature"

# 7. Push to fork
git push origin feature/amazing-feature

# 8. Create Pull Request
```

## 📁 Struktur Proyek

```
ARTHEN-LANG/
├── compiler/           # Compiler dan code generation
│   ├── arthen_compiler_architecture.py
│   └── arthen_compiler.py
├── parser/            # AI-native parser dan lexer
│   └── ai_native_parser.py
├── stdlib/            # Standard library
│   ├── arthen_stdlib.arthen
│   └── arthen_stdlib_implementation.py
├── consensus/         # ML consensus mechanisms
│   └── ml_harmony_consensus.arthen
├── examples/          # Contoh proyek ARTHEN
│   ├── defi_liquidity_pool.arthen
│   ├── ai_governance_system.arthen
│   └── ...
├── tests/            # Test suite
├── docs/             # Dokumentasi
├── bin/              # CLI executables
└── scripts/          # Development scripts
```

## 📏 Coding Standards

### Python Code Style
- Gunakan **Black** untuk formatting
- Ikuti **PEP 8** guidelines
- Gunakan **type hints** untuk semua functions
- Docstrings menggunakan **Google style**

```python
def compile_arthen_code(
    source_code: str, 
    target_platform: str,
    optimization_level: int = 3
) -> CompilationResult:
    """Compile ARTHEN source code to target platform.
    
    Args:
        source_code: The ARTHEN source code to compile
        target_platform: Target blockchain platform (ethereum, solana, etc.)
        optimization_level: AI optimization level (1-5)
        
    Returns:
        CompilationResult containing compiled code and metadata
        
    Raises:
        CompilationError: If compilation fails
    """
    # Implementation here
    pass
```

### JavaScript/Node.js Code Style
- Gunakan **Prettier** untuk formatting
- Ikuti **ESLint** rules
- Gunakan **JSDoc** untuk dokumentasi

```javascript
/**
 * Deploy ARTHEN contract to blockchain network
 * @param {string} contractPath - Path to compiled contract
 * @param {string} network - Target network name
 * @param {Object} options - Deployment options
 * @returns {Promise<DeploymentResult>} Deployment result
 */
async function deployContract(contractPath, network, options = {}) {
    // Implementation here
}
```

### ARTHEN Code Style
- Gunakan mathematical notation yang konsisten
- Dokumentasikan AI functions dengan confidence levels
- Sertakan security annotations

```arthen
/**
 * AI-powered liquidity pool with ML consensus
 * @security: high
 * @ai_optimization: enabled
 */
∆contract AILiquidityPool {
    ⟨⟨calculateOptimalFee⟩⟩(∆u256 volume) -> Ω{0.95} ∆u256 {
        // AI implementation
    }
}
```

## 🧪 Testing Guidelines

### Test Structure
```
tests/
├── unit/              # Unit tests
├── integration/       # Integration tests
├── e2e/              # End-to-end tests
├── fixtures/         # Test data
└── utils/            # Test utilities
```

### Writing Tests

#### Python Tests (pytest)
```python
import pytest
from compiler.arthen_compiler import ARTHENCompiler

class TestARTHENCompiler:
    def test_compile_ethereum_contract(self):
        """Test compilation to Ethereum platform"""
        compiler = ARTHENCompiler()
        source = "∆contract Test { }"
        
        result = compiler.compile(source, target="ethereum")
        
        assert result.success
        assert "pragma solidity" in result.output
        
    @pytest.mark.asyncio
    async def test_ai_optimization(self):
        """Test AI optimization features"""
        # Test implementation
        pass
```

#### JavaScript Tests (Jest)
```javascript
const { deployContract } = require('../bin/arthen');

describe('ARTHEN Deployment', () => {
    test('should deploy to Ethereum testnet', async () => {
        const result = await deployContract(
            'test-contract.arthen',
            'ethereum-testnet'
        );
        
        expect(result.success).toBe(true);
        expect(result.contractAddress).toBeDefined();
    });
});
```

### Test Coverage
- Minimum 80% code coverage untuk semua modules
- 100% coverage untuk critical security functions
- Integration tests untuk semua blockchain platforms

## 📝 Pull Request Process

### 1. Pre-submission Checklist
- [ ] Code follows style guidelines
- [ ] Tests pass locally (`make test`)
- [ ] Code is properly formatted (`make format`)
- [ ] No linting errors (`make lint`)
- [ ] Documentation updated if needed
- [ ] CHANGELOG.md updated for significant changes

### 2. PR Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests pass
```

### 3. Review Process
1. **Automated Checks**: CI/CD pipeline runs tests
2. **Code Review**: Maintainers review code quality
3. **AI Review**: Automated AI code analysis
4. **Security Review**: Security-focused review for critical changes
5. **Final Approval**: Maintainer approval required

## 🐛 Issue Guidelines

### Bug Reports
Gunakan template berikut untuk bug reports:

```markdown
**Bug Description**
Clear description of the bug

**To Reproduce**
Steps to reproduce:
1. Go to '...'
2. Click on '....'
3. See error

**Expected Behavior**
What you expected to happen

**Environment**
- OS: [e.g. Windows 11]
- Python Version: [e.g. 3.9]
- ARTHEN Version: [e.g. 1.0.0]
- Node.js Version: [e.g. 18.0.0]

**Additional Context**
Any other context about the problem
```

### Feature Requests
```markdown
**Feature Description**
Clear description of the proposed feature

**Use Case**
Why is this feature needed?

**Proposed Solution**
How should this feature work?

**Alternatives Considered**
Other solutions you've considered

**Additional Context**
Any other context or screenshots
```

## 🏷️ Commit Message Guidelines

Gunakan [Conventional Commits](https://www.conventionalcommits.org/) format:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

### Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test additions/modifications
- `chore`: Maintenance tasks

### Examples:
```
feat(compiler): add Solana compilation target

Add support for compiling ARTHEN contracts to Solana programs
using Rust backend with AI optimization.

Closes #123
```

```
fix(parser): resolve AI tokenization issue

Fix issue where AI-native parser failed to tokenize mathematical
operators in complex expressions.

Fixes #456
```

## 🌟 Recognition

Kontributor akan diakui dalam:
- **CONTRIBUTORS.md** file
- **Release notes** untuk kontribusi signifikan
- **Hall of Fame** di website untuk kontributor reguler
- **Special badges** di GitHub profile

## 🎯 Areas Needing Help

Kami khususnya membutuhkan bantuan di area:

### 🔥 High Priority
- [ ] **AI Model Optimization**: Improve ML consensus algorithms
- [ ] **Cross-Chain Bridges**: Implement more bridge protocols
- [ ] **Security Analysis**: Enhance AI security scanner
- [ ] **Performance**: Optimize compilation speed

### 📚 Documentation
- [ ] **API Documentation**: Complete API reference
- [ ] **Tutorials**: Step-by-step guides
- [ ] **Examples**: More real-world examples
- [ ] **Translations**: Multi-language support

### 🧪 Testing
- [ ] **E2E Tests**: End-to-end testing suite
- [ ] **Blockchain Tests**: Test on more networks
- [ ] **Performance Tests**: Benchmarking suite
- [ ] **Security Tests**: Penetration testing

### 🌐 Ecosystem
- [ ] **IDE Plugins**: VS Code, IntelliJ extensions
- [ ] **Package Managers**: npm, pip integration
- [ ] **CI/CD**: GitHub Actions workflows
- [ ] **Docker**: Container support

## 💬 Community

### Communication Channels
- **GitHub Discussions**: Technical discussions
- **Discord**: Real-time chat ([Join here](https://discord.gg/arthen))
- **Telegram**: Community updates ([@ARTHENLang](https://t.me/ARTHENLang))
- **Twitter**: News and announcements ([@ARTHENLang](https://twitter.com/ARTHENLang))

### Community Guidelines
1. **Be Respectful**: Treat everyone with respect
2. **Be Constructive**: Provide helpful feedback
3. **Be Patient**: Remember everyone is learning
4. **Be Inclusive**: Welcome newcomers
5. **Be Professional**: Maintain professional standards

## 🎉 Getting Started

Ready to contribute? Here's how to get started:

1. **Join our Discord** untuk berkenalan dengan komunitas
2. **Browse Good First Issues** di GitHub
3. **Read the documentation** untuk memahami ARTHEN
4. **Set up development environment** menggunakan panduan di atas
5. **Pick an issue** dan mulai berkontribusi!

## 📞 Need Help?

Jika Anda membutuhkan bantuan:

1. **Check Documentation**: Baca dokumentasi terlebih dahulu
2. **Search Issues**: Cari di GitHub Issues
3. **Ask in Discord**: Tanyakan di channel #help
4. **Create Discussion**: Buat GitHub Discussion
5. **Contact Maintainers**: Email team@arthen-lang.org

---

**Terima kasih telah berkontribusi pada ARTHEN! Bersama-sama kita membangun masa depan blockchain development yang lebih cerdas.** 🚀

*Happy Coding!* 💻✨