# Contributing to Brain-Inspired AI

Thank you for your interest in contributing to Brain-Inspired AI! This document provides guidelines and information for contributors.

## 🎯 Table of Contents

- [Code of Conduct](#-code-of-conduct)
- [Getting Started](#-getting-started)
- [Development Setup](#-development-setup)
- [Contribution Guidelines](#-contribution-guidelines)
- [Coding Standards](#-coding-standards)
- [Testing](#-testing)
- [Documentation](#-documentation)
- [Pull Request Process](#-pull-request-process)
- [Issue Reporting](#-issue-reporting)
- [Community](#-community)

## 📋 Code of Conduct

This project adheres to a Code of Conduct that all contributors are expected to follow. Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before contributing.

## 🚀 Getting Started

### Ways to Contribute

- 🐛 **Bug Reports**: Help us identify and fix bugs
- 💡 **Feature Requests**: Suggest new features or improvements
- 🔧 **Code Contributions**: Submit bug fixes or new features
- 📖 **Documentation**: Improve docs, examples, and tutorials
- 🧪 **Testing**: Help us improve test coverage
- 🎨 **UI/UX**: Improve user interface and experience
- 🌐 **Translations**: Help translate documentation

### Quick Start

1. **Fork the repository**
   ```bash
   git clone https://github.com/your-username/brain-inspired-ai.git
   cd brain-inspired-ai
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # venv\Scripts\activate  # Windows
   ```

3. **Install development dependencies**
   ```bash
   pip install -e ".[dev,test,docs]"
   pre-commit install
   ```

4. **Run tests to verify setup**
   ```bash
   pytest tests/ --cov=brain_ai
   ```

## 🛠️ Development Setup

### Prerequisites

- Python 3.8 or higher
- Git
- Recommended: CUDA-compatible GPU (for full functionality)

### Environment Setup

```bash
# Clone with submodules
git clone --recursive https://github.com/your-username/brain-inspired-ai.git
cd brain-inspired-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install in development mode
pip install -e ".[dev,test,docs,all]"

# Install pre-commit hooks
pre-commit install

# Verify installation
python demo_quick_test.py
```

### Docker Development

```bash
# Build development container
docker build -f Dockerfile.dev -t brain-ai:dev .

# Run with volume mount for development
docker run -it -v $(pwd):/app brain-ai:dev bash

# Or use docker-compose
docker-compose -f docker-compose.dev.yml up -d
```

### IDE Setup

#### VS Code
```json
// .vscode/settings.json
{
  "python.defaultInterpreterPath": "./venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.flake8Enabled": true,
  "python.linting.mypyEnabled": true,
  "python.formatting.provider": "black",
  "python.sortImports.args": ["--profile", "black"],
  "files.exclude": {
    "**/__pycache__": true,
    "**/*.pyc": true,
    "**/.pytest_cache": true,
    "**/build": true,
    "**/dist": true
  }
}
```

## 📝 Contribution Guidelines

### Branch Naming Convention

- `feature/feature-name` - New features
- `bugfix/issue-description` - Bug fixes
- `docs/document-name` - Documentation changes
- `refactor/refactor-description` - Code refactoring
- `test/test-description` - Test improvements

### Commit Message Format

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
type(scope): description

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```bash
feat(hippocampus): add pattern completion algorithm
fix(memory): resolve memory leak in consolidation
docs(api): update hippocampus documentation
test(continual_learning): add EWC test coverage
```

## 🎨 Coding Standards

### Python Style Guide

We follow PEP 8 with some modifications:

```bash
# Code formatting
black src/ tests/ examples/
isort src/ tests/ examples/

# Linting
flake8 src/ tests/
mypy src/ --ignore-missing-imports

# Security scanning
bandit -r src/
```

### Type Hints

All public APIs must include type hints:

```python
from typing import Dict, List, Optional, Tuple
import numpy as np

def process_memory(
    data: np.ndarray,
    config: Dict[str, float],
    threshold: Optional[float] = None
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Process memory data with optional threshold."""
    pass
```

### Documentation Standards

All modules, classes, and functions must have docstrings:

```python
class HippocampusNetwork:
    """Hippocampus-inspired memory network.
    
    This class implements a neural network architecture inspired by
    the hippocampus brain region, providing memory formation,
    storage, and retrieval capabilities.
    
    Attributes:
        memory_capacity: Maximum number of memory items to store
        encoding_dim: Dimensionality of memory encoding
        retrieval_threshold: Minimum similarity for retrieval
        
    Example:
        >>> network = HippocampusNetwork(memory_capacity=10000)
        >>> memory = network.learn(data, labels)
        >>> retrieved = network.recall(query)
    """
```

### Error Handling

Use specific exceptions and proper error handling:

```python
from brain_ai.exceptions import MemoryCapacityError, InvalidInputError

def add_memory(self, item: MemoryItem) -> None:
    """Add item to memory."""
    if len(self.memories) >= self.memory_capacity:
        raise MemoryCapacityError(
            f"Memory capacity exceeded: {self.memory_capacity}"
        )
    
    if not isinstance(item, MemoryItem):
        raise InvalidInputError(
            f"Expected MemoryItem, got {type(item)}"
        )
```

## 🧪 Testing

### Test Structure

```
tests/
├── unit/           # Unit tests for individual components
├── integration/    # Integration tests for component interactions
├── performance/    # Performance and benchmark tests
├── fixtures/       # Shared test fixtures and data
└── conftest.py     # Pytest configuration
```

### Writing Tests

```python
import pytest
import numpy as np
from brain_ai import HippocampusNetwork

class TestHippocampusNetwork:
    """Test suite for HippocampusNetwork."""
    
    @pytest.fixture
    def network(self):
        """Fixture for network instance."""
        return HippocampusNetwork(memory_capacity=1000)
    
    @pytest.fixture
    def sample_data(self):
        """Fixture for sample data."""
        return np.random.rand(100, 784)
    
    def test_memory_capacity(self, network):
        """Test memory capacity enforcement."""
        # Test implementation
        pass
    
    def test_memory_retrieval(self, network, sample_data):
        """Test memory retrieval accuracy."""
        # Test implementation
        pass
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=brain_ai --cov-report=html

# Run specific test file
pytest tests/unit/test_hippocampus.py -v

# Run with markers
pytest -m "not slow"  # Skip slow tests
pytest -m "gpu"       # Run GPU tests only

# Run performance tests
pytest tests/performance/ --benchmark-only
```

### Performance Testing

```python
import pytest

@pytest.mark.performance
def test_memory_training_performance(network, benchmark_data):
    """Benchmark memory training performance."""
    result = pytest.benchmark(
        network.train,
        benchmark_data,
        epochs=10
    )
    assert result.stats.mean < 1.0  # Training should take < 1 second
```

## 📖 Documentation

### Documentation Types

1. **API Documentation**: Automatically generated from docstrings
2. **User Guide**: Tutorial-style documentation
3. **Examples**: Code examples and notebooks
4. **Research Papers**: Academic documentation

### Building Documentation

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build documentation
mkdocs build

# Serve locally for development
mkdocs serve

# Deploy to GitHub Pages
mkdocs gh-deploy
```

### Writing Examples

Examples should be:

- Complete and runnable
- Well-commented
- Demonstrate real-world usage
- Include expected output

```python
"""
Example: Basic Memory Learning

This example demonstrates how to use the hippocampus network
to learn and retrieve memories.
"""

import numpy as np
from brain_ai import HippocampusNetwork

# Create network
network = HippocampusNetwork(memory_capacity=1000)

# Prepare data
data = np.random.rand(100, 784)  # 100 images of 28x28
labels = np.random.randint(0, 10, 100)  # 10 classes

# Train network
network.train(data, labels, epochs=10)

# Retrieve memory
query = data[0]  # First image
retrieved = network.recall(query)

print(f"Retrieved with similarity: {retrieved.similarity:.3f}")
```

## 🔄 Pull Request Process

### Before Submitting

1. **Update tests** to cover your changes
2. **Run full test suite** to ensure no regressions
3. **Update documentation** as needed
4. **Check code style** compliance
5. **Add changelog entry** if applicable

### Pull Request Template

Ensure your PR includes:

- Clear description of changes
- Test coverage for new code
- Documentation updates
- Breaking changes section (if applicable)
- Migration guide (if breaking changes)

### Review Process

1. **Automated Checks**: All CI tests must pass
2. **Code Review**: At least one core team member approval
3. **Documentation Review**: Docs updated and accurate
4. **Performance Review**: No significant performance regressions
5. **Biological Accuracy**: Changes align with brain-inspired principles

### Review Checklist

**For Authors:**
- [ ] Tests pass locally and in CI
- [ ] Code style compliance checked
- [ ] Documentation updated
- [ ] Examples provided (if applicable)
- [ ] Changelog updated

**For Reviewers:**
- [ ] Code quality and style
- [ ] Test coverage adequacy
- [ ] Documentation completeness
- [ ] Performance impact
- [ ] Biological accuracy
- [ ] Breaking changes handling

## 🐛 Issue Reporting

### Bug Reports

Use the [bug report template](.github/ISSUE_TEMPLATE/bug_report.md) and include:

- Clear description of the issue
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, etc.)
- Minimal code example
- Error logs

### Feature Requests

Use the [feature request template](.github/ISSUE_TEMPLATE/feature_request.md) and include:

- Clear description of the requested feature
- Problem it solves
- Proposed implementation
- Alternative solutions considered
- Impact assessment

### Performance Issues

Use the [performance issue template](.github/ISSUE_TEMPLATE/performance_issue.md) and include:

- Current performance metrics
- Expected performance
- Environment details
- Benchmark code
- Profiling results

## 🤝 Community

### Getting Help

- **Discussions**: Use GitHub Discussions for questions
- **Issues**: Use GitHub Issues for bug reports
- **Discord**: Join our [Discord server](https://discord.gg/brain-ai)
- **Email**: team@brain-ai.org

### Community Guidelines

- Be respectful and welcoming
- Help others learn and grow
- Share knowledge and experiences
- Follow the Code of Conduct
- Constructive feedback only

### Recognition

Contributors are recognized in:

- [CONTRIBUTORS.md](CONTRIBUTORS.md)
- Release notes
- Project documentation
- Conference presentations

## 🎯 Development Roadmap

### Current Priorities

- [ ] **Memory Consolidation**: Improve consolidation algorithms
- [ ] **Performance**: Optimize training and inference speed
- [ ] **Documentation**: Expand tutorials and examples
- [ ] **Research**: Publish findings in top-tier venues

### How to Help

1. Check the [Roadmap](ROADMAP.md) for current priorities
2. Look for `good first issue` labels
3. Contribute to documentation
4. Help with community support
5. Share your use cases and feedback

## 📊 Contribution Stats

We track contributions to recognize our contributors:

- **Code Contributions**: Lines of code, commits, PRs
- **Documentation**: Documentation pages, examples
- **Testing**: Test coverage, performance benchmarks
- **Community**: Issues resolved, discussions helped

## 🏆 Contributor Levels

### 🌟 **Starter**: First contribution
- Made first PR or issue report
- Recognized in release notes

### 🔧 **Contributor**: Regular contributor  
- 5+ meaningful contributions
- Invited to contributor Discord channel
- Listed in contributors page

### 🚀 **Maintainer**: Core contributor
- 25+ meaningful contributions
- Code review permissions
- Direct repository access

### 👑 **Architect**: Project leader
- 100+ meaningful contributions
- Project governance role
- Strategic decision making

---

## 📞 Contact

- **Maintainers**: @maintainers
- **Email**: team@brain-ai.org
- **Discord**: https://discord.gg/brain-ai
- **Website**: https://brain-ai.org

**Thank you for contributing to Brain-Inspired AI! 🧠**

Your contributions help advance the field of brain-inspired artificial intelligence and make the technology accessible to researchers and developers worldwide.