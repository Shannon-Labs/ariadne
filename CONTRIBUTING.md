# Contributing to Ariadne 🔮

Thank you for your interest in contributing to Ariadne! We're building the future of quantum threat detection, and we welcome contributions from developers, security researchers, and quantum computing enthusiasts.

## 🚀 Quick Start

1. **Fork** the repository
2. **Clone** your fork: `git clone https://github.com/your-username/shannon-mono.git`
3. **Navigate** to Ariadne: `cd apps/ariadne`
4. **Create** a feature branch: `git checkout -b feature/amazing-quantum-feature`
5. **Install** dependencies: `pip install -e ".[dev]"`
6. **Make** your changes
7. **Test** your changes: `pytest`
8. **Submit** a pull request

## 📋 Contribution Guidelines

### Code Style
- Follow [PEP 8](https://pep8.org/) style guidelines
- Use type hints for all function signatures
- Write docstrings for all public functions and classes
- Keep line length under 88 characters
- Use descriptive variable and function names

### Testing
- Write tests for all new functionality
- Ensure all tests pass: `pytest`
- Include unit tests and integration tests where applicable
- Test edge cases and error conditions

### Documentation
- Update docstrings for any modified functions
- Add examples to the `examples/` directory for new features
- Update the README if adding new functionality

## 🔍 What We're Looking For

### High-Impact Contributions
- **Quantum Attack Detection**: New detection algorithms and methods
- **Performance Optimizations**: Faster anomaly detection
- **Integration Features**: Better CbAD integration
- **Security Research**: Novel quantum threat patterns
- **Enterprise Features**: Production-ready capabilities

### Bug Fixes
- Security vulnerabilities
- Performance issues
- Integration problems
- Documentation errors

## 🛠 Development Setup

### Prerequisites
- Python 3.8+
- pip or conda for package management
- Git for version control

### Environment Setup
```bash
# Clone the repository
git clone https://github.com/Shannon-Labs/shannon-mono.git
cd apps/ariadne

# Create virtual environment
python -m venv ariadne-env
source ariadne-env/bin/activate  # On Windows: ariadne-env\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=ariadne

# Run specific test file
pytest tests/test_quantum_detector.py

# Run only unit tests
pytest -m unit

# Run integration tests
pytest -m integration
```

### Code Quality
```bash
# Format code
black ariadne/ tests/

# Sort imports
isort ariadne/ tests/

# Type checking
mypy ariadne/

# Lint code
ruff check ariadne/ tests/

# Fix linting issues
ruff check --fix ariadne/ tests/
```

## 📝 Pull Request Process

1. **Update** the CHANGELOG.md with your changes
2. **Write** clear commit messages following [Conventional Commits](https://conventionalcommits.org/)
3. **Reference** any related issues in your PR description
4. **Ensure** all tests pass
5. **Request** review from maintainers

### PR Template
```
## Description
[Brief description of changes]

## Related Issues
Fixes #123
Closes #456

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
- [ ] Tests pass
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
```

## 🏗 Project Structure

```
apps/ariadne/
├── ariadne/           # Main package
│   ├── __init__.py
│   ├── quantum_detector.py    # Core detection logic
│   ├── cbad_integration.py    # CbAD integration
│   ├── api.py                 # REST API
│   └── cli.py                 # Command line interface
├── tests/             # Test files
├── examples/          # Example scripts
├── docs/              # Documentation
└── configs/           # Configuration files
```

## 🔐 Security Considerations

When contributing security-related code:

- Follow secure coding practices
- Avoid introducing new attack vectors
- Consider timing attacks and side-channel vulnerabilities
- Document any security assumptions
- Include security tests

## 📄 License

By contributing, you agree that your contributions will be licensed under the same license as the original project (MIT).

## 🤝 Community

- **Discussions**: Use GitHub Discussions for questions and ideas
- **Issues**: Use GitHub Issues for bug reports and feature requests
- **Security**: Report security vulnerabilities to security@shannonlabs.com

## 🙏 Acknowledgments

Thank you for helping us build the future of quantum security! Your contributions help make Ariadne the leading quantum threat detection platform.

---

**Shannon Labs** - Building quantum security for the classical world 🚀