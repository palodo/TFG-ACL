# Contributing to ACL Classifier

Thank you for your interest in contributing to the ACL Classifier project! This document provides guidelines and instructions for contributing.

## 🎯 Code of Conduct

We are committed to providing a welcoming and inspiring community for all. Please be respectful and constructive in all interactions.

## 🐛 Reporting Issues

Before creating a bug report, please check the issue list - your issue may already exist.

When reporting a bug, include:

- **Clear description**: What did you expect vs. what actually happened?
- **Reproduction steps**: Exact steps to reproduce the issue
- **Environment**: Python version, CUDA version, GPU model, OS
- **Logs/Error messages**: Full traceback if applicable
- **Minimal example**: Minimal code to reproduce the issue

## 💡 Suggesting Features

Great ideas are always welcome! Before suggesting a new feature:

1. Check existing issues to see if it's already proposed
2. Explain the use case and why it would be beneficial
3. Provide examples of how it would be used

## 🔧 Development Setup

### 1. Fork and Clone

```bash
git clone https://github.com/your-username/acl_classifier.git
cd acl_classifier
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or for bug fixes:
git checkout -b fix/bug-description
```

Use descriptive branch names:
- `feature/add-new-model` ✓
- `fix/gpu-memory-leak` ✓
- `docs/update-training-guide` ✓
- `my-branch` ✗

## 📝 Code Style

We follow PEP 8 with these guidelines:

### Python Code

```python
# Use type hints where possible
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    num_epochs: int = 100
) -> Dict[str, List[float]]:
    """
    Train model on provided data.
    
    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        num_epochs: Number of epochs to train
        
    Returns:
        Dictionary with training history
    """
    pass
```

### Formatting

- **Line length**: 88 characters (Black formatter standard)
- **Imports**: Group as `stdlib > third-party > local`
- **Docstrings**: Use Google style format

### Naming Conventions

- Functions/variables: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Private members: `_single_leading_underscore`

## 🧪 Testing

Before submitting, test your changes:

```bash
# Test imports
python -c "from src.models import ViTTinyMultiSliceClassifier"

# Test on GPU (if applicable)
python -c "import torch; print(torch.cuda.is_available())"
```

For notebook changes, verify:
- All cells run without errors
- Output is deterministic or documented
- Memory usage is reasonable

## 📤 Submitting Changes

### 1. Commit Guidelines

```bash
# Good commits
git commit -m "Add plane-specific augmentation strategies"
git commit -m "Fix memory leak in DataLoader"
git commit -m "docs: update installation instructions"

# Bad commits
git commit -m "fixes"
git commit -m "changes"
```

Use these prefixes:
- `feature:` - New functionality
- `fix:` - Bug fixes
- `docs:` - Documentation updates
- `refactor:` - Code refactoring
- `perf:` - Performance improvements
- `test:` - Test additions/updates

### 2. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub with:

- **Clear title**: What does this PR do?
- **Description**: 
  - Why is this change needed?
  - What problems does it solve?
  - Any breaking changes?
- **Related issues**: Link to issues with `Closes #123`
- **Type of change**: 
  - [ ] Bug fix
  - [ ] New feature
  - [ ] Breaking change
  - [ ] Documentation update

### 3. PR Checklist

Before submitting, ensure:

- [ ] Code follows project style guidelines
- [ ] Changes are documented (docstrings, README updates)
- [ ] No breaking changes (or clearly documented)
- [ ] GPU/memory usage is reasonable
- [ ] Notebooks run without errors
- [ ] Commit messages are clear and descriptive

## 📋 Pull Request Process

1. **Automated checks**: CI/CD pipeline runs tests and style checks
2. **Code review**: Maintainers review your changes
3. **Discussion**: Address feedback and make updates as needed
4. **Approval**: PR is approved by maintainers
5. **Merge**: Changes are merged to main branch

## 🎓 Areas for Contribution

We especially welcome contributions in:

### High Priority
- [ ] Model optimization and efficiency improvements
- [ ] Better documentation and tutorials
- [ ] Cross-platform GPU compatibility fixes
- [ ] Unit tests and test coverage

### Medium Priority
- [ ] New model architectures (ResNet, EfficientNet variants)
- [ ] Additional augmentation strategies
- [ ] Visualization improvements
- [ ] Performance profiling and benchmarks

### Good for Beginners
- [ ] Documentation improvements
- [ ] Code comments and docstrings
- [ ] README updates
- [ ] Issue triage

## 📚 Documentation

When adding new features:

1. **Docstrings**: Add comprehensive docstrings to functions/classes
2. **Comments**: Explain complex logic
3. **README**: Update if feature is user-facing
4. **Notebooks**: Add example cells showing how to use new feature

Docstring template:

```python
def your_function(param1: Type, param2: Type = default) -> ReturnType:
    """
    Brief description of what function does.
    
    Longer description if needed, explaining the algorithm
    or any important details.
    
    Args:
        param1: Description of param1
        param2: Description of param2, defaults to X
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When input is invalid
        
    Example:
        >>> result = your_function(param1, param2)
        >>> print(result)
    """
    pass
```

## 🔄 Keeping Your Fork Updated

```bash
# Add upstream remote
git remote add upstream https://github.com/original-repo/acl_classifier.git

# Update your fork
git fetch upstream
git rebase upstream/main
git push origin main
```

## ❓ Questions?

- **Discussion**: Start a GitHub Discussion for questions
- **Issues**: Use Issues for bug reports and features
- **Documentation**: Check README and Training docs first

## 📄 Additional Notes

- **Data sharing**: Do NOT commit large model weights or datasets
- **Experiments**: Share results in Issues/Discussions, not in commits
- **Breaking changes**: Must be discussed and approved
- **Dependencies**: Minimize new dependencies, discuss first

## 🙏 Thank You!

Your contributions make this project better for everyone. We appreciate your help!

---

**Questions about contributing?** Open a GitHub Discussion or Issue tagged with `question`.
