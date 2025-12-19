# Contributing to HierarchicalVLM

Thank you for your interest in contributing to HierarchicalVLM! We welcome contributions from the community to help improve this project.

## How to Contribute

### Reporting Bugs
If you find a bug, please create an issue with:
- Clear title and description
- Steps to reproduce the issue
- Expected vs. actual behavior
- Your environment (Python version, PyTorch version, GPU type, CUDA version)

### Suggesting Enhancements
Enhancement suggestions are tracked as GitHub issues. Please provide:
- Clear description of the proposed enhancement
- Use cases and expected benefits
- Examples of similar implementations if applicable

### Code Contributions

#### Development Setup
```bash
# Clone your fork
git clone https://github.com/your-username/hierarchical-vlm.git
cd HierarchicalVLM

# Create development environment
python3 -m venv venv
source venv/bin/activate

# Install with development dependencies
pip install -r requirements.txt
pip install pytest black isort flake8
```

#### Coding Standards
- **Style**: Follow PEP 8 conventions
- **Formatting**: Use `black` for code formatting
  ```bash
  black hierarchicalvlm/
  ```
- **Import Organization**: Use `isort` for import sorting
  ```bash
  isort hierarchicalvlm/
  ```
- **Linting**: Check with `flake8`
  ```bash
  flake8 hierarchicalvlm/ --max-line-length=100
  ```
- **Type Hints**: Use Python type hints where possible
- **Docstrings**: Include docstrings for all classes and functions
  ```python
  def function_name(param1: str, param2: int) -> dict:
      """
      Brief description of function.
      
      Args:
          param1: Description of param1
          param2: Description of param2
          
      Returns:
          Description of return value
      """
  ```

#### Testing
- Add tests for new functionality in `tests/`
- Run tests before submitting PR:
  ```bash
  pytest tests/
  ```

#### Git Workflow
1. Create a feature branch: `git checkout -b feature/your-feature-name`
2. Make your changes with clear, logical commits
3. Ensure code passes all checks
4. Push to your fork: `git push origin feature/your-feature-name`
5. Open a Pull Request with:
   - Clear description of changes
   - Reference to related issues
   - Screenshots/results if applicable

#### Pull Request Guidelines
- Keep PRs focused and reasonably sized
- Update documentation if adding/changing features
- Include tests for new code
- Add entry to CHANGELOG if significant
- Ensure CI/CD checks pass

### Documentation Contributions
- Grammar and clarity improvements welcome
- Fix broken links and outdated information
- Add examples and tutorials
- Improve docstrings and comments

## Development Practices

### Before Submitting
1. **Run formatting and linting:**
   ```bash
   black hierarchicalvlm/
   isort hierarchicalvlm/
   flake8 hierarchicalvlm/
   ```

2. **Run tests:**
   ```bash
   pytest tests/ -v
   ```

3. **Test documentation:**
   ```bash
   cd docs
   make html
   ```

### Commit Message Guidelines
- Use clear, descriptive commit messages
- Start with present tense verb: "Add", "Fix", "Update", "Refactor"
- Reference issue numbers: "Fixes #123" or "Related to #456"

Example:
```
Add hierarchical aggregation module

- Implements attention-weighted pooling
- Adds 130x memory reduction for long sequences
- Includes comprehensive unit tests
- Fixes #123
```

## Areas for Contribution

We particularly welcome contributions in these areas:

1. **Dataset Support**: Add support for additional datasets (MSR-VTT, YouCook2, etc.)
2. **Model Improvements**: Optimize hierarchical aggregation, add new attention mechanisms
3. **Documentation**: Write tutorials, add examples, improve API documentation
4. **Testing**: Increase test coverage, add integration tests
5. **Performance**: Optimize memory usage, improve training speed
6. **Deployment**: Docker support, model serving examples, edge device optimization

## Community Standards

- Be respectful and inclusive
- Provide constructive feedback
- Help others in the community
- Give credit to original authors
- Follow the MIT License terms

## Getting Help

- Check [FAQ](docs/FAQ.md) for common questions
- Search existing issues and discussions
- Read the [documentation](docs/)
- Ask questions in GitHub Discussions

## License

By contributing to HierarchicalVLM, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to HierarchicalVLM! 🎉
