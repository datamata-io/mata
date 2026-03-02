# Contributing to MATA

Thank you for your interest in contributing! This guide covers everything you need to get started.

## Development Setup

**Requirements**: Python 3.10+

```bash
git clone https://github.com/datamata-io/mata.git
cd mata
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

Verify the installation:

```bash
python verify_install.py
```

## Running Tests

```bash
# Full test suite (4,307+ tests)
pytest tests/ -v

# Single test file
pytest tests/test_api.py -v

# Single test
pytest tests/test_api.py::test_load_detect -v

# With coverage (target: ≥70%)
pytest tests/ --cov=mata --cov-report=html
```

## Code Style

| Tool  | Purpose       | Config                     |
| ----- | ------------- | -------------------------- |
| Black | Formatting    | `line-length = 120`        |
| Ruff  | Linting       | `pyproject.toml`           |
| Mypy  | Type checking | `--ignore-missing-imports` |

Run all checks at once:

```bash
black src/ tests/ && ruff check src/ && mypy src/mata/ --ignore-missing-imports
```

Check-only (no writes, for CI):

```bash
black --check src/ tests/
ruff check src/
mypy src/mata/ --ignore-missing-imports
```

## Pull Request Process

1. Fork the repository and create a feature branch: `git checkout -b feature/my-feature`
2. Write tests for your changes in `tests/test_<feature>.py`
3. Ensure all tests pass and linters are clean
4. Submit a PR with a clear description of what changed and why
5. Link any related issues in the PR description

## Adding a New Adapter

All adapters inherit from base classes in `src/mata/adapters/base.py`:

```python
from mata.adapters.base import PyTorchBaseAdapter
from mata.core.types import VisionResult

class MyDetectAdapter(PyTorchBaseAdapter):
    def __init__(self, model_path: str, **kwargs):
        super().__init__(
            device=kwargs.get("device", "auto"),
            threshold=kwargs.get("threshold", 0.3),
        )
        # Load your model here

    def predict(self, image, **kwargs) -> VisionResult:
        pil_image = self._load_image(image)   # base class helper
        # Run inference, build instances
        return VisionResult(instances=[...])
```

**Checklist for new adapters:**

1. Inherit from `PyTorchBaseAdapter` or `ONNXBaseAdapter` (both in `base.py`)
2. Implement `predict()` returning the task-specific result type  
   (`VisionResult`, `ClassifyResult`, or `DepthResult`)
3. Add tests in `tests/test_<adapter_name>.py` covering all supported runtimes
4. Register the adapter in the appropriate task loader
5. Update `docs/` if the adapter introduces new user-facing behaviour

## Result Types

| Task           | Return type      | Notes                         |
| -------------- | ---------------- | ----------------------------- |
| detect/segment | `VisionResult`   | `instances: list[Instance]`   |
| classify       | `ClassifyResult` | `.top1`, `.top5` accessors    |
| depth          | `DepthResult`    | `depth_map: np.ndarray` (H×W) |

All result types support `.save(path)` and `.to_json()`.

## Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add TorchScript depth adapter
fix: correct xyxy bbox conversion in ONNX adapter
docs: update CLIP quick-start guide
test: add edge-case tests for zero-shot detection
refactor: extract shared pre-processing into BaseAdapter
```

## Code of Conduct

All contributors are expected to follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## Questions?

Open a [GitHub Discussion](https://github.com/datamata-io/mata/discussions) for general questions, or a [GitHub Issue](https://github.com/datamata-io/mata/issues) for bugs and feature requests.
