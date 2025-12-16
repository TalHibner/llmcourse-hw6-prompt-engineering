# ADR-005: Use pytest with Mocking for Unit Testing

**Status**: Accepted

**Date**: December 2025

**Decision Makers**: Tal Hibner

---

## Context

The research platform requires comprehensive unit tests to ensure correctness, prevent regressions, and demonstrate code quality. The codebase has several external dependencies that make testing challenging:

1. **Ollama HTTP API**: LLM inference requires running Ollama service
2. **sentence-transformers**: ML model requires downloading 80MB model
3. **File System**: Reading/writing JSON files
4. **Statistical Computations**: NumPy/SciPy operations

### Testing Goals

- **Fast Execution**: Tests should run in seconds, not minutes
- **No External Dependencies**: Tests should not require Ollama, internet, or large downloads
- **Deterministic**: Same code → same test results (no flakiness)
- **High Coverage**: ≥80% coverage on core modules
- **Isolated Units**: Test each module independently
- **Continuous Integration Ready**: Run in CI/CD pipelines without special setup

### Challenges

- **Ollama Dependency**: Cannot assume Ollama is installed/running
- **ML Model Size**: sentence-transformers model is 80MB download
- **Non-Deterministic LLM**: Same prompt may yield different responses
- **Slow Inference**: Real LLM calls take 2-5 seconds each

---

## Alternatives Considered

### 1. Integration Tests Only (No Mocking)

**Approach**: Test against real Ollama and real sentence-transformers.

```python
def test_experiment_flow():
    # Requires Ollama running
    evaluator = ExperimentEvaluator()
    results = evaluator.evaluate_technique(...)
    assert results['similarity'] > 0.5
```

**Pros**:
- ✅ Tests real system behavior
- ✅ Catches integration issues

**Cons**:
- ❌ **Slow**: 192 LLM calls × 3 seconds = 10 minutes per test run
- ❌ **Requires setup**: Ollama must be installed and running
- ❌ **Non-deterministic**: LLM responses vary, tests flaky
- ❌ **CI/CD complexity**: Need to install Ollama in pipeline
- ❌ **Fails without internet**: sentence-transformers model download

---

### 2. Fixture-Based Testing (Pre-Recorded Responses)

**Approach**: Record real responses once, replay in tests.

```python
@pytest.fixture
def recorded_responses():
    return {
        "What is 2+2?": "4",
        "Capital of France?": "Paris"
    }

def test_with_fixtures(recorded_responses):
    response = recorded_responses["What is 2+2?"]
    assert response == "4"
```

**Pros**:
- ✅ Fast (no real API calls)
- ✅ Deterministic results

**Cons**:
- ❌ **Brittle**: Adding new test requires recording new response
- ❌ **Maintenance burden**: Update fixtures when prompts change
- ❌ **Limited coverage**: Only tests recorded scenarios
- ❌ **Doesn't test code paths**: Hard to test error handling, retries

---

### 3. Test Doubles (Fakes/Stubs)

**Approach**: Create fake implementations that mimic real behavior.

```python
class FakeOllamaProvider:
    def generate(self, prompt: str) -> str:
        if "2+2" in prompt:
            return "4"
        return "I don't know"

def test_with_fake():
    provider = FakeOllamaProvider()
    response = provider.generate("What is 2+2?")
    assert response == "4"
```

**Pros**:
- ✅ Fast, no external dependencies
- ✅ Controlled behavior

**Cons**:
- ❌ **Duplication**: Maintain both real and fake implementations
- ❌ **Drift risk**: Fake behavior may diverge from real implementation
- ❌ **Incomplete**: Hard to fake complex behaviors (retries, errors)

---

### 4. pytest with unittest.mock (SELECTED)

**Approach**: Use `unittest.mock` to replace external dependencies in tests.

```python
from unittest.mock import Mock, patch

@patch('requests.post')
def test_ollama_provider(mock_post):
    # Mock HTTP response
    mock_response = Mock()
    mock_response.json.return_value = {"response": "Paris"}
    mock_post.return_value = mock_response

    # Test real code with mocked dependency
    provider = OllamaProvider()
    result = provider.generate("Capital of France?")

    # Verify behavior
    assert result == "Paris"
    mock_post.assert_called_once()
```

**Pros**:
- ✅ **Fast**: No real API calls, runs in milliseconds
- ✅ **No setup required**: No Ollama, no model downloads
- ✅ **Deterministic**: Mocks return predictable values
- ✅ **Comprehensive**: Can test error paths, edge cases
- ✅ **Built-in**: `unittest.mock` in Python standard library
- ✅ **pytest integration**: Clean syntax with fixtures
- ✅ **CI/CD friendly**: No special dependencies

**Cons**:
- ⚠️ **Not testing real integrations**: Mocks may not match reality
  - **Mitigation**: Add separate integration test suite (optional, slower)
- ⚠️ **Test implementation details**: Tightly coupled to internal calls
  - **Mitigation**: Mock at boundary (HTTP, file I/O), not internals

---

## Decision

**We will use pytest as the testing framework with unittest.mock for mocking external dependencies (HTTP requests, ML models, file system).**

### Testing Architecture

```
tests/
├── conftest.py                      ← Shared fixtures
├── providers/
│   └── test_ollama_provider.py      ← Mock requests.post
├── techniques/
│   └── test_techniques.py           ← Pure logic, no mocks needed
├── evaluation/
│   └── test_similarity.py           ← Mock SentenceTransformer
├── analysis/
│   ├── test_stats_calculator.py     ← Pure math, no mocks
│   └── test_statistical_tests.py    ← Test scipy functions
└── integration/                     ← (Optional) Real Ollama tests
    └── test_end_to_end.py
```

### Mocking Strategy

#### 1. Mock External HTTP Calls

**Target**: `requests.post` in `OllamaProvider`

```python
@patch('requests.post')
def test_ollama_generate_success(mock_post):
    mock_response = Mock()
    mock_response.json.return_value = {"response": "Test response"}
    mock_response.status_code = 200
    mock_post.return_value = mock_response

    provider = OllamaProvider()
    result = provider.generate("Test prompt")

    assert result == "Test response"
    assert mock_post.call_count == 1
```

#### 2. Mock ML Models

**Target**: `SentenceTransformer` in `SimilarityCalculator`

```python
@pytest.fixture
def mock_sentence_transformer():
    with patch('src.evaluation.similarity.SentenceTransformer') as mock:
        mock_model = Mock()
        mock_model.encode.return_value = np.array([1.0, 0.0, 0.0])
        mock.return_value = mock_model
        yield mock_model

def test_similarity_calculation(mock_sentence_transformer):
    calc = SimilarityCalculator()
    score = calc.calculate("text1", "text2")
    assert 0.0 <= score <= 1.0
```

#### 3. Mock File System

**Target**: File I/O in `DatasetIO`

```python
@patch('builtins.open', new_callable=mock_open, read_data='{"name": "test"}')
def test_load_dataset(mock_file):
    dataset = DatasetIO.load_dataset(Path("test.json"))
    assert dataset.name == "test"
    mock_file.assert_called_once()
```

#### 4. Pure Functions (No Mocking)

**Target**: Statistical calculations, prompt formatting

```python
def test_cohens_d():
    # Pure function, no mocking needed
    scores1 = [0.6, 0.7, 0.8]
    scores2 = [0.8, 0.9, 0.9]
    effect_size = StatisticalTests.cohens_d(scores1, scores2)
    assert effect_size > 0  # scores2 higher
```

---

## Implementation

### Test Organization

**conftest.py** (Shared Fixtures):
```python
import pytest
from unittest.mock import Mock

@pytest.fixture
def sample_question():
    return "What is 2+2?"

@pytest.fixture
def sample_answer():
    return "4"

@pytest.fixture
def sample_dataset():
    return [
        {"id": "1", "question": "Q1", "expected_answer": "A1"},
        {"id": "2", "question": "Q2", "expected_answer": "A2"}
    ]

@pytest.fixture
def mock_ollama_provider():
    provider = Mock()
    provider.generate.return_value = "Mocked response"
    return provider
```

### Coverage Configuration

**.coveragerc**:
```ini
[run]
source = src
omit =
    */tests/*
    */test_*.py
    */__init__.py

[report]
precision = 2
show_missing = True
skip_covered = False
```

**pytest.ini**:
```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --cov=src --cov-report=html --cov-report=term-missing
```

### Running Tests

```bash
# Run all tests with coverage
pytest

# Run specific test file
pytest tests/providers/test_ollama_provider.py

# Run with verbose output
pytest -v

# Generate HTML coverage report
pytest --cov=src --cov-report=html
open htmlcov/index.html
```

---

## Consequences

### Positive

1. **Fast Tests**: 89 tests run in ~3 seconds (vs 10+ minutes without mocking)

2. **No External Dependencies**: Tests pass without Ollama or internet
   ```bash
   # Works in fresh environment
   git clone repo
   uv pip install -e ".[dev]"
   pytest  # All pass
   ```

3. **Deterministic**: Same code → same test results (no flaky tests)

4. **CI/CD Ready**: GitHub Actions, GitLab CI work without special setup
   ```yaml
   # .github/workflows/test.yml
   - run: uv pip install -e ".[dev]"
   - run: pytest
   ```

5. **High Coverage Achieved**: 100% on core modules (providers, techniques, evaluation, statistics)

6. **Error Path Testing**: Easy to test retries, timeouts, exceptions
   ```python
   @patch('requests.post', side_effect=ConnectionError("Network error"))
   def test_ollama_connection_error(mock_post):
       provider = OllamaProvider()
       with pytest.raises(ConnectionError):
           provider.generate("test")
   ```

7. **Isolated Units**: Each module tested independently, easier debugging

### Negative

1. **Not Testing Real Integration**: Mocks may not match actual behavior
   - **Mitigation**: Created separate integration test file (optional, requires Ollama)
   - **Acceptance**: Unit tests verify logic, integration tests verify system

2. **Mocking Complexity**: Learning curve for `patch`, `Mock`, `MagicMock`
   - **Mitigation**: Documented examples in conftest.py
   - **Resources**: unittest.mock documentation, pytest-mock plugin

3. **Brittle to Refactoring**: If implementation changes (e.g., switch from requests to httpx), mocks break
   - **Mitigation**: Mock at stable boundaries (HTTP client interface, not internals)

4. **False Confidence**: Tests pass but real code might fail
   - **Mitigation**: Run integration tests before major releases
   - **Strategy**: 90% mocked unit tests + 10% real integration tests

### Test Coverage Results

**Achieved**:
```
src/providers/          100%
src/techniques/         100%
src/evaluation/         100%
src/analysis/stats      100%
src/config/             100%

Overall: 34% (includes visualization, which is pending)
Core modules: 100%
```

---

## Validation

### Test Quality Metrics

✅ **All tests pass**: 89/89 passing
✅ **Fast execution**: 3 seconds total
✅ **No external deps**: Works offline
✅ **Comprehensive**: 100% coverage on core logic
✅ **Isolated**: Each test independent
✅ **Documented**: Clear test names, docstrings

### Example Test Quality

**Good Test** (specific, isolated, fast):
```python
def test_baseline_technique_formats_question_directly():
    """Baseline should return question without modification"""
    technique = BaselinePrompt()
    result = technique.apply("What is 2+2?")
    assert result == "What is 2+2?"
    assert "Let's think step by step" not in result
```

**Bad Test** (vague, slow, dependent):
```python
def test_everything():
    """Test that system works"""
    # Requires Ollama
    evaluator = ExperimentEvaluator()
    # Runs 192 LLM calls (slow)
    results = evaluator.run_all()
    # Vague assertion
    assert len(results) > 0
```

---

## References

- **pytest Documentation**: https://docs.pytest.org/
- **unittest.mock**: https://docs.python.org/3/library/unittest.mock.html
- **pytest-mock Plugin**: https://pytest-mock.readthedocs.io/
- **Test Pyramid**: Martin Fowler - https://martinfowler.com/articles/practical-test-pyramid.html
- **Python Testing with pytest** by Brian Okken

---

## Future Considerations

### Enhancements

1. **Property-Based Testing**: Use `hypothesis` for generative tests
   ```python
   from hypothesis import given, strategies as st

   @given(st.text(), st.text())
   def test_similarity_range(text1, text2):
       score = calc.calculate(text1, text2)
       assert 0 <= score <= 1
   ```

2. **Mutation Testing**: Verify tests catch bugs
   ```bash
   pip install mutpy
   mutpy --target src --unit-test tests
   ```

3. **Integration Test Suite**: Optional Ollama tests for pre-release validation
   ```python
   @pytest.mark.integration
   @pytest.mark.requires_ollama
   def test_real_ollama():
       provider = OllamaProvider()
       response = provider.generate("What is 2+2?")
       assert "4" in response
   ```

4. **Contract Testing**: Verify mocks match real API behavior
   ```python
   @pytest.mark.contract
   def test_ollama_api_contract():
       # Verify mock matches real Ollama response structure
       response = requests.post("http://localhost:11434/api/generate", ...)
       assert "response" in response.json()  # Validate structure
   ```

### When to Reconsider

- **High integration risk**: If mocks diverge significantly from reality
- **Complex external APIs**: If mocking becomes more complex than real calls
- **Shared test infrastructure**: If team has dedicated Ollama test server

---

**ADR Status**: ✅ Implemented and Validated

**Last Review**: December 16, 2025

**Related ADRs**: ADR-001 (Ollama), ADR-002 (Sentence-BERT)
