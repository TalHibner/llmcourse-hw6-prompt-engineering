# ADR-003: Use JSON for Dataset and Results Storage

**Status**: Accepted

**Date**: December 2025

**Decision Makers**: Tal Hibner

---

## Context

The research platform needs to persist two types of data:

1. **Datasets**: Question-answer pairs for experiments
   - Sentiment analysis examples (40 examples)
   - Chain-of-thought reasoning problems (8 examples)
   - Metadata: categories, difficulty, expected steps

2. **Experimental Results**: LLM responses and evaluation metrics
   - Per-technique results (baseline, standard, few-shot, CoT)
   - Similarity scores, timestamps, model info
   - Statistical aggregations

### Requirements

- **Human-Readable**: Researchers should be able to inspect and edit files
- **Version Control Friendly**: Git-compatible format (text-based, diff-friendly)
- **Language-Agnostic**: Readable by Python, JavaScript, R, etc.
- **Schema-Flexible**: Easy to add new fields without breaking compatibility
- **Simple Serialization**: Built-in Python support (no external dependencies)
- **Widely Supported**: Standard format with good tooling

---

## Alternatives Considered

### 1. CSV (Comma-Separated Values)

```csv
id,question,expected_answer,category
sent_001,"What is your opinion of this product?","positive","positive_sentiment"
```

**Pros**:
- ✅ Simplest format
- ✅ Excel-compatible
- ✅ Small file size
- ✅ Easy parsing

**Cons**:
- ❌ **No nested structures**: Cannot represent metadata dictionaries
- ❌ **Escape complexity**: Commas, quotes in text require escaping
- ❌ **Type ambiguity**: Everything is a string (no int/float/bool distinction)
- ❌ **Poor for results**: Multiple similarity scores per example don't fit flat structure
- ❌ **Not hierarchical**: Can't represent technique → examples → scores

---

### 2. SQLite Database

```python
import sqlite3
conn = sqlite3.connect('data.db')
cursor.execute('''CREATE TABLE examples (
    id TEXT PRIMARY KEY,
    question TEXT,
    expected_answer TEXT,
    category TEXT
)''')
```

**Pros**:
- ✅ Efficient queries (SELECT, JOIN, aggregations)
- ✅ ACID transactions
- ✅ Built-in Python support (`sqlite3`)
- ✅ Enforced schema

**Cons**:
- ❌ **Not human-readable**: Binary format, requires SQL client to inspect
- ❌ **Poor version control**: Binary files don't diff well in Git
- ❌ **Overkill for scale**: Only ~50 examples, no complex queries needed
- ❌ **Migration complexity**: Schema changes require ALTER TABLE migrations
- ❌ **Not language-agnostic**: Requires SQL knowledge to query

---

### 3. YAML (YAML Ain't Markup Language)

```yaml
dataset_name: sentiment_analysis
examples:
  - id: sent_001
    question: What is your opinion of this product?
    expected_answer: positive
    category: positive_sentiment
    metadata:
      difficulty: easy
```

**Pros**:
- ✅ Very human-readable (minimal syntax)
- ✅ Supports nested structures
- ✅ Comments allowed
- ✅ Popular in config files

**Cons**:
- ❌ **Whitespace sensitivity**: Indentation errors break parsing
- ❌ **Complex spec**: YAML 1.2 has surprising behaviors (Norway problem: `NO` → False)
- ❌ **Slower parsing**: More complex than JSON
- ❌ **Security concerns**: `yaml.load()` can execute arbitrary code (requires `safe_load`)
- ❌ **Less universal**: Not all languages have great YAML support

---

### 4. Pickle (Python Serialization)

```python
import pickle
with open('data.pkl', 'wb') as f:
    pickle.dump(dataset, f)
```

**Pros**:
- ✅ Native Python objects
- ✅ Preserves types exactly
- ✅ Fast serialization

**Cons**:
- ❌ **Binary format**: Not human-readable at all
- ❌ **Python-only**: Other languages cannot read
- ❌ **Security risk**: Unpickling untrusted data executes code
- ❌ **Version fragility**: Python version changes can break compatibility
- ❌ **No version control**: Binary diffs are useless

---

### 5. JSON (JavaScript Object Notation) - **SELECTED**

```json
{
  "dataset_name": "sentiment_analysis",
  "description": "Sentiment classification examples",
  "num_examples": 40,
  "examples": [
    {
      "id": "sent_pos_001",
      "question": "What is your opinion of this product?",
      "expected_answer": "positive",
      "category": "positive_sentiment",
      "metadata": {
        "difficulty": "easy",
        "domain": "product_review"
      }
    }
  ]
}
```

**Pros**:
- ✅ **Human-readable**: Clear structure, easy to inspect
- ✅ **Widely supported**: Native support in Python, JavaScript, R, Java, etc.
- ✅ **Simple syntax**: Minimal ambiguity, predictable parsing
- ✅ **Version control friendly**: Text format diffs nicely in Git
- ✅ **Schema-flexible**: Can add new fields without breaking old code
- ✅ **Built-in Python**: `json` module in standard library (no dependencies)
- ✅ **Nested structures**: Supports arbitrary depth (metadata, lists of objects)
- ✅ **Type preservation**: Distinguishes strings, numbers, booleans, null

**Cons**:
- ⚠️ **No comments**: Cannot embed documentation in data files
  - **Mitigation**: Use separate README or description fields
- ⚠️ **Trailing commas forbidden**: `[1, 2,]` is invalid
  - **Mitigation**: Use `json.dump()` for serialization (automatic)
- ⚠️ **Verbose**: More bytes than CSV for flat data
  - **Mitigation**: File size is negligible (~50KB for all datasets)

---

## Decision

**We will use JSON as the standard format for all datasets and experimental results.**

### Implementation

**Dataset Schema**:
```json
{
  "dataset_name": "string",
  "description": "string",
  "num_examples": "integer",
  "examples": [
    {
      "id": "string (unique)",
      "question": "string",
      "expected_answer": "string",
      "category": "string",
      "metadata": {
        "difficulty": "easy|medium|hard",
        "steps": "integer (for CoT)",
        "reasoning_steps": ["string"] (for CoT)
      }
    }
  ]
}
```

**Results Schema**:
```json
{
  "technique": "string",
  "dataset": "string",
  "model": "string",
  "timestamp": "ISO 8601 string",
  "config": {
    "temperature": "float",
    "model_version": "string"
  },
  "results": [
    {
      "example_id": "string",
      "question": "string",
      "expected_answer": "string",
      "llm_response": "string",
      "similarity_score": "float [0, 1]",
      "inference_time_ms": "integer"
    }
  ],
  "summary": {
    "mean_similarity": "float",
    "std_similarity": "float",
    "num_examples": "integer"
  }
}
```

**Python Serialization**:
```python
import json
from pathlib import Path

def save_dataset(dataset: Dataset, output_path: Path) -> None:
    data = {
        "dataset_name": dataset.name,
        "description": dataset.description,
        "num_examples": len(dataset.examples),
        "examples": [
            {
                "id": ex.id,
                "question": ex.question,
                "expected_answer": ex.expected_answer,
                "category": ex.category,
                "metadata": ex.metadata
            }
            for ex in dataset.examples
        ]
    }
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

def load_dataset(input_path: Path) -> Dataset:
    with open(input_path, 'r') as f:
        data = json.load(f)
    # Convert to Dataset objects...
```

---

## Consequences

### Positive

1. **Git-Friendly**:
   ```diff
   {
     "examples": [
       {
         "id": "sent_001",
   -     "expected_answer": "positive"
   +     "expected_answer": "positive sentiment"
       }
     ]
   }
   ```
   → Clear diff shows exactly what changed

2. **Cross-Platform**: Can be read by R, JavaScript, curl, jq, etc.
   ```bash
   # Query with jq
   cat data/datasets/sentiment_analysis.json | jq '.examples[] | select(.category=="positive")'
   ```

3. **Schema Evolution**: Easy to add new fields
   ```python
   # Old code ignores unknown fields (forward compatibility)
   example = {
       "id": "sent_001",
       "question": "...",
       "new_field": "new_value"  # Old parsers ignore this
   }
   ```

4. **Debugging-Friendly**: Inspect files directly in any text editor

5. **No Dependencies**: Built-in `json` module, no pip install required

6. **Validation**: JSON schema validators available (e.g., `jsonschema` library)

### Negative

1. **No Comments**: Cannot document edge cases inline
   - **Mitigation**: Use `"_comment"` fields or separate documentation

2. **Manual Editing Errors**: Invalid JSON breaks parsing
   - **Mitigation**: Use proper JSON editors, validate before commit
   ```bash
   # Validate JSON
   python -m json.tool data.json > /dev/null
   ```

3. **Large Files**: Results can grow to MB for many experiments
   - **Mitigation**: One file per technique (baseline.json, few_shot.json), not cumulative

4. **No Data Types Beyond Basics**: No datetime, no Decimal
   - **Mitigation**: Store as ISO strings, parse in code
   ```python
   from datetime import datetime
   timestamp = datetime.fromisoformat(data["timestamp"])
   ```

---

## Validation

### Testing
```python
def test_json_round_trip():
    """Ensure datasets can be saved and loaded without loss"""
    original = Dataset(name="test", description="test", examples=[...])
    save_dataset(original, Path("test.json"))
    loaded = load_dataset(Path("test.json"))
    assert original.name == loaded.name
    assert len(original.examples) == len(loaded.examples)
    assert original.examples[0].question == loaded.examples[0].question
```

### Schema Validation (Optional Enhancement)
```python
import jsonschema

DATASET_SCHEMA = {
    "type": "object",
    "required": ["dataset_name", "examples"],
    "properties": {
        "dataset_name": {"type": "string"},
        "examples": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "question", "expected_answer"],
                "properties": {
                    "id": {"type": "string"},
                    "question": {"type": "string"},
                    "expected_answer": {"type": "string"}
                }
            }
        }
    }
}

jsonschema.validate(data, DATASET_SCHEMA)  # Raises error if invalid
```

---

## References

- **JSON Specification**: RFC 8259 (https://tools.ietf.org/html/rfc8259)
- **Python json module**: https://docs.python.org/3/library/json.html
- **JSON Schema**: https://json-schema.org/
- **jq (JSON processor)**: https://stedolan.github.io/jq/

---

## Future Considerations

### When to Reconsider

1. **Scale Increase**: If datasets grow to >10,000 examples, consider SQLite or Parquet
2. **Query Complexity**: If complex filtering/aggregation needed, use database
3. **Binary Performance**: If file I/O becomes bottleneck, consider MessagePack or Parquet

### Potential Enhancements

- **Compression**: gzip JSON files (e.g., `results.json.gz`) for large experiments
- **Streaming**: Use `ijson` for incremental parsing of very large files
- **Schema Enforcement**: Add JSON Schema validation to CI/CD pipeline
- **Versioning**: Include schema version in files (`"schema_version": "1.0"`)

---

**ADR Status**: ✅ Implemented and Validated

**Last Review**: December 16, 2025
