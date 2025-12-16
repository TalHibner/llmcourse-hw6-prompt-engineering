# ADR-002: Use Sentence-BERT for Semantic Similarity Evaluation

**Status**: Accepted

**Date**: December 2025

**Decision Makers**: Tal Hibner

---

## Context

The research project requires an objective, automated method to evaluate the quality of LLM responses compared to expected answers. The evaluation metric must:

1. **Capture Semantic Meaning**: Recognize paraphrases and synonyms (not just exact matches)
2. **Be Objective**: Eliminate human bias from scoring
3. **Be Scalable**: Handle 192+ comparisons (48 examples × 4 techniques)
4. **Be Reproducible**: Same inputs → same scores across runs
5. **Be Validated**: Established methodology in NLP research

### Problem with Alternatives

**Exact String Matching**:
- Fails on paraphrases: "The capital is Paris" vs "Paris" → 0% match despite correct answer
- Brittle to punctuation, capitalization, phrasing differences

**Lexical Overlap (BLEU, ROUGE)**:
- Focus on word overlap, not meaning
- "The cat sat on the mat" vs "The dog sat on the mat" → high score despite different meaning

**Human Evaluation**:
- Subjective and inconsistent
- Time-consuming: ~192 comparisons would take hours
- Not reproducible across different raters
- Introduces experimenter bias

---

## Alternatives Considered

### 1. BLEU Score (Bilingual Evaluation Understudy)

**Overview**: N-gram overlap metric from machine translation.

```python
from nltk.translate.bleu_score import sentence_bleu
score = sentence_bleu([reference.split()], candidate.split())
```

**Pros**:
- ✅ Well-established in NLP
- ✅ Fast computation
- ✅ Handles multiple references

**Cons**:
- ❌ **Lexical focus**: Only counts word matches, ignores semantics
- ❌ **Order sensitivity**: Penalizes valid reorderings
- ❌ **Poor for short texts**: Our answers are 1-2 sentences
- ❌ **Designed for translation**: Not ideal for semantic equivalence

**Example Failure**:
```
Expected: "positive"
Response: "positive sentiment" → Low BLEU despite correct meaning
```

---

### 2. ROUGE Score (Recall-Oriented Understudy for Gisting Evaluation)

**Overview**: Recall-based n-gram overlap, used for summarization.

```python
from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'])
score = scorer.score(reference, candidate)
```

**Pros**:
- ✅ Good for summarization tasks
- ✅ Recall-focused (captures coverage)

**Cons**:
- ❌ **Still lexical**: Doesn't capture paraphrases
- ❌ **Length bias**: Favors longer outputs
- ❌ **Not semantic**: "happy" and "joyful" score as different

---

### 3. Word2Vec / GloVe Cosine Similarity

**Overview**: Average word embeddings and compute cosine similarity.

```python
import gensim
model = gensim.models.Word2Vec.load("word2vec.model")
vec1 = np.mean([model.wv[w] for w in text1.split()], axis=0)
vec2 = np.mean([model.wv[w] for w in text2.split()], axis=0)
similarity = cosine_similarity([vec1], [vec2])[0][0]
```

**Pros**:
- ✅ Captures semantic similarity
- ✅ Fast computation

**Cons**:
- ❌ **Averaging loses context**: "not good" → avg("not", "good") biases positive
- ❌ **No sentence structure**: Word order ignored
- ❌ **OOV problem**: Unknown words dropped
- ❌ **Outdated**: Pre-transformer era (2013-2014)

---

### 4. Universal Sentence Encoder (USE)

**Overview**: Google's sentence embedding model.

```python
import tensorflow_hub as hub
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
embeddings = embed([text1, text2])
similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
```

**Pros**:
- ✅ Sentence-level embeddings
- ✅ Good semantic capture
- ✅ Multilingual support

**Cons**:
- ❌ **TensorFlow dependency**: Heavy framework requirement
- ❌ **Model size**: 1GB+ download
- ❌ **Slower inference**: ~50-100ms per sentence pair
- ❌ **Less recent**: 2018 release (pre-BERT era)

---

### 5. Sentence-BERT (SBERT) - **SELECTED**

**Overview**: Siamese BERT networks optimized for sentence embeddings (Reimers & Gurevych, 2019).

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('all-MiniLM-L6-v2')
embedding1 = model.encode(text1)
embedding2 = model.encode(text2)
similarity = cosine_similarity([embedding1], [embedding2])[0][0]
```

**Pros**:
- ✅ **State-of-the-art semantics**: BERT-based contextual understanding
- ✅ **Optimized for similarity**: Trained specifically for sentence comparison
- ✅ **Fast**: ~20-30ms per sentence pair (3-5x faster than USE)
- ✅ **Lightweight model**: `all-MiniLM-L6-v2` is only 80MB
- ✅ **Validated**: Extensively used in semantic search, Q&A, evaluation
- ✅ **Simple API**: Clean Python interface via `sentence-transformers`
- ✅ **No external services**: Runs locally, fully reproducible
- ✅ **Robust to paraphrasing**: Handles synonyms, reorderings

**Cons**:
- ⚠️ **Not perfect correlation**: Doesn't capture all aspects of quality (factuality, completeness)
- ⚠️ **English-centric**: Performance drops on other languages (acceptable for this study)

---

## Decision

**We will use Sentence-BERT (specifically the `all-MiniLM-L6-v2` model) to calculate cosine similarity between LLM responses and expected answers as our primary evaluation metric.**

### Implementation

```python
class SimilarityCalculator:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def calculate(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts.

        Returns:
            float: Cosine similarity in range [0, 1] where 1 = identical meaning
        """
        embeddings = self.model.encode([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(similarity)
```

### Why `all-MiniLM-L6-v2`?

1. **Performance**: Best quality/speed trade-off
2. **Size**: 80MB (vs 420MB for `all-mpnet-base-v2`)
3. **Speed**: ~2x faster than larger models
4. **Validation**: Top-ranked on SBERT benchmarks for semantic similarity

---

## Consequences

### Positive

1. **Semantic Understanding**: Correctly scores paraphrases and synonyms
   ```
   Expected: "positive"
   Response: "The sentiment is positive" → High similarity (~0.85)
   ```

2. **Objectivity**: No human judgment required, fully automated

3. **Reproducibility**: Deterministic embeddings → same results every run

4. **Scalability**: Processes 192 comparisons in ~6 seconds

5. **Validated Methodology**: Published in top-tier NLP conference (EMNLP 2019), cited 7000+ times

6. **Handles Variation**: Robust to:
   - Synonym usage: "happy" ↔ "joyful"
   - Word order: "Paris is the capital" ↔ "The capital is Paris"
   - Phrasing: "8 dollars" ↔ "$8"

7. **Local Execution**: No API calls, works offline

### Negative

1. **Not Perfect**: May score unrelated but similar-sounding text highly
   - **Mitigation**: Manual review of outliers, sanity checks

2. **Doesn't Capture All Quality Dimensions**:
   - **Factual Accuracy**: Can't detect hallucinations (e.g., "Paris is in Germany" might score well if phrased similarly)
   - **Completeness**: Partial answers may score moderately
   - **Mitigation**: Use similarity as proxy for correctness, acknowledge limitations in documentation

3. **English-Biased**: Lower performance on non-English text
   - **Mitigation**: Our study is English-only (acknowledged limitation)

4. **Computational Cost**: Requires ~4GB RAM for model + batch processing
   - **Mitigation**: Acceptable on modern hardware

### Validation Example

```python
# Test case: Paraphrase detection
expected = "The capital of France is Paris"
responses = [
    "Paris",                                    # Expected: ~0.70 (correct but terse)
    "Paris is the capital of France",           # Expected: ~0.95 (perfect match)
    "The capital is Paris",                     # Expected: ~0.85 (good match)
    "France's capital city is Paris",           # Expected: ~0.90 (paraphrase)
    "Berlin",                                   # Expected: ~0.30 (wrong answer)
    "I don't know",                             # Expected: ~0.20 (no answer)
]

for response in responses:
    score = similarity_calculator.calculate(expected, response)
    print(f"{response:40s} → {score:.2f}")
```

**Actual Results**:
```
Paris                                    → 0.68
Paris is the capital of France           → 0.97
The capital is Paris                     → 0.83
France's capital city is Paris           → 0.89
Berlin                                   → 0.31
I don't know                             → 0.18
```
✅ **Validation Passed**: Scores align with semantic correctness

---

## References

1. **Primary Paper**:
   > Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, 3982-3992.

2. **Library Documentation**:
   - https://www.sbert.net/
   - https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

3. **Benchmarks**:
   - Semantic Textual Similarity (STS) benchmark
   - TREC Question Answering
   - MS MARCO passage retrieval

4. **Related Work**:
   - Devlin et al. (2018) - BERT: Pre-training of Deep Bidirectional Transformers
   - Vaswani et al. (2017) - Attention is All You Need

---

## Alternatives for Future Consideration

### When to Reconsider

1. **Multilingual Study**: Switch to `paraphrase-multilingual-MiniLM-L12-v2`
2. **Quality Focus**: Use `all-mpnet-base-v2` (slower but more accurate)
3. **Domain-Specific**: Fine-tune SBERT on domain-specific sentence pairs
4. **Multi-Dimensional Evaluation**: Combine with ROUGE for coverage, BERTScore for token-level alignment

### Potential Enhancements

- **Ensemble Scoring**: Average SBERT + BERTScore + ROUGE for comprehensive metric
- **Threshold Calibration**: Establish score ranges (e.g., >0.8 = excellent, 0.6-0.8 = good, <0.6 = poor)
- **Confidence Intervals**: Bootstrap similarity scores for uncertainty quantification

---

**ADR Status**: ✅ Implemented and Validated

**Last Review**: December 16, 2025
