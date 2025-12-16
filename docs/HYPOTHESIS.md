# Formal Research Hypotheses

**Project:** Prompt Engineering for LLM Response Optimization
**Date:** December 16, 2025
**Research Domain:** Natural Language Processing, Large Language Models

---

## Research Question

**How do different prompt engineering techniques affect Large Language Model (LLM) response accuracy and consistency when measured through vector similarity to ground-truth answers?**

---

## Hypotheses

### Primary Hypothesis (H1)

**H₁:** Structured prompt engineering techniques (Standard Improved, Few-Shot, Chain-of-Thought) will produce statistically significant improvements in response accuracy compared to baseline prompting, as measured by cosine similarity scores between LLM responses and expected answers.

**Null Hypothesis (H₀):** There is no statistically significant difference in response accuracy between structured prompt engineering techniques and baseline prompting (p ≥ 0.05).

**Operationalization:**
- **Independent Variable:** Prompt engineering technique (Baseline, Standard, Few-Shot, CoT)
- **Dependent Variable:** Mean cosine similarity score (range: 0-1)
- **Statistical Test:** Independent samples t-test
- **Significance Level:** α = 0.05
- **Effect Size:** Cohen's d

---

### Secondary Hypotheses

#### H2: Few-Shot Learning Effect

**H₂ₐ:** Few-shot prompting will show greater improvement over baseline in sentiment analysis tasks (classification) compared to chain-of-thought reasoning tasks.

**H₀₂ₐ:** The improvement ratio of few-shot prompting over baseline will be equivalent across task types.

**Rationale:** Few-shot learning provides concrete examples that may be more beneficial for pattern-matching tasks like classification than for complex reasoning.

---

#### H3: Chain-of-Thought Effect

**H₃ₐ:** Chain-of-thought prompting will show greater improvement in multi-step reasoning tasks compared to simple classification tasks.

**H₀₃ₐ:** Chain-of-thought improvement will be equivalent across task types.

**Rationale:** CoT explicitly requests step-by-step reasoning, which should particularly benefit tasks requiring logical deduction.

---

#### H4: Response Consistency

**H₄ₐ:** Structured prompt techniques will demonstrate lower variance (higher consistency) in response quality compared to baseline prompting.

**H₀₄ₐ:** There is no significant difference in response variance between techniques.

**Operationalization:**
- **Metric:** Standard deviation of similarity scores
- **Test:** Levene's test for equality of variances
- **Lower std = better consistency**

---

## Expected Outcomes

### Quantitative Predictions

1. **Improvement Magnitude:**
   - Few-Shot: +15-25% improvement over baseline
   - Chain-of-Thought: +10-20% improvement over baseline
   - Standard Improved: +5-15% improvement over baseline

2. **Statistical Significance:**
   - Expected p-values < 0.05 for all structured techniques vs baseline
   - Expected Cohen's d ≥ 0.5 (medium effect size)

3. **Consistency:**
   - σ(structured techniques) < σ(baseline)
   - Expected 10-20% reduction in standard deviation

---

## Confounding Variables & Controls

### Controlled Variables

1. **Model:**
   - Single model (Ollama llama3.2)
   - Consistent version throughout experiments
   - Local execution (eliminates API variability)

2. **Temperature:**
   - Fixed at default for all experiments
   - Ensures reproducibility

3. **Dataset:**
   - Fixed questions and expected answers
   - Balanced categories
   - Consistent complexity within task type

### Potential Confounds

1. **Question Length:** Controlled by categorization
2. **Answer Length:** Evaluated via similarity (accommodates variation)
3. **Model Training Data:** Fixed (same model throughout)
4. **Temporal Effects:** All experiments within same session

---

## Measurement Validity

### Construct Validity

**Accuracy Measurement:**
- **Metric:** Cosine similarity of sentence embeddings
- **Model:** sentence-transformers (all-MiniLM-L6-v2)
- **Range:** [0, 1] where 1 = perfect similarity
- **Validity:** Validated in semantic similarity literature (Reimers & Gurevych, 2019)

**Why Cosine Similarity:**
1. Captures semantic meaning (not just lexical overlap)
2. Robust to paraphrasing
3. Normalized metric (comparable across questions)
4. Widely used in NLP research

### Reliability

**Test-Retest Reliability:**
- Deterministic embedding model
- Fixed dataset
- Reproducible results

**Internal Consistency:**
- Multiple examples per category
- Balanced categories
- Consistent format

---

## Statistical Power

### Sample Size Justification

- **Sentiment Analysis:** 40 examples (15 pos, 15 neg, 10 neutral)
- **Chain-of-Thought:** 8 examples (5 math, 3 logic)
- **Total per technique:** 48 evaluations

**Power Analysis:**
- Expected effect size: d = 0.5 (medium)
- Alpha: 0.05
- Power (1-β): ~0.80 for n=40
- Adequate for detecting medium-large effects

---

## Limitations

1. **Single Model:** Results specific to llama3.2 (not generalizable to all LLMs)
2. **Task Domain:** Limited to sentiment & reasoning (not comprehensive)
3. **Language:** English only
4. **Similarity Metric:** Cosine similarity may not capture all aspects of quality
5. **Local Execution:** Different hardware may produce slight variations

---

## Research Contribution

This study contributes to the growing body of empirical research on prompt engineering by:

1. **Quantifying** the impact of specific techniques using objective metrics
2. **Comparing** multiple techniques under controlled conditions
3. **Providing** statistical evidence (not just anecdotal examples)
4. **Using** open-source tools (Ollama) for reproducibility
5. **Establishing** a methodology for systematic prompt evaluation

---

## References

- Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *EMNLP*.
- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences*. Routledge.
- Wei, J., et al. (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. *NeurIPS*.

---

*Document Version: 1.0*
*Last Updated: December 16, 2025*
