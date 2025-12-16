# Literature Review: Prompt Engineering for Large Language Models

**Author:** Tal Hibner
**Date:** December 16, 2025
**Course:** LLM Applications - Homework 6

---

## Abstract

This literature review examines the current state of research on prompt engineering techniques for Large Language Models (LLMs). We survey key works on few-shot learning, chain-of-thought prompting, and evaluation methodologies, providing theoretical grounding for our experimental investigation.

---

## 1. Introduction to Prompt Engineering

Prompt engineering has emerged as a critical technique for eliciting desired behaviors from large language models without fine-tuning (Brown et al., 2020; Liu et al., 2023). As models have grown in scale and capability, the formulation of prompts has become increasingly important for task performance.

### 1.1 Definition & Scope

**Prompt engineering** is the practice of designing and optimizing input prompts to guide LLM outputs toward desired results (Reynolds & McDonell, 2021). This encompasses:
- Structure and formatting
- Example selection (for few-shot learning)
- Instruction clarity and specificity
- Reasoning guidance (e.g., chain-of-thought)

---

## 2. Foundational Works

### 2.1 In-Context Learning & Few-Shot Prompting

**Brown et al. (2020)** introduced in-context learning with GPT-3, demonstrating that LLMs can adapt to new tasks from a few examples without gradient updates. Key findings:

- **Few-shot learning:** 10-100 examples in prompt → significant performance gains
- **Task diversity:** Effective across translation, QA, and reasoning tasks
- **Scaling laws:** Larger models show stronger few-shot abilities

**Citation:**
> Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., ... & Amodei, D. (2020). Language Models are Few-Shot Learners. *Advances in Neural Information Processing Systems*, 33, 1877-1901.

**Relevance to This Study:** Our few-shot prompting technique is directly based on this paradigm, providing 2-3 examples before the target question.

---

### 2.2 Chain-of-Thought Prompting

**Wei et al. (2022)** introduced chain-of-thought (CoT) prompting, showing that requesting step-by-step reasoning dramatically improves performance on complex tasks.

**Key Contributions:**
- Explicit reasoning steps → 2-3x improvement on arithmetic/logic tasks
- Emergent ability in large models (>100B parameters)
- Generalizes across domains (math, commonsense, symbolic reasoning)

**Citation:**
> Wei, J., Wang, X., Schuurmans, D., Bosma, M., Chi, E., Le, Q., & Zhou, D. (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. *Conference on Neural Information Processing Systems (NeurIPS)*.

**Relevance:** Our chain-of-thought technique directly implements this approach, requesting "step-by-step" reasoning.

---

### 2.3 Instruction Tuning & Prompting

**Ouyang et al. (2022)** demonstrated that instruction-following can be improved through both training (RLHF) and prompting strategies.

**Citation:**
> Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C. L., Mishkin, P., ... & Lowe, R. (2022). Training language models to follow instructions with human feedback. *Advances in Neural Information Processing Systems*.

---

## 3. Prompt Design Strategies

### 3.1 Role-Based Prompting

**White et al. (2023)** showed that assigning explicit roles ("You are an expert...") improves output quality and alignment.

**Finding:** Role assignment → 10-15% improvement in subjective quality ratings.

**Citation:**
> White, J., Fu, Q., Hays, S., Sandborn, M., Olea, C., Gilbert, H., ... & Schmidt, D. C. (2023). A Prompt Pattern Catalog to Enhance Prompt Engineering with ChatGPT. *arXiv preprint arXiv:2302.11382*.

---

### 3.2 Zero-Shot vs. Few-Shot Trade-offs

**Liu et al. (2023)** conducted comprehensive surveys of prompting techniques, finding:

- Few-shot: Higher accuracy but token-costly
- Zero-shot: More efficient but less reliable
- Optimal shot count: Task-dependent (typically 3-5 for most tasks)

**Citation:**
> Liu, P., Yuan, W., Fu, J., Jiang, Z., Hayashi, H., & Neubig, G. (2023). Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in Natural Language Processing. *ACM Computing Surveys*, 55(9), 1-35.

---

## 4. Evaluation Methodologies

### 4.1 Semantic Similarity Metrics

**Reimers & Gurevych (2019)** introduced Sentence-BERT (SBERT), enabling efficient semantic similarity computation via sentence embeddings.

**Method:** Siamese BERT networks → fixed-size embeddings → cosine similarity

**Advantages:**
- Captures semantic meaning (vs. lexical overlap)
- Efficient for large-scale comparison
- Robust to paraphrasing

**Citation:**
> Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, 3982-3992.

**Relevance:** Our evaluation methodology uses sentence-transformers (SBERT) to compute cosine similarity between LLM outputs and expected answers.

---

### 4.2 Prompt Evaluation Frameworks

**Perez et al. (2021)** developed the "PromptSource" framework for systematic prompt evaluation:

- Multiple prompt templates per task
- Human evaluation + automatic metrics
- Finding: Prompt variance can be as large as model variance

**Citation:**
> Perez, E., Kiela, D., & Cho, K. (2021). True Few-Shot Learning with Language Models. *Advances in Neural Information Processing Systems*, 34, 11054-11070.

---

## 5. Task-Specific Findings

### 5.1 Sentiment Analysis

**Zhao et al. (2021)** studied prompt sensitivity in classification tasks:

- **Finding:** Classification tasks highly sensitive to:
  - Example selection (few-shot)
  - Label wording
  - Template structure

**Citation:**
> Zhao, T. Z., Wallace, E., Feng, S., Klein, D., & Singh, S. (2021). Calibrate Before Use: Improving Few-Shot Performance of Language Models. *International Conference on Machine Learning (ICML)*, 12697-12706.

---

### 5.2 Mathematical Reasoning

**Kojima et al. (2022)** showed that simply adding "Let's think step by step" dramatically improves math reasoning:

- **Zero-shot-CoT:** 40-50% accuracy → 60-80% accuracy on GSM8K
- **No examples needed:** Just the prompting strategy

**Citation:**
> Kojima, T., Gu, S. S., Reid, M., Matsuo, Y., & Iwasawa, Y. (2022). Large Language Models are Zero-Shot Reasoners. *Advances in Neural Information Processing Systems*, 35, 22199-22213.

---

## 6. Theoretical Frameworks

### 6.1 Computational Models of Prompting

**Dohan et al. (2022)** developed a theoretical framework treating prompts as "programs":

- Prompts define computation paths through the model
- Structured prompts reduce search space
- Analogy to software engineering: prompts as APIs

---

### 6.2 Statistical Effect Size

**Cohen (1988)** established conventions for interpreting effect sizes:

- **Small:** d = 0.2
- **Medium:** d = 0.5
- **Large:** d = 0.8

**Citation:**
> Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Hillsdale, NJ: Lawrence Erlbaum Associates.

**Relevance:** We use Cohen's d to quantify the magnitude of prompt technique improvements.

---

## 7. Current Challenges & Gaps

### 7.1 Reproducibility

**Liang et al. (2022)** highlighted reproducibility challenges:
- Model version differences
- Temperature/sampling variations
- API changes over time

**Gap:** Need for controlled, local evaluation (addressed by our Ollama approach).

---

### 7.2 Generalization

Most studies evaluate on:
- Single models
- English only
- Specific task types

**Gap:** Limited cross-model, cross-lingual findings.

---

## 8. Contribution of This Study

Our work contributes to the literature by:

1. **Empirical Quantification:** Systematic comparison with objective metrics
2. **Reproducible Methodology:** Open-source tools (Ollama, sentence-transformers)
3. **Statistical Rigor:** Hypothesis testing, effect sizes, significance tests
4. **Multi-Technique Comparison:** Baseline, Standard, Few-Shot, CoT in one study
5. **Local Execution:** Eliminates API variability and cost

---

## 9. Synthesis & Research Positioning

### Alignment with Prior Work

- **Brown et al. (2020):** We adopt few-shot paradigm
- **Wei et al. (2022):** We implement CoT prompting
- **Reimers & Gurevych (2019):** We use SBERT for evaluation

### Novel Aspects

- **Unified comparison:** Multiple techniques, consistent methodology
- **Statistical analysis:** T-tests, Cohen's d (not just accuracy tables)
- **Open-source stack:** Fully reproducible without proprietary APIs

---

## 10. Conclusion

The literature establishes that prompt engineering significantly impacts LLM performance, with few-shot learning and chain-of-thought prompting showing particular promise. However, most studies lack systematic, quantitative comparisons across techniques. Our work addresses this gap through controlled experimentation with statistical validation.

---

## References

1. Brown, T. B., et al. (2020). Language Models are Few-Shot Learners. *NeurIPS*.

2. Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences*. Routledge.

3. Dohan, D., et al. (2022). Language Model Cascades. *arXiv:2207.10342*.

4. Kojima, T., et al. (2022). Large Language Models are Zero-Shot Reasoners. *NeurIPS*.

5. Liang, P., et al. (2022). Holistic Evaluation of Language Models. *arXiv:2211.09110*.

6. Liu, P., et al. (2023). Pre-train, Prompt, and Predict. *ACM Computing Surveys*, 55(9).

7. Ouyang, L., et al. (2022). Training language models to follow instructions. *NeurIPS*.

8. Perez, E., et al. (2021). True Few-Shot Learning with Language Models. *NeurIPS*.

9. Reimers, N., & Gurevych, I. (2019). Sentence-BERT. *EMNLP*.

10. Reynolds, L., & McDonell, K. (2021). Prompt Programming for Large Language Models. *arXiv:2102.07350*.

11. Wei, J., et al. (2022). Chain-of-Thought Prompting Elicits Reasoning. *NeurIPS*.

12. White, J., et al. (2023). A Prompt Pattern Catalog. *arXiv:2302.11382*.

13. Zhao, T. Z., et al. (2021). Calibrate Before Use. *ICML*.

---

**Document Version:** 1.0
**Total References:** 13
**Last Updated:** December 16, 2025
