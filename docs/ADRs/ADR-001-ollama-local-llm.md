# ADR-001: Use Ollama for Local LLM Inference

**Status**: Accepted

**Date**: December 2025

**Decision Makers**: Tal Hibner

---

## Context

The prompt engineering research project requires consistent, reproducible access to Large Language Models (LLMs) for experimentation. We need to evaluate multiple prompt techniques across dozens of examples while maintaining:

1. **Reproducibility**: Consistent results across runs
2. **Cost Control**: No per-token API charges
3. **Privacy**: No data leaving local machine
4. **Offline Operation**: No internet dependency during experiments
5. **Version Control**: Fixed model version for entire study

### Alternatives Considered

1. **OpenAI API** (GPT-3.5/GPT-4)
   - ✅ High quality outputs
   - ✅ Well-documented API
   - ❌ Cost: ~$0.002-0.06 per 1K tokens (expensive for 48+ examples × 4 techniques)
   - ❌ API rate limits
   - ❌ Model versions change over time
   - ❌ Requires internet connection
   - ❌ Data sent to external servers

2. **Anthropic Claude API**
   - ✅ Excellent reasoning capabilities
   - ✅ Long context windows
   - ❌ Similar cost concerns (~$0.008-0.024 per 1K tokens)
   - ❌ API rate limits
   - ❌ Requires API key management
   - ❌ Privacy concerns for research data

3. **Google Gemini API**
   - ✅ Free tier available
   - ✅ Good performance
   - ❌ Quota limits (60 requests/minute)
   - ❌ Still requires internet
   - ❌ Less control over model version

4. **HuggingFace Transformers** (Direct PyTorch/TensorFlow)
   - ✅ Completely local
   - ✅ Free
   - ❌ Requires GPU for reasonable performance
   - ❌ Complex setup (CUDA, model quantization)
   - ❌ Memory intensive (8-16GB+ VRAM for good models)
   - ❌ Slow CPU inference

5. **Ollama** (Selected)
   - ✅ **Local execution**: Complete privacy
   - ✅ **Free**: No API costs
   - ✅ **Simple setup**: One command to install
   - ✅ **CPU-optimized**: Runs well on laptops via quantization
   - ✅ **Reproducible**: Fixed model version (llama3.2)
   - ✅ **HTTP API**: Standard REST interface
   - ✅ **Model management**: Easy model switching (`ollama pull`)
   - ✅ **Active development**: Regular updates, good community
   - ⚠️ **Performance**: Slower than cloud APIs (~2-5s per inference)
   - ⚠️ **Quality**: Good but not GPT-4 level

---

## Decision

**We will use Ollama with the llama3.2 model for all LLM inference in this project.**

### Implementation Details

- **Model**: `llama3.2` (Meta's Llama 3.2 via Ollama)
- **API**: HTTP REST API on `http://localhost:11434`
- **Endpoint**: `POST /api/generate`
- **Configuration**: Default temperature and sampling parameters
- **Installation**: `curl -fsSL https://ollama.ai/install.sh | sh`
- **Model Download**: `ollama pull llama3.2`

### Integration Approach

```python
class OllamaProvider(BaseProvider):
    def __init__(self, model: str = "llama3.2", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    def generate(self, prompt: str) -> str:
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False}
        )
        return response.json()["response"]
```

---

## Consequences

### Positive

1. **Zero Cost**: Unlimited experiments without API charges (critical for iterative research)
2. **Reproducibility**: Same model version throughout study eliminates temporal drift
3. **Privacy**: Sensitive research data never leaves local machine
4. **Offline Work**: No internet required after initial model download
5. **Fast Iteration**: No API rate limits or quotas
6. **Educational Value**: Students can replicate experiments without API keys
7. **Version Control**: Fixed `llama3.2` ensures consistent baselines
8. **Simple Setup**: One-command installation, minimal dependencies

### Negative

1. **Performance**: 2-5 seconds per inference (vs <1s for cloud APIs)
   - **Mitigation**: Acceptable for research context (total runtime ~5 min)
2. **Model Quality**: Not GPT-4 level reasoning
   - **Mitigation**: Sufficient for demonstrating prompt engineering techniques
3. **Hardware Requirements**: Needs ~4GB RAM, benefits from multi-core CPU
   - **Mitigation**: Most modern laptops meet requirements
4. **Limited to Local Machine**: Cannot easily distribute inference across servers
   - **Mitigation**: Not needed for this scale (48 examples)
5. **Model Selection**: Limited to Ollama's model library
   - **Mitigation**: Library includes llama3, mistral, phi, gemma - sufficient variety

### Risks and Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Ollama service crashes during experiment | High | Low | Implement retry logic with exponential backoff |
| Model updates break compatibility | Medium | Low | Pin to specific model version (`llama3.2:latest`) |
| Slow inference blocks experimentation | Medium | Medium | Use progress bars, optimize batch processing |
| Hardware insufficient for model | High | Low | Document minimum requirements, test on target hardware |

---

## Validation

### Success Criteria

- ✅ Successfully runs 48 examples × 4 techniques = 192 inferences
- ✅ Average inference time <10 seconds
- ✅ Consistent output format across all runs
- ✅ No crashes or timeouts during full experiment run
- ✅ Results reproducible across multiple experiment runs

### Testing

```python
def test_ollama_availability():
    """Verify Ollama is running and accessible"""
    response = requests.get("http://localhost:11434/api/tags")
    assert response.status_code == 200
    models = [m["name"] for m in response.json()["models"]]
    assert "llama3.2" in models or any("llama3.2" in m for m in models)
```

---

## References

- Ollama Official Site: https://ollama.ai/
- Llama 3.2 Model Card: https://ai.meta.com/llama/
- Ollama API Documentation: https://github.com/ollama/ollama/blob/main/docs/api.md
- Research on LLM Reproducibility: Liang et al. (2022) - Holistic Evaluation of Language Models

---

## Future Considerations

### When to Reconsider This Decision

1. **Scale Increase**: If experiments grow to >1000 examples, cloud APIs may be faster
2. **Quality Requirements**: If GPT-4 level reasoning becomes necessary
3. **Multi-Model Comparison**: If comparing across providers becomes primary goal
4. **Real-Time Applications**: If sub-second latency is required

### Potential Enhancements

- **Model Switching**: Support multiple Ollama models (llama3, mistral, etc.)
- **Parallel Execution**: Run multiple Ollama instances for concurrent inference
- **Result Caching**: Store LLM responses to avoid re-running identical prompts
- **Hybrid Approach**: Use Ollama for development, cloud APIs for final validation

---

**ADR Status**: ✅ Implemented and Validated

**Last Review**: December 16, 2025
