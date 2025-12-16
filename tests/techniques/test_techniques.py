"""Tests for all prompt techniques."""
import pytest
from src.techniques.base import PromptTechnique
from src.techniques.baseline import BaselinePrompt
from src.techniques.standard import StandardPrompt
from src.techniques.few_shot import FewShotPrompt
from src.techniques.chain_of_thought import ChainOfThoughtPrompt


class TestBasePromptTechnique:
    """Tests for base PromptTechnique interface."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that abstract base class cannot be instantiated."""
        with pytest.raises(TypeError):
            PromptTechnique()

    def test_mock_implementation(self):
        """Test that mock implementation works."""
        class MockTechnique(PromptTechnique):
            def format_prompt(self, question: str, context=None) -> str:
                return f"Mock: {question}"

            def get_name(self) -> str:
                return "mock"

        technique = MockTechnique()
        assert technique.format_prompt("test") == "Mock: test"
        assert technique.get_name() == "mock"


class TestBaselinePrompt:
    """Tests for baseline prompt technique."""

    @pytest.fixture
    def baseline(self):
        return BaselinePrompt()

    def test_initialization(self, baseline):
        """Test baseline prompt initialization."""
        assert isinstance(baseline, PromptTechnique)
        assert baseline.get_name() == "baseline"

    def test_format_prompt_simple_question(self, baseline):
        """Test formatting a simple question."""
        question = "What is 2+2?"
        result = baseline.format_prompt(question)

        assert isinstance(result, str)
        assert question in result
        assert "Answer this question:" in result

    def test_format_prompt_with_context(self, baseline):
        """Test that context parameter is accepted (even if not used)."""
        question = "What is the capital?"
        context = {"country": "France"}

        result = baseline.format_prompt(question, context)
        assert isinstance(result, str)
        assert question in result

    def test_format_prompt_none_context(self, baseline):
        """Test with None context."""
        result = baseline.format_prompt("Test question", None)
        assert "Test question" in result

    def test_format_prompt_empty_question(self, baseline):
        """Test with empty question."""
        result = baseline.format_prompt("")
        assert isinstance(result, str)

    def test_format_prompt_special_characters(self, baseline):
        """Test with special characters."""
        question = "What's the answer to: 'x + y = ?'"
        result = baseline.format_prompt(question)
        assert question in result


class TestStandardPrompt:
    """Tests for standard improved prompt technique."""

    @pytest.fixture
    def standard(self):
        return StandardPrompt()

    def test_initialization(self, standard):
        """Test standard prompt initialization."""
        assert isinstance(standard, PromptTechnique)
        assert standard.get_name() == "standard_improved"

    def test_format_prompt_includes_role(self, standard):
        """Test that prompt includes expert role."""
        question = "What is AI?"
        result = standard.format_prompt(question)

        assert "expert" in result.lower()
        assert question in result

    def test_format_prompt_has_structure(self, standard):
        """Test that prompt has clear structure."""
        question = "Test question"
        result = standard.format_prompt(question)

        assert "Question:" in result
        assert "Answer:" in result
        assert question in result

    def test_format_prompt_multiline(self, standard):
        """Test that result is multiline formatted."""
        result = standard.format_prompt("Test")
        assert "\n" in result

    def test_format_prompt_with_context(self, standard):
        """Test with context (should work even if not used)."""
        result = standard.format_prompt("Test", {"key": "value"})
        assert isinstance(result, str)


class TestFewShotPrompt:
    """Tests for few-shot prompt technique."""

    @pytest.fixture
    def few_shot(self):
        return FewShotPrompt()

    @pytest.fixture
    def examples_context(self):
        return {
            "examples": [
                {"question": "What is 1+1?", "answer": "2"},
                {"question": "What is 2+2?", "answer": "4"},
                {"question": "What is 3+3?", "answer": "6"}
            ]
        }

    def test_initialization(self, few_shot):
        """Test few-shot prompt initialization."""
        assert isinstance(few_shot, PromptTechnique)
        assert few_shot.get_name() == "few_shot"

    def test_format_prompt_without_examples(self, few_shot):
        """Test formatting without examples falls back gracefully."""
        question = "What is 4+4?"
        result = few_shot.format_prompt(question)

        assert question in result
        assert "Q:" in result
        assert "A:" in result

    def test_format_prompt_with_examples(self, few_shot, examples_context):
        """Test formatting with examples."""
        question = "What is 5+5?"
        result = few_shot.format_prompt(question, examples_context)

        # Check examples are included
        assert "1+1" in result
        assert "2+2" in result

        # Check question is included
        assert question in result

        # Check formatting
        assert "Q:" in result
        assert "A:" in result

    def test_format_prompt_limits_examples(self, few_shot):
        """Test that only 3 examples are used maximum."""
        context = {
            "examples": [
                {"question": f"Q{i}", "answer": f"A{i}"}
                for i in range(10)  # Provide 10 examples
            ]
        }

        result = few_shot.format_prompt("Test?", context)

        # Count Q: occurrences (should be 4: 3 examples + 1 question)
        count = result.count("Q:")
        assert count == 4

    def test_format_prompt_none_context(self, few_shot):
        """Test with None context."""
        result = few_shot.format_prompt("Test?", None)
        assert isinstance(result, str)
        assert "Test?" in result

    def test_format_prompt_empty_examples(self, few_shot):
        """Test with empty examples list."""
        context = {"examples": []}
        result = few_shot.format_prompt("Test?", context)
        assert "Test?" in result

    def test_format_prompt_context_without_examples_key(self, few_shot):
        """Test with context that doesn't have 'examples' key."""
        context = {"other_key": "value"}
        result = few_shot.format_prompt("Test?", context)
        assert "Test?" in result


class TestChainOfThoughtPrompt:
    """Tests for chain-of-thought prompt technique."""

    @pytest.fixture
    def cot(self):
        return ChainOfThoughtPrompt()

    def test_initialization(self, cot):
        """Test CoT prompt initialization."""
        assert isinstance(cot, PromptTechnique)
        assert cot.get_name() == "chain_of_thought"

    def test_format_prompt_includes_reasoning_instruction(self, cot):
        """Test that prompt includes step-by-step instruction."""
        question = "What is the area of a circle with radius 5?"
        result = cot.format_prompt(question)

        assert "step by step" in result.lower()
        assert question in result

    def test_format_prompt_has_structure(self, cot):
        """Test that prompt has clear structure."""
        question = "Solve for x: 2x + 5 = 15"
        result = cot.format_prompt(question)

        assert "Question:" in result
        assert question in result
        assert "reasoning" in result.lower()

    def test_format_prompt_requests_brevity(self, cot):
        """Test that prompt requests concise reasoning."""
        result = cot.format_prompt("Test")
        assert "concise" in result.lower() or "brief" in result.lower()

    def test_format_prompt_with_context(self, cot):
        """Test with context (should work even if not used)."""
        result = cot.format_prompt("Test", {"key": "value"})
        assert isinstance(result, str)
        assert "Test" in result


class TestTechniqueComparison:
    """Tests comparing different techniques."""

    def test_all_techniques_have_unique_names(self):
        """Test that each technique has a unique name."""
        techniques = [
            BaselinePrompt(),
            StandardPrompt(),
            FewShotPrompt(),
            ChainOfThoughtPrompt()
        ]

        names = [t.get_name() for t in techniques]
        assert len(names) == len(set(names))  # All unique

    def test_all_techniques_format_same_question_differently(self):
        """Test that different techniques produce different prompts."""
        question = "What is machine learning?"

        baseline = BaselinePrompt().format_prompt(question)
        standard = StandardPrompt().format_prompt(question)
        cot = ChainOfThoughtPrompt().format_prompt(question)

        # All should contain the question
        assert question in baseline
        assert question in standard
        assert question in cot

        # But they should be different
        assert baseline != standard
        assert standard != cot
        assert baseline != cot

    def test_all_techniques_accept_context(self):
        """Test that all techniques accept context parameter."""
        techniques = [
            BaselinePrompt(),
            StandardPrompt(),
            FewShotPrompt(),
            ChainOfThoughtPrompt()
        ]

        context = {"test": "value"}

        for technique in techniques:
            # Should not raise an exception
            result = technique.format_prompt("Test?", context)
            assert isinstance(result, str)

    def test_all_techniques_return_non_empty_strings(self):
        """Test that all techniques return non-empty strings."""
        techniques = [
            BaselinePrompt(),
            StandardPrompt(),
            FewShotPrompt(),
            ChainOfThoughtPrompt()
        ]

        for technique in techniques:
            result = technique.format_prompt("Test question")
            assert len(result) > 0
            assert isinstance(result, str)
