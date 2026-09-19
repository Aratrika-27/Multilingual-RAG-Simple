"""
Multilingual generator for the code-mix chatbot.
"""
from typing import Any, List, Mapping, Optional, Dict
import re

# Try to import from transformers
try:
    from transformers import MT5ForConditionalGeneration, MT5Tokenizer, pipeline
except ImportError:
    print("Warning: transformers package not found. Please install it with: pip install transformers")
    # Define placeholder classes to avoid errors
    class MT5ForConditionalGeneration:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            raise ImportError("transformers package not installed")

    class MT5Tokenizer:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            raise ImportError("transformers package not installed")

    def pipeline(*args, **kwargs):
        raise ImportError("transformers package not installed")

# Try to import torch
try:
    import torch
except ImportError:
    print("Warning: torch package not found. Please install it with: pip install torch")
    # Create a placeholder module
    class torch:
        class backends:
            class mps:
                @staticmethod
                def is_available():
                    return False

        @staticmethod
        def cuda_is_available():
            return False

# Try to import from langchain
try:
    # Try newer imports first
    from langchain_core.language_models.llms import LLM
    from langchain_huggingface import HuggingFacePipeline
except ImportError:
    try:
        # Fall back to older imports
        from langchain.llms.base import LLM
        from langchain.llms import HuggingFacePipeline
    except ImportError:
        print("Warning: langchain packages not found. Please install them with: pip install langchain langchain-huggingface")
        # Define placeholder classes
        class LLM:
            def __init__(self, *args, **kwargs):
                pass

        class HuggingFacePipeline:
            def __init__(self, *args, **kwargs):
                pass

from ..config import GENERATOR_MODEL, PROMPT_TEMPLATE, MAX_RESPONSE_LENGTH

# Language detection patterns
LANGUAGE_PATTERNS = {
    "en": r'[a-zA-Z]',  # English
    "hi": r'[\u0900-\u097F]',  # Devanagari (Hindi)
    "bn": r'[\u0980-\u09FF]',  # Bengali
}

class MultilingualGenerator(LLM):
    """Multilingual generator for code-mixed responses."""

    model_name: str = GENERATOR_MODEL

    def __init__(self, model_name: str = GENERATOR_MODEL):
        """Initialize the generator."""
        super().__init__()
        self._model_name = model_name

        # Load tokenizer and model
        print(f"Loading model: {model_name}")
        self._tokenizer = MT5Tokenizer.from_pretrained(model_name)
        self._model = MT5ForConditionalGeneration.from_pretrained(model_name)

        # Move model to appropriate device
        self._device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self._model = self._model.to(self._device)
        print(f"Model loaded on device: {self._device}")

    def detect_languages(self, text: str) -> Dict[str, float]:
        """
        Detect languages in the text and their approximate proportions.

        Args:
            text: Input text

        Returns:
            Dictionary mapping language codes to their proportions in the text
        """
        # Count characters matching each language pattern
        counts = {}
        total_chars = len(text)

        for lang, pattern in LANGUAGE_PATTERNS.items():
            matches = re.findall(pattern, text)
            counts[lang] = len(matches) / max(total_chars, 1)

        return counts

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Generate text based on the input prompt."""
        try:
            # Tokenize the input
            inputs = self._tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            # Generate with appropriate settings
            outputs = self._model.generate(
                **inputs,
                max_length=MAX_RESPONSE_LENGTH,
                num_beams=5,
                length_penalty=1.0,
                early_stopping=True,
                do_sample=True,
                temperature=0.7,  # Add some randomness
                top_p=0.9,  # Nucleus sampling
            )

            # Decode the output
            response = self._tokenizer.decode(outputs[0], skip_special_tokens=True)

            # If the response is empty or too short, provide a fallback
            if not response or len(response) < 10:
                return "I couldn't generate a proper response. Please try rephrasing your question."

            return response

        except Exception as e:
            # Provide a fallback response in case of errors
            return f"I encountered an error while processing your request: {str(e)}. Please try again with a different query."

    @property
    def _llm_type(self) -> str:
        """Return the type of LLM."""
        return "multilingual_generator"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Return identifying parameters."""
        return {"model_name": self._model_name}

def load_generator(model_name: Optional[str] = GENERATOR_MODEL) -> HuggingFacePipeline:
    """
    Load a multilingual generator using the Hugging Face pipeline.

    Args:
        model_name: Name of the model to load

    Returns:
        HuggingFacePipeline for text generation
    """
    # Use default model if None is provided
    if model_name is None:
        model_name = "google/mt5-small"

    print(f"Loading model: {model_name}")

    # Load tokenizer and model
    tokenizer = MT5Tokenizer.from_pretrained(model_name)
    model = MT5ForConditionalGeneration.from_pretrained(model_name)

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)
    print(f"Model loaded on device: {device}")

    # Create pipeline
    gen_pipeline = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=MAX_RESPONSE_LENGTH,
        device=0 if device == "cuda" else -1
    )

    # Wrap in LangChain's HuggingFacePipeline
    return HuggingFacePipeline(pipeline=gen_pipeline)

def build_prompt(context: str, query: str) -> str:
    """Build a prompt for the RAG system."""
    return PROMPT_TEMPLATE.format(context=context, query=query)
