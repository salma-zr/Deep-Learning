"""RAG Generator combining retrieval and generation."""

import time
from typing import Optional
from dataclasses import dataclass, field

from src.rag.retrievers import BaseRetriever, RetrievalResult, get_retriever
from src.generation.base import BaseGenerator, GenerationResult
from src.generation.prompts import load_rag_prompt, get_prompt_manager
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RAGResult:
    """Result from RAG generation."""
    
    prediction: str
    model: str
    prompt_id: str
    latency_ms: float
    retrieval_latency_ms: float
    generation_latency_ms: float
    token_usage: Optional[dict] = None
    cost_estimate: Optional[float] = None
    retrieval_info: dict = field(default_factory=dict)
    error: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "prediction": self.prediction,
            "model": self.model,
            "prompt_id": self.prompt_id,
            "latency_ms": round(self.latency_ms, 2),
            "retrieval_latency_ms": round(self.retrieval_latency_ms, 2),
            "generation_latency_ms": round(self.generation_latency_ms, 2),
            "token_usage": self.token_usage,
            "cost_estimate": self.cost_estimate,
            "retrieval_info": self.retrieval_info,
            "error": self.error,
        }


class RAGGenerator:
    """
    RAG Generator that combines retrieval with LLM generation.
    
    Flow:
    1. Retrieve relevant documents for the question
    2. Format context + question into a prompt
    3. Generate answer using the LLM
    """
    
    def __init__(
        self,
        generator: BaseGenerator,
        retriever: Optional[BaseRetriever] = None,
        retriever_type: str = "wikipedia",
        top_k: int = 3,
        max_context_chars: int = 2000,
        prompt_template: Optional[str] = None,
    ):
        """
        Initialize RAG generator.
        
        Args:
            generator: The LLM generator to use
            retriever: Optional pre-configured retriever
            retriever_type: Type of retriever if not provided ("wikipedia", "web", "hybrid")
            top_k: Number of documents to retrieve
            max_context_chars: Maximum characters for context
            prompt_template: Custom RAG prompt template
        """
        self.generator = generator
        self.retriever = retriever or get_retriever(retriever_type)
        self.top_k = top_k
        self.max_context_chars = max_context_chars
        
        # Load default RAG prompt if not provided
        if prompt_template is None:
            try:
                self.prompt_template = load_rag_prompt()
            except FileNotFoundError:
                self.prompt_template = self._default_prompt()
        else:
            self.prompt_template = prompt_template
    
    def _default_prompt(self) -> str:
        """Default RAG prompt template."""
        return """Based on the following context, answer the medical question in ONE concise sentence.

Context:
{context}

Question: {question}

Answer in one sentence:"""
    
    def generate(
        self,
        question: str,
        additional_context: Optional[str] = None,
    ) -> RAGResult:
        """
        Generate an answer using RAG.
        
        Args:
            question: The question to answer
            additional_context: Optional additional context to include
            
        Returns:
            RAGResult with prediction and metadata
        """
        start_time = time.perf_counter()
        
        # Step 1: Retrieve
        retrieval_start = time.perf_counter()
        retrieval_result = self.retriever.retrieve(question, top_k=self.top_k)
        retrieval_latency = (time.perf_counter() - retrieval_start) * 1000
        
        # Build context
        context = retrieval_result.get_context(
            max_docs=self.top_k,
            max_chars=self.max_context_chars,
        )
        
        if additional_context:
            context = f"{additional_context}\n\n{context}"
        
        # Handle no retrieval results
        if not context.strip():
            context = "No relevant context found. Answer based on general medical knowledge."
        
        # Step 2: Format prompt
        prompt = self.prompt_template.format(
            context=context,
            question=question,
        )
        
        # Step 3: Generate
        generation_start = time.perf_counter()
        gen_result = self.generator.generate(prompt)
        generation_latency = (time.perf_counter() - generation_start) * 1000
        
        total_latency = (time.perf_counter() - start_time) * 1000
        
        # Build retrieval info
        retrieval_info = {
            "num_docs": len(retrieval_result.documents),
            "sources": [d.get("source", "") for d in retrieval_result.documents],
            "retrieval_error": retrieval_result.error,
            "top_k": self.top_k,
        }
        
        return RAGResult(
            prediction=gen_result.prediction,
            model=gen_result.model,
            prompt_id=f"rag_{gen_result.prompt_id}",
            latency_ms=total_latency,
            retrieval_latency_ms=retrieval_latency,
            generation_latency_ms=generation_latency,
            token_usage=gen_result.token_usage,
            cost_estimate=gen_result.cost_estimate,
            retrieval_info=retrieval_info,
            error=gen_result.error,
        )
    
    def batch_generate(
        self,
        questions: list[str],
        show_progress: bool = True,
    ) -> list[RAGResult]:
        """Generate answers for multiple questions."""
        from tqdm import tqdm
        
        results = []
        iterator = tqdm(questions, desc="RAG Generation") if show_progress else questions
        
        for question in iterator:
            result = self.generate(question)
            results.append(result)
        
        return results


def create_rag_generator(
    config: dict,
) -> RAGGenerator:
    """
    Create a RAG generator from config.
    
    Config should include:
        - generator config (backend, model, etc.)
        - retriever_type: "wikipedia", "web", or "hybrid"
        - top_k: number of documents to retrieve
        - max_context_chars: maximum context length
    """
    from src.generation.cli import get_generator
    
    # Create the base generator
    generator = get_generator(config)
    
    # Get RAG-specific config
    retriever_type = config.get("retriever_type", "wikipedia")
    top_k = config.get("top_k", 3)
    max_context_chars = config.get("max_context_chars", 2000)
    
    return RAGGenerator(
        generator=generator,
        retriever_type=retriever_type,
        top_k=top_k,
        max_context_chars=max_context_chars,
    )
