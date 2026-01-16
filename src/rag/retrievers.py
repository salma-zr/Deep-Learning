"""Retriever implementations for RAG."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import time

from src.utils.logger import get_logger
from src.utils.cache import get_cache

logger = get_logger(__name__)


@dataclass
class RetrievalResult:
    """Result from a retrieval operation."""
    
    query: str
    documents: list[dict]  # Each dict has: content, source, score (optional)
    latency_ms: float
    error: Optional[str] = None
    
    def get_context(self, max_docs: int = 3, max_chars: int = 2000) -> str:
        """Get formatted context string from retrieved documents."""
        if not self.documents:
            return ""
        
        context_parts = []
        total_chars = 0
        
        for i, doc in enumerate(self.documents[:max_docs]):
            content = doc.get("content", "")
            source = doc.get("source", f"Source {i+1}")
            
            # Truncate if needed
            remaining = max_chars - total_chars
            if remaining <= 0:
                break
            
            if len(content) > remaining:
                content = content[:remaining] + "..."
            
            context_parts.append(f"[{source}]: {content}")
            total_chars += len(content)
        
        return "\n\n".join(context_parts)


class BaseRetriever(ABC):
    """Base class for retrievers."""
    
    def __init__(self, use_cache: bool = True, cache_namespace: str = "retrieval"):
        self.use_cache = use_cache
        self.cache = get_cache(cache_namespace) if use_cache else None
    
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 3) -> RetrievalResult:
        """Retrieve relevant documents for a query."""
        pass
    
    def _get_cache_key(self, query: str, top_k: int) -> str:
        """Generate cache key for a query."""
        import hashlib
        key_str = f"{self.__class__.__name__}:{query}:{top_k}"
        return hashlib.sha256(key_str.encode()).hexdigest()[:32]


class WikipediaRetriever(BaseRetriever):
    """Retriever using Wikipedia API."""
    
    def __init__(
        self,
        language: str = "en",
        use_cache: bool = True,
    ):
        super().__init__(use_cache, "wikipedia")
        self.language = language
        
        try:
            import wikipediaapi
            self.wiki = wikipediaapi.Wikipedia(
                language=language,
                user_agent="MedicalQAProject/1.0 (research; educational)"
            )
            self._available = True
        except ImportError:
            logger.warning("wikipedia-api not installed")
            self._available = False
    
    def retrieve(self, query: str, top_k: int = 3) -> RetrievalResult:
        """Retrieve from Wikipedia."""
        if not self._available:
            return RetrievalResult(
                query=query,
                documents=[],
                latency_ms=0,
                error="Wikipedia API not available",
            )
        
        # Check cache
        if self.use_cache:
            cache_key = self._get_cache_key(query, top_k)
            cached = self.cache.get(cache_key)
            if cached is not None:
                return RetrievalResult(**cached)
        
        start_time = time.perf_counter()
        documents = []
        error = None
        
        try:
            # Search Wikipedia
            # Extract key medical terms from query
            search_terms = self._extract_search_terms(query)
            
            for term in search_terms[:top_k]:
                page = self.wiki.page(term)
                
                if page.exists():
                    # Get summary (first section)
                    summary = page.summary[:1500] if page.summary else ""
                    
                    if summary:
                        documents.append({
                            "content": summary,
                            "source": f"Wikipedia: {page.title}",
                            "url": page.fullurl,
                        })
                
                if len(documents) >= top_k:
                    break
            
        except Exception as e:
            error = str(e)
            logger.warning(f"Wikipedia retrieval error: {e}")
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        result = RetrievalResult(
            query=query,
            documents=documents,
            latency_ms=latency_ms,
            error=error,
        )
        
        # Cache result
        if self.use_cache and error is None:
            self.cache.set(cache_key, {
                "query": query,
                "documents": documents,
                "latency_ms": latency_ms,
                "error": error,
            })
        
        return result
    
    def _extract_search_terms(self, query: str) -> list[str]:
        """Extract search terms from query."""
        # Simple extraction - get nouns and medical terms
        # In production, use NLP/NER for better extraction
        
        # Remove common question words
        stopwords = {
            "what", "is", "are", "the", "a", "an", "of", "in", "to", "for",
            "how", "why", "when", "where", "which", "does", "do", "can",
            "will", "would", "should", "could", "has", "have", "had",
            "this", "that", "these", "those", "be", "been", "being",
        }
        
        words = query.lower().replace("?", "").replace(".", "").split()
        terms = [w for w in words if w not in stopwords and len(w) > 2]
        
        # Also try the full query as a search term
        full_query = " ".join(terms[:5])
        
        return [full_query] + terms


class WebRetriever(BaseRetriever):
    """Retriever using DuckDuckGo web search."""
    
    def __init__(
        self,
        use_cache: bool = True,
        max_results: int = 5,
    ):
        super().__init__(use_cache, "web")
        self.max_results = max_results
        
        try:
            from duckduckgo_search import DDGS
            self.ddgs = DDGS()
            self._available = True
        except ImportError:
            logger.warning("duckduckgo-search not installed")
            self._available = False
    
    def retrieve(self, query: str, top_k: int = 3) -> RetrievalResult:
        """Retrieve from web search."""
        if not self._available:
            return RetrievalResult(
                query=query,
                documents=[],
                latency_ms=0,
                error="DuckDuckGo search not available",
            )
        
        # Check cache
        if self.use_cache:
            cache_key = self._get_cache_key(query, top_k)
            cached = self.cache.get(cache_key)
            if cached is not None:
                return RetrievalResult(**cached)
        
        start_time = time.perf_counter()
        documents = []
        error = None
        
        try:
            # Add medical context to search
            medical_query = f"medical {query}"
            
            results = list(self.ddgs.text(
                medical_query,
                max_results=min(top_k, self.max_results),
            ))
            
            for result in results:
                documents.append({
                    "content": result.get("body", ""),
                    "source": result.get("title", "Web"),
                    "url": result.get("href", ""),
                })
            
        except Exception as e:
            error = str(e)
            logger.warning(f"Web search error: {e}")
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        result = RetrievalResult(
            query=query,
            documents=documents,
            latency_ms=latency_ms,
            error=error,
        )
        
        # Cache result
        if self.use_cache and error is None:
            self.cache.set(cache_key, {
                "query": query,
                "documents": documents,
                "latency_ms": latency_ms,
                "error": error,
            })
        
        return result


class HybridRetriever(BaseRetriever):
    """Combines multiple retrievers."""
    
    def __init__(
        self,
        retrievers: list[BaseRetriever],
        use_cache: bool = True,
    ):
        super().__init__(use_cache, "hybrid")
        self.retrievers = retrievers
    
    def retrieve(self, query: str, top_k: int = 3) -> RetrievalResult:
        """Retrieve from all retrievers and merge results."""
        start_time = time.perf_counter()
        
        all_documents = []
        errors = []
        
        # Get documents from each retriever
        docs_per_retriever = max(1, top_k // len(self.retrievers))
        
        for retriever in self.retrievers:
            result = retriever.retrieve(query, top_k=docs_per_retriever)
            all_documents.extend(result.documents)
            
            if result.error:
                errors.append(result.error)
        
        # Deduplicate by source
        seen_sources = set()
        unique_docs = []
        for doc in all_documents:
            source = doc.get("source", "")
            if source not in seen_sources:
                seen_sources.add(source)
                unique_docs.append(doc)
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        return RetrievalResult(
            query=query,
            documents=unique_docs[:top_k],
            latency_ms=latency_ms,
            error="; ".join(errors) if errors else None,
        )


def get_retriever(
    retriever_type: str = "wikipedia",
    **kwargs,
) -> BaseRetriever:
    """Factory function to create a retriever."""
    if retriever_type == "wikipedia":
        return WikipediaRetriever(**kwargs)
    elif retriever_type == "web":
        return WebRetriever(**kwargs)
    elif retriever_type == "hybrid":
        retrievers = [
            WikipediaRetriever(**kwargs),
            WebRetriever(**kwargs),
        ]
        return HybridRetriever(retrievers, **kwargs)
    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}")
