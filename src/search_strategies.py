"""
Search Strategies for RAG System.

Implements Strategy pattern for different search algorithms:
- Semantic Search: Using vector embeddings
- Keyword Search: Using traditional keyword matching
- Hybrid Search: Combining both approaches
"""

import re
from typing import List, Tuple, Dict, Any, Optional
from abc import ABC, abstractmethod
import numpy as np

from patterns import Strategy, StrategyContext, AdaptiveStrategy


class SearchStrategy(Strategy):
    """
    Base class for search strategies.

    All search strategies must implement this interface.
    """

    def __init__(self, documents: List[Dict], embedding_model=None):
        """
        Initialize search strategy.

        Args:
            documents: List of document chunks
            embedding_model: Optional embedding model for semantic search
        """
        self.documents = documents
        self.embedding_model = embedding_model
        self._embeddings: Optional[np.ndarray] = None

    @property
    def embeddings(self) -> Optional[np.ndarray]:
        """Get document embeddings."""
        return self._embeddings

    @embeddings.setter
    def embeddings(self, value: np.ndarray) -> None:
        """Set document embeddings."""
        self._embeddings = value

    @abstractmethod
    def search(self, query: str, top_k: int = 3,
               min_score: float = 0.0) -> List[Tuple[Dict, float]]:
        """
        Search for relevant documents.

        Args:
            query: Search query
            top_k: Number of results to return
            min_score: Minimum score threshold

        Returns:
            List of (document, score) tuples
        """
        pass

    def get_name(self) -> str:
        """Get strategy name."""
        return self.__class__.__name__


class SemanticSearchStrategy(SearchStrategy):
    """
    Semantic search using vector embeddings.

    Uses cosine similarity between query and document embeddings.
    """

    def search(self, query: str, top_k: int = 3,
               min_score: float = 0.0) -> List[Tuple[Dict, float]]:
        """
        Perform semantic search.

        Args:
            query: Search query
            top_k: Number of results
            min_score: Minimum similarity score

        Returns:
            Ranked list of documents with scores
        """
        if not self.documents:
            return []

        if self._embeddings is None or self.embedding_model is None:
            # Fall back to keyword search if no embeddings
            return KeywordSearchStrategy(self.documents).search(query, top_k, min_score)

        # Encode query
        query_emb = self.embedding_model.encode([query], is_query=True)[0]

        # Calculate similarities
        similarities = np.dot(self._embeddings, query_emb)

        # Get top indices
        top_indices = np.argsort(similarities)[-top_k * 2:][::-1]

        # Filter by min_score
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score >= min_score:
                results.append((self.documents[idx], score))

        return results[:top_k]

    def get_description(self) -> str:
        """Get strategy description."""
        return "Semantic search using vector embeddings and cosine similarity"


class KeywordSearchStrategy(SearchStrategy):
    """
    Keyword-based search using traditional matching.

    Uses word overlap and phrase matching for scoring.
    """

    def search(self, query: str, top_k: int = 3,
               min_score: float = 0.0) -> List[Tuple[Dict, float]]:
        """
        Perform keyword search.

        Args:
            query: Search query
            top_k: Number of results
            min_score: Minimum score threshold

        Returns:
            Ranked list of documents with scores
        """
        if not self.documents:
            return []

        # Tokenize query
        query_words = set(query.lower().split())
        query_phrases = self._extract_phrases(query)

        scored = []

        for doc in self.documents:
            content_lower = doc["content"].lower()
            content_words = set(content_lower.split())

            # Word overlap score
            matches = query_words & content_words
            word_score = len(matches)

            # Phrase match bonus
            phrase_score = 0
            for phrase in query_phrases:
                if phrase.lower() in content_lower:
                    phrase_score += 2

            # Combined score
            total_score = word_score + (phrase_score * 0.5)

            if total_score > min_score:
                scored.append((doc, total_score))

        # Sort by score and return top-k
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def _extract_phrases(self, query: str) -> List[str]:
        """Extract quoted phrases from query."""
        phrases = re.findall(r'"([^"]+)"', query)
        return phrases

    def get_description(self) -> str:
        """Get strategy description."""
        return "Keyword search using word overlap and phrase matching"


class HybridSearchStrategy(SearchStrategy):
    """
    Hybrid search combining semantic and keyword approaches.

    Merges results from both strategies with combined scoring.
    """

    def __init__(self, documents: List[Dict], embedding_model=None,
                 semantic_weight: float = 0.7):
        """
        Initialize hybrid search.

        Args:
            documents: List of document chunks
            embedding_model: Optional embedding model
            semantic_weight: Weight for semantic search (0-1)
        """
        super().__init__(documents, embedding_model)
        self.semantic_weight = semantic_weight
        self.keyword_weight = 1.0 - semantic_weight

        # Create sub-strategies
        self.semantic_strategy = SemanticSearchStrategy(documents, embedding_model)
        self.semantic_strategy.embeddings = self._embeddings
        self.keyword_strategy = KeywordSearchStrategy(documents)

    def search(self, query: str, top_k: int = 3,
               min_score: float = 0.0) -> List[Tuple[Dict, float]]:
        """
        Perform hybrid search.

        Args:
            query: Search query
            top_k: Number of results
            min_score: Minimum score threshold

        Returns:
            Ranked list of documents with combined scores
        """
        if not self.documents:
            return []

        # Get results from both strategies
        semantic_results = self.semantic_strategy.search(query, top_k * 2, min_score)
        keyword_results = self.keyword_strategy.search(query, top_k * 2, min_score)

        # Combine scores
        combined_scores: Dict[str, Tuple[Dict, float]] = {}

        # Normalize and weight semantic scores
        if semantic_results:
            max_semantic = max(score for _, score in semantic_results)
            for doc, score in semantic_results:
                doc_id = doc.get('chunk_id', id(doc))
                normalized_score = (score / max_semantic) if max_semantic > 0 else 0
                combined_scores[doc_id] = (doc, normalized_score * self.semantic_weight)

        # Add keyword scores
        if keyword_results:
            max_keyword = max(score for _, score in keyword_results)
            for doc, score in keyword_results:
                doc_id = doc.get('chunk_id', id(doc))
                normalized_score = (score / max_keyword) if max_keyword > 0 else 0
                weighted_score = normalized_score * self.keyword_weight

                if doc_id in combined_scores:
                    # Combine scores
                    doc, existing_score = combined_scores[doc_id]
                    combined_scores[doc_id] = (doc, existing_score + weighted_score)
                else:
                    combined_scores[doc_id] = (doc, weighted_score)

        # Filter by min_score and sort
        results = [(doc, score) for doc, score in combined_scores.values()
                   if score >= min_score]
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_k]

    def get_description(self) -> str:
        """Get strategy description."""
        return f"Hybrid search (semantic: {self.semantic_weight:.0%}, keyword: {self.keyword_weight:.0%})"


class FuzzySearchStrategy(SearchStrategy):
    """
    Fuzzy search using approximate string matching.

    Useful for handling typos and variations in spelling.
    """

    def __init__(self, documents: List[Dict], embedding_model=None,
                 fuzzy_threshold: int = 2):
        """
        Initialize fuzzy search.

        Args:
            documents: List of document chunks
            embedding_model: Optional (not used for fuzzy search)
            fuzzy_threshold: Maximum edit distance for fuzzy match
        """
        super().__init__(documents, embedding_model)
        self.fuzzy_threshold = fuzzy_threshold

    def search(self, query: str, top_k: int = 3,
               min_score: float = 0.0) -> List[Tuple[Dict, float]]:
        """
        Perform fuzzy search.

        Args:
            query: Search query
            top_k: Number of results
            min_score: Minimum score threshold

        Returns:
            Ranked list of documents with fuzzy match scores
        """
        if not self.documents:
            return []

        try:
            from rapidfuzz import fuzz, process
        except ImportError:
            # Fall back to keyword search if rapidfuzz not available
            return KeywordSearchStrategy(self.documents).search(query, top_k, min_score)

        # Extract query words
        query_words = query.lower().split()

        scored = []

        for doc in self.documents:
            content = doc["content"].lower()

            # Calculate fuzzy scores for each query word
            word_scores = []
            for word in query_words:
                # Find best match in document
                result = process.extractOne(
                    word,
                    content.split(),
                    scorer=fuzz.ratio
                )
                if result:
                    match_score = result[1]
                    if match_score >= 70:  # Minimum similarity threshold
                        word_scores.append(match_score)

            if word_scores:
                # Average word scores
                avg_score = sum(word_scores) / len(word_scores)

                # Normalize to 0-1
                normalized_score = avg_score / 100.0

                if normalized_score >= min_score:
                    scored.append((doc, normalized_score))

        # Sort by score and return top-k
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def get_description(self) -> str:
        """Get strategy description."""
        return f"Fuzzy search with edit distance threshold of {self.fuzzy_threshold}"


class SearchContext(StrategyContext):
    """
    Context class for search strategies.

    Manages search strategy selection and execution.
    """

    def __init__(self, documents: List[Dict], embedding_model=None):
        """
        Initialize search context.

        Args:
            documents: List of document chunks
            embedding_model: Optional embedding model
        """
        super().__init__()
        self.documents = documents
        self.embedding_model = embedding_model

        # Create strategies
        self._semantic = SemanticSearchStrategy(documents, embedding_model)
        self._keyword = KeywordSearchStrategy(documents)
        self._hybrid = HybridSearchStrategy(documents, embedding_model)

    def search(self, query: str, top_k: int = 3,
               min_score: float = 0.0) -> List[Tuple[Dict, float]]:
        """
        Execute search using current strategy.

        Args:
            query: Search query
            top_k: Number of results
            min_score: Minimum score threshold

        Returns:
            Search results
        """
        # Ensure strategy has current data
        if self._strategy:
            if hasattr(self._strategy, 'embeddings'):
                self._strategy.embeddings = self._embeddings
            return self._strategy.search(query, top_k, min_score)

        # Default to keyword search
        return self._keyword.search(query, top_k, min_score)

    @property
    def embeddings(self) -> Optional[np.ndarray]:
        """Get embeddings."""
        if self._strategy and hasattr(self._strategy, 'embeddings'):
            return self._strategy.embeddings
        return None

    @embeddings.setter
    def embeddings(self, value: np.ndarray) -> None:
        """Set embeddings on current strategy."""
        if self._strategy and hasattr(self._strategy, 'embeddings'):
            self._strategy.embeddings = value
        if hasattr(self, '_semantic'):
            self._semantic.embeddings = value
            self._hybrid.embeddings = value

    def use_semantic_search(self) -> None:
        """Use semantic search strategy."""
        self.set_strategy(self._semantic)

    def use_keyword_search(self) -> None:
        """Use keyword search strategy."""
        self.set_strategy(self._keyword)

    def use_hybrid_search(self, semantic_weight: float = 0.7) -> None:
        """
        Use hybrid search strategy.

        Args:
            semantic_weight: Weight for semantic search (0-1)
        """
        self._hybrid = HybridSearchStrategy(
            self.documents, self.embedding_model, semantic_weight
        )
        self._hybrid.embeddings = self.embeddings
        self.set_strategy(self._hybrid)


class AdaptiveSearchStrategy(AdaptiveStrategy):
    """
    Adaptive search that selects the best strategy based on query characteristics.

    Analyzes the query to determine which search strategy would be most effective.
    """

    def __init__(self, documents: List[Dict], embedding_model=None):
        """
        Initialize adaptive search.

        Args:
            documents: List of document chunks
            embedding_model: Optional embedding model
        """
        super().__init__()
        self.documents = documents
        self.embedding_model = embedding_model

        # Register strategies
        self.register_strategy(
            'semantic',
            SemanticSearchStrategy(documents, embedding_model),
            lambda q: self._should_use_semantic(q)
        )
        self.register_strategy(
            'keyword',
            KeywordSearchStrategy(documents),
            lambda q: self._should_use_keyword(q)
        )
        self.register_strategy(
            'hybrid',
            HybridSearchStrategy(documents, embedding_model),
            lambda q: True  # Default fallback
        )

    def _should_use_semantic(self, query: str) -> bool:
        """Check if semantic search is appropriate."""
        # Use semantic for natural language queries
        query_lower = query.lower()

        # Indicators of semantic intent
        semantic_indicators = [
            'what', 'how', 'why', 'when', 'where', 'who', 'explain',
            'describe', 'compare', 'summarize', 'analyze'
        ]

        return any(indicator in query_lower for indicator in semantic_indicators)

    def _should_use_keyword(self, query: str) -> bool:
        """Check if keyword search is appropriate."""
        # Use keyword for exact matches, quoted phrases, technical terms
        has_quotes = '"' in query or "'" in query
        has_special_chars = bool(re.search(r'[<>{}[\]|\\]', query))

        # Check for technical terms (capitalized, acronyms)
        has_technical = bool(re.search(r'\b[A-Z]{2,}\b', query))

        return has_quotes or has_special_chars or has_technical


def create_search_context(documents: List[Dict],
                         embedding_model=None,
                         strategy: str = 'hybrid') -> SearchContext:
    """
    Create a search context with the specified strategy.

    Args:
        documents: List of document chunks
        embedding_model: Optional embedding model
        strategy: Strategy to use ('semantic', 'keyword', 'hybrid')

    Returns:
        Configured SearchContext
    """
    context = SearchContext(documents, embedding_model)

    if strategy == 'semantic':
        context.use_semantic_search()
    elif strategy == 'keyword':
        context.use_keyword_search()
    else:  # hybrid
        context.use_hybrid_search()

    return context
