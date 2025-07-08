import numpy as np
from typing import List, Dict, Union
from src.embeddings.text_image_embeddings import MultimodalEmbeddingManager

class MultimodalRetriever:
    def __init__(self, embedding_manager: MultimodalEmbeddingManager):
        self.manager = embedding_manager
    
    def retrieve(
        self,
        query: str = None,
        image_path: str = None,
        filters: Dict[str, str] = None,
        k: int = 5
    ) -> List[Dict]:
        """
        Unified retrieval for text/image queries with metadata filtering.
        
        Args:
            query: Text query
            image_path: Path to query image
            filters: {"source": "doc1.pdf", "type": "chart"}
            k: Number of results
        
        Returns:
            List of chunks with scores and metadata
        """
        # Get raw results
        if query and image_path:
            results = self._multimodal_search(query, image_path, k)
        elif query:
            results = self.manager.search_similar_text(query, k)
        elif image_path:
            results = self.manager.search_similar_images(image_path, k)
        else:
            raise ValueError("Must provide query or image")
        
        # Apply metadata filters
        if filters:
            results = self._filter_results(results, filters)
        
        # Rerank by combined score
        return self._rerank(results)

    def _multimodal_search(
        self,
        query: str,
        image_path: str,
        k: int
    ) -> List[Dict]:
        """Combine text and image search results"""
        text_results = self.manager.search_similar_text(query, k*2)
        img_results = self.manager.search_similar_images(image_path, k*2)
        
        # Merge and deduplicate
        all_results = text_results + img_results
        unique_results = {r[0]['content']: r for r in all_results}.values()
        
        return sorted(unique_results, key=lambda x: x[1], reverse=True)[:k]

    def _filter_results(
        self,
        results: List[tuple],
        filters: Dict[str, str]
    ) -> List[tuple]:
        """Filter by metadata fields"""
        filtered = []
        for chunk, score in results:
            match = True
            for key, val in filters.items():
                if chunk.get(key) != val:
                    match = False
                    break
            if match:
                filtered.append((chunk, score))
        return filtered

    def _rerank(self, results: List[tuple]) -> List[Dict]:
        """Apply custom ranking logic"""
        reranked = []
        for chunk, score in results:
            # Boost score if chunk has both text and image
            if 'image_data' in chunk and len(chunk['content']) > 10:
                score *= 1.2
            
            reranked.append({
                "content": chunk['content'],
                "score": float(score),
                "type": chunk['type'],
                "source": chunk['source'],
                "page": chunk.get('page'),
                "image_data": chunk.get('image_data')
            })
        
        return sorted(reranked, key=lambda x: x['score'], reverse=True)
