import base64
import os
import sys

# Adjust sys.path to allow relative imports when running as script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.embeddings.text_image_embeddings import MultimodalEmbeddingManager
from src.retrieval.retrieval import MultimodalRetriever

import re
from src.llm.llm_integration import MistralLLM

class SemanticSearcher:
    def __init__(self):
        # Initialize embedding manager and load embeddings
        self.embedding_manager = MultimodalEmbeddingManager()
        self.embedding_manager.load_embeddings_and_indices()
        
        # Initialize retriever
        self.retriever = MultimodalRetriever(self.embedding_manager)
        
        # Initialize LLM
        self.llm = MistralLLM()
    
    def preprocess_query(self, query: str) -> str:
        query = query.lower()
        query = re.sub(r'[^a-z0-9\s]', '', query)
        query = re.sub(r'\s+', ' ', query).strip()
        return query
    
    def search(self, text_query: str = None, image_query = None, top_k: int = 5):
        # Preprocess text query
        preprocessed_query = self.preprocess_query(text_query) if text_query else None
        
        # Perform retrieval
        results = self.retriever.retrieve(
            query=preprocessed_query,
            image_path=None,  # image_query is PIL Image, retriever expects path, so skip image retrieval here
            k=top_k
        )
        
        # Prepare context from retrieved results
        context = ""
        for res in results:
            context += res['content'] + "\n"
        
        # Construct system prompt
        system_prompt = (
            "You are a highly intelligent, helpful assistant. Use the provided context to answer the user query accurately, clearly, and concisely.\n\n"
            "Instructions:\n"
            "- Prefer brevity when the question is straightforward or when the user explicitly requests a short answer.\n"
            "- If the query suggests the user needs an explanation or elaboration, respond in a detailed and structured manner.\n"
            "- Avoid repeating the context. Synthesize and summarize only what’s necessary.\n"
            "- Be direct, respectful, and avoid filler words.\n"
            "- If the context does not contain enough information, politely mention it rather than guessing.\n\n"
            "Context:\n"
            f"{context}\n"
            "Query:\n"
            f"{text_query}\n"
            "Answer:"
        )
        
        # Generate response from LLM
        response = self.llm.generate_response(query=text_query, context=context)
        
        # Format results for frontend
        formatted_results = []
        for res in results:
            formatted_results.append({
                'title': res.get('title', 'Document Section'),
                'content': res.get('content', ''),
                'similarity_score': res.get('score', 0.0),
                'source_file': res.get('source', 'Unknown'),
                'page_number': res.get('page', 1),
                'content_type': res.get('type', 'text')
            })
        
        return formatted_results

if __name__ == "__main__":
    # For CLI testing, you can keep the main function or remove it
    pass
