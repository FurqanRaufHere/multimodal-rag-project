import base64
import os
import sys

# Adjust sys.path to allow relative imports when running as script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.embeddings.text_image_embeddings import MultimodalEmbeddingManager
from src.retrieval.retrieval import MultimodalRetriever

import re
from src.llm.llm_integration import MistralLLM

def preprocess_query(query: str) -> str:
    """
    Preprocess the query string by cleaning and normalizing.
    For example, lowercasing, removing extra spaces, special characters etc.
    """
    query = query.lower()
    query = re.sub(r'[^a-z0-9\s]', '', query)
    query = re.sub(r'\s+', ' ', query).strip()
    return query

import os
from dotenv import load_dotenv

load_dotenv()

def main():
    print("=================================================")
    print("Semantic Search CLI with text and image queries")
    print("You can enter a text query, an image file path, or both.")
    print("Press Enter to skip an input.")
    
    text_query = input("Enter text query: ").strip()
    image_path = input("Enter image file path: ").strip()
    filter_source = input("Enter source file filter (optional): ").strip()
    k_input = input("Enter number of results to return (default 5): ").strip()
    
    if not text_query and not image_path:
        print("Error: Provide at least a text query or an image query.")
        return
    
    try:
        k = int(k_input) if k_input else 5
    except ValueError:
        print("Invalid number for results count. Using default 5.")
        k = 5
    
    # Initialize embedding manager and load saved embeddings and indices
    embedding_manager = MultimodalEmbeddingManager()
    embedding_manager.load_embeddings_and_indices()
    
    # Initialize retriever
    retriever = MultimodalRetriever(embedding_manager)
    
    # Prepare filters
    filters = {}
    if filter_source:
        filters["source"] = filter_source
    
    # Helper function to encode image to base64
    def encode_image_to_base64(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    # Encode image if provided
    image_data = None
    if image_path:
        try:
            image_data = encode_image_to_base64(image_path)
        except Exception as e:
            print(f"Error encoding image: {e}")
            return
    
    # Preprocess query
    preprocessed_query = preprocess_query(text_query) if text_query else None
    
    # Perform retrieval
    try:
        results = retriever.retrieve(query=preprocessed_query if preprocessed_query else None, image_path=image_path if image_path else None, filters=filters, k=k)
    except Exception as e:
        print(f"Error during retrieval: {e}")
        return
    
    # Prepare context from retrieved results
    context = ""
    for res in results:
        context += res['content'] + "\n"
    
    # Initialize LLM
    llm = MistralLLM()
    
    # Construct system prompt
    system_prompt = (
        "You are a helpful assistant. Use the following context to answer the query in a clear, structured, and concise manner.\n"
        "Context:\n"
        f"{context}\n"
        "Query:\n"
        f"{text_query}\n"
        "Answer:"
    )
    
    # Generate response from LLM
    response = llm.generate_response(system_prompt)
    
    # Display LLM response
    print("\nGenerated Response:\n")
    print(response)

if __name__ == "__main__":
    main()
