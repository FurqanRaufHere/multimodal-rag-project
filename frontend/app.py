import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add parent directory to path to import from other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import base64
from io import BytesIO
from PIL import Image
import json

# Import your existing modules
try:
    from src.embeddings.text_image_embeddings import MultimodalEmbeddingManager
    from src.retrieval.retrieval import MultimodalRetriever
    from src.llm.llm_integration import MistralLLM
    from src.semantic.semantic_search import SemanticSearcher
except ImportError as e:
    print(f"Warning: Could not import modules: {e}")
    print("Make sure your modules are properly structured")

import logging

app = Flask(__name__)
CORS(app)

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Debug print to verify API key loading
api_key = os.getenv("MISTRAL_API_KEY")
logger.debug(f"MISTRAL_API_KEY loaded: {api_key is not None and api_key != ''}")

# Initialize your RAG components
class RAGSystem:
    def __init__(self):
        try:
            logger.info("Initializing MultimodalEmbeddingManager...")
            self.embedder = MultimodalEmbeddingManager()
            logger.info("MultimodalEmbeddingManager initialized successfully.")
            
            logger.info("Initializing MultimodalRetriever...")
            self.retriever = MultimodalRetriever(self.embedder)
            logger.info("MultimodalRetriever initialized successfully.")
            
            logger.info("Initializing MistralLLM...")
            self.llm = MistralLLM()
            logger.info("MistralLLM initialized successfully.")
            
            logger.info("Initializing SemanticSearcher...")
            self.searcher = SemanticSearcher()
            logger.info("SemanticSearcher initialized successfully.")
            
            logger.info("RAG System initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing RAG system: {e}", exc_info=True)
            self.embedder = None
            self.retriever = None
            self.llm = None
            self.searcher = None

# Global RAG system instance
rag_system = RAGSystem()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/query', methods=['POST'])
def process_query():
    try:
        data = request.json
        text_query = data.get('text', '')
        image_data = data.get('image', None)
        
        logger.debug(f"Received query: {text_query}, image included: {bool(image_data)}")
        logger.debug(f"Raw text_query: '{text_query}'")
        
        # Process image if provided
        image = None
        if image_data:
            try:
                # Remove data:image/jpeg;base64, prefix
                image_data_clean = image_data.split(',')[1]
                image_bytes = base64.b64decode(image_data_clean)
                image = Image.open(BytesIO(image_bytes))
                logger.debug("Image processed successfully.")
                logger.debug(f"Image object: {image}")
            except Exception as e:
                logger.error(f"Error processing image data: {e}", exc_info=True)
                return jsonify({'success': False, 'error': 'Invalid image data'}), 400
        
        # Validate input: must have either non-empty text_query or image
        if (not text_query or text_query.strip() == "") and not image:
            logger.error("No query or image provided")
            return jsonify({'success': False, 'error': 'Must provide text query or image'}), 400

        # Perform multimodal search
        if not rag_system.searcher:
            logger.error("RAG searcher not initialized")
            return jsonify({'success': False, 'error': 'Searcher not initialized'}), 500
        
        try:
            search_results = rag_system.searcher.search(
                text_query=text_query,
                image_query=image,
                top_k=3,  # Adjust top_k as needed
            )
            logger.debug(f"Search results obtained: {len(search_results)} items")
        except Exception as e:
            logger.error(f"Error during search: {e}", exc_info=True)
            return jsonify({'success': False, 'error': 'Search failed'}), 500
        
        if not rag_system.llm:
            logger.error("RAG LLM not initialized")
            return jsonify({'success': False, 'error': 'LLM not initialized'}), 500
        
        try:
            # Combine search results content as context string
            context_text = "\n\n".join([result.get('content', '') for result in search_results])
            response = rag_system.llm.generate_response(
                query=text_query,
                context=context_text
            )
            logger.debug("LLM response generated successfully.")
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}", exc_info=True)
            return jsonify({'success': False, 'error': 'Response generation failed'}), 500
        
        # Check if response is relevant (non-empty and meaningful)
        if not response or response.strip() == "" or len(response.strip()) < 10:
            response = "Sorry, I don't have a relevant answer for that."
        
        # Format results for frontend
        formatted_results = []
        for result in search_results:
            formatted_results.append({
                'title': result.get('title', 'Document Section'),
                'snippet': result.get('content', '')[:200] + '...',
                'score': result.get('similarity_score', 0.0),
                'source': result.get('source_file', 'Unknown'),
                'page': result.get('page_number', 1),
                'type': result.get('content_type', 'text')
            })
        
        return jsonify({
            'success': True,
            'answer': response,
            'search_results': formatted_results,
            'query_time': 0.5  # You can track actual time
        })
        
    except Exception as e:
        logger.error(f"Unhandled exception in process_query: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/document/<path:filename>')
def serve_document(filename):
    """Serve document files"""
    try:
        # Adjust path to your documents
        doc_path = os.path.join('Data', filename)
        if os.path.exists(doc_path):
            return send_file(doc_path)
        else:
            return jsonify({'error': 'Document not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def generate_mock_search_results():
    """Removed mock search results as per user request"""
    return []

def generate_mock_response(query, has_image):
    """Removed mock response as per user request"""
    return "Sorry, I don't have a relevant answer for that."

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
