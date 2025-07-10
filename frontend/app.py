import os
import sys
from dotenv import load_dotenv
import re 
import time 

last_search_results = []
start_time = time.time()

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
        text_query = data.get('text', '').strip()
        image_data = data.get('image', None)

        logger.debug(f"Received query: '{text_query}', image included: {bool(image_data)}")

        # Handle optional image input
        image = None
        if image_data:
            try:
                image_data_clean = image_data.split(',')[1]
                image_bytes = base64.b64decode(image_data_clean)
                image = Image.open(BytesIO(image_bytes))
                logger.debug("Image processed successfully.")
            except Exception as e:
                logger.error("Error processing image", exc_info=True)
                return jsonify({'success': False, 'error': 'Invalid image data'}), 400

        # Validate that at least a text query or image is provided
        if not text_query and not image:
            return jsonify({'success': False, 'error': 'Must provide text query or image'}), 400

        # Check for greeting-only query
        if is_greeting_only(text_query):
            return jsonify({'success': True, 'answer': "Hello! How can I assist you today?", 'search_results': []})

        # Perform search
        if not rag_system.searcher:
            return jsonify({'success': False, 'error': 'Searcher not initialized'}), 500

        try:
            search_results, response = rag_system.searcher.search(
                text_query=text_query,
                image_query=image,
                top_k=2  # Adjust top_k as needed
            )
        except Exception as e:
            logger.error("Search error", exc_info=True)
            return jsonify({'success': False, 'error': 'Search failed'}), 500

        # Dynamically determine prompting strategy
        query_lower = text_query.lower()
        context_text = "\n\n".join([r.get("content", "") for r in search_results])

        try:
            if not rag_system.llm:
                return jsonify({'success': False, 'error': 'LLM not initialized'}), 500

            if "explain" in query_lower or "how" in query_lower or "why" in query_lower:
                response = rag_system.llm.generate_cot_response(
                    prompt=text_query,
                    context=context_text
                )
                logger.debug("Used Chain-of-Thought prompting.")
            elif "example" in query_lower or "give me" in query_lower:
                examples = [
                    "Q: What is a function?\nA: A reusable block of code to perform a specific task.",
                    "Q: What is a class?\nA: A blueprint for creating objects in OOP."
                ]
                response = rag_system.llm.generate_few_shot(
                    examples=examples,
                    prompt=text_query,
                    context=context_text
                )
                logger.debug("Used Few-Shot prompting.")
            else:
                response = rag_system.llm.generate_response(
                    query=text_query,
                    context=context_text
                )
                logger.debug("Used Zero-Shot prompting.")
        except Exception as e:
            logger.error("LLM response generation error", exc_info=True)
            return jsonify({'success': False, 'error': 'LLM response failed'}), 500

        if not response or response.strip() == "" or len(response.strip()) < 10:
            response = "Sorry, I don't have a relevant answer for that."

        # Prepare frontend search result snippets
        formatted_results = []
        if "outside the scope of the provided knowledge base" not in response.lower():
            for r in search_results:
                formatted_results.append({
                    'title': r.get('title', 'Document Section'),
                    'snippet': r.get('content', '')[:200] + '...',
                    'score': r.get('similarity_score', 0.0),
                    'source': r.get('source_file', 'Unknown'),
                    'page': r.get('page_number', 1),
                    'type': r.get('content_type', 'text')
                })

        global last_search_results
        last_search_results = formatted_results

        query_duration = round(time.time() - start_time, 3)

        return jsonify({
            'success': True,
            'answer': response,
            'search_results': formatted_results,
            'query_time': query_duration
        })

    except Exception as e:
        logger.error("Unhandled exception in /api/query", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500





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

def is_greeting_only(query: str) -> bool:
    greeting_patterns = [
        r"\bhi\b", r"\bhello\b", r"\bhey\b", r"\bgood (morning|evening|afternoon)\b",
        r"\bsalaam\b", r"\bassalamualaikum\b", r"\bnamaste\b", r"\bhowdy\b",
        r"\bhow (are|r) (you|ya)\b", r"\bwhat'?s up\b", r"\bhow's it going\b", r"\bgreetings\b", r"\bthanks\b", r"\bthank you\b",
        r"\bthank you very much\b", r"\bappreciate it\b", r"\bappreciate your help\b", r"\bappreciate your assistance\b",
        r"\bappreciate your support\b", r"\bappreciate your time\b", r"\bappreciate your response\b",
        r"\bappreciate your input\b", r"\bappreciate your feedback\b", r"\bappreciate your assistance\b", r"\bappreciate your cooperation\b",
        r"\bappreciate your understanding\b", r"\bappreciate your patience\b", r"\bappreciate your kindness\b",
        r"\bappreciate your generosity\b", r"\bappreciate your thoughtfulness\b", r"\bappreciate your consideration\b",
        r"\bappreciate your effort\b", r"\bappreciate your support\b", r"\bappreciate your help\b", r"\bappreciate your assistance\b",
    ]
    cleaned = query.strip().lower()
    return any(re.search(pattern, cleaned) for pattern in greeting_patterns)

def generate_mock_search_results():
    """Removed mock search results as per user request"""
    return []

def generate_mock_response(query, has_image):
    """Removed mock response as per user request"""
    return "Sorry, I don't have a relevant answer for that."

@app.route('/semantic-viz')
def semantic_visualization():
    global last_search_results
    logger.info("Accessed /semantic-viz route")
    return render_template('semantic_viz.html', results=last_search_results)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

