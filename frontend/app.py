import os
import sys
from dotenv import load_dotenv
import re 

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
        
        query_type = "Text + Image" if text_query and image_data else "Text only" if text_query else "Image only"
        logger.debug(f"📨 Query type: {query_type}")
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
                image_base64_for_embedding = image_data_clean  # ✅ preserve base64 string for embedding
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
        
        # Check if it's a greeting-only query
        greeting_only = is_greeting_only(text_query)

        search_results = []
        context_text = ""
        if not greeting_only:
            try:
                search_results, _ = rag_system.searcher.search(
                    text_query=text_query if text_query.strip() else None,
                    image_query=image_base64_for_embedding if image else None,
                    top_k=2,  # Adjust top_k as needed
                )
                logger.debug(f"Search results obtained: {len(search_results)} items")
                context_text = "\n\n".join([result.get('content', '') for result in search_results])
            except Exception as e:
                logger.error(f"Error during search: {e}", exc_info=True)
                return jsonify({'success': False, 'error': 'Search failed'}), 500
        else:
            logger.debug("Greeting-only message detected. Skipping document search.")

        
        if not rag_system.llm:
            logger.error("RAG LLM not initialized")
            return jsonify({'success': False, 'error': 'LLM not initialized'}), 500
        
        try:

            # Combine search results content as context string
            context_text = "\n\n".join([result.get('content', '') for result in search_results])

            if not text_query.strip():
                # Fallback prompt if no text provided (image-only query)
                text_query = "Describe the image or related content from the documents."

            # Dynamically choose prompting technique
            query_lower = text_query.lower()

            if "explain" in query_lower or "how" in query_lower:
                response = rag_system.llm.generate_cot_response(
                    prompt=text_query,
                    context=context_text
                )
                logger.debug("Using Chain-of-Thought prompting.")
                
            elif "example" in query_lower or "give me" in query_lower:
                few_shot_examples = [
                    "Q: What is a database?\nA: A database is an organized collection of data.",
                    "Q: What is a query?\nA: A query is a request for data from a database."
                ]
                response = rag_system.llm.generate_few_shot(
                    examples=few_shot_examples,
                    prompt=text_query,
                    context=context_text
                )
                logger.debug("Using Few-Shot prompting.")

            else:
                response = rag_system.llm.generate_response(
                    query=text_query,
                    context=context_text
                )
                logger.debug("Using Zero-Shot prompting.")
        
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}", exc_info=True)
            return jsonify({'success': False, 'error': 'Response generation failed'}), 500
        
        
        # Check if response is relevant (non-empty and meaningful)
        if not response or response.strip() == "" or len(response.strip()) < 10:
            response = "Sorry, I don't have a relevant answer for that."
        
        # ✅ Sort search results by similarity score (descending)
        search_results = sorted(search_results, key=lambda x: x.get('similarity_score', 0), reverse=True)


        # Format results for frontend
        # formatted_results = []
        # if "outside the scope of the provided knowledge base" not in response.lower():
        #     for result in search_results:
        #         formatted_results.append({
        #             'title': result.get('title', 'Document Section'),
        #             'snippet': result.get('content', '')[:200] + '...',
        #             'score': result.get('similarity_score', 0.0),
        #             'source': result.get('source_file', 'Unknown'),
        #             'page': result.get('page_number', 1),
        #             'type': result.get('content_type', 'text')
        #         })

        # ✅ Enhanced filtering: only include search results if LLM considers query relevant
        formatted_results = []

        irrelevant_phrases = [
            "outside the scope of the provided knowledge base",
            "i don't have a relevant answer",
            "this topic is not covered",
            "sorry, this topic is outside"
        ]

        if not any(phrase in response.lower() for phrase in irrelevant_phrases):
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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)


@app.route('/semantic-viz')
def semantic_visualization():
    sample_results = [
        {"title": "Page 1", "score": 0.85},
        {"title": "Page 2", "score": 0.75},
        {"title": "Page 3", "score": 0.60}
    ]
    return render_template("semantic_viz.html", results=sample_results)
