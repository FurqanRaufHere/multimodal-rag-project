"""
Text and Image Embedding System
Uses Sentence-BERT, CLIP, and FAISS for multimodal embeddings
"""
import os
import json
import pickle
import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import base64
from io import BytesIO
from datetime import datetime
import uuid

# ML libraries
import torch
import torch.nn.functional as F
from PIL import Image
import faiss
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MultimodalEmbeddingManager:
    """
    Comprehensive embedding system for text and images using free models
    """
    
    def __init__(self, output_dir: str = "embeddings_data"):
        """
        Initialize the embedding manager
        
        Args:
            output_dir: Directory to store embeddings and indices
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Statistics
        self.embedding_stats = {
            'text_embeddings_created': 0,
            'image_embeddings_created': 0,
            'total_chunks_processed': 0,
            'faiss_indices_created': 0,
            'processing_time': 0,
            'model_info': {}
        }
        
        # Create subdirectories
        (self.output_dir / "indices").mkdir(exist_ok=True)
        (self.output_dir / "embeddings").mkdir(exist_ok=True)
        (self.output_dir / "visualizations").mkdir(exist_ok=True)
        
        # Initialize models
        self._initialize_models()
        
        # Initialize FAISS indices
        self._initialize_faiss_indices()
        
        # Storage for chunks and embeddings
        self.text_chunks = []
        self.image_chunks = []
        self.mixed_chunks = []  # For unified search
        
        logger.info("Multimodal Embedding Manager initialized successfully")
    
    def _initialize_models(self):
        """Initialize Sentence-BERT and CLIP models"""
        try:
            logger.info("Loading Sentence-BERT model...")
            # Load Sentence-BERT for text embeddings
            self.text_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.text_dimension = 384  # Dimension for all-MiniLM-L6-v2
            
            logger.info("Loading CLIP model...")
            # Load CLIP for image embeddings
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.image_dimension = 512  # CLIP ViT-B/32 dimension
            
            # Set models to evaluation mode
            self.clip_model.eval()
            
            # Move to GPU if available
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.clip_model.to(self.device)
            
            # Store model info
            self.embedding_stats['model_info'] = {
                'text_model': 'sentence-transformers/all-MiniLM-L6-v2',
                'text_dimension': self.text_dimension,
                'image_model': 'openai/clip-vit-base-patch32',
                'image_dimension': self.image_dimension,
                'device': str(self.device)
            }
            
            logger.info(f"Models loaded successfully on device: {self.device}")
            
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            raise
    
    def _initialize_faiss_indices(self):
        """Initialize FAISS indices for efficient similarity search"""
        try:
            # Text index (Inner Product for cosine similarity)
            self.text_index = faiss.IndexIVFFlat(faiss.IndexFlatIP(self.text_dimension), self.text_dimension, 100)
            self.text_index.train(np.random.rand(1000, self.text_dimension).astype('float32'))
            
            # Image index
            self.image_index = faiss.IndexIVFFlat(faiss.IndexFlatIP(self.image_dimension), self.image_dimension, 100)
            self.image_index.train(np.random.rand(1000, self.image_dimension).astype('float32'))
            
            # Mixed index for unified search (we'll use text dimension)
            self.mixed_index = faiss.IndexIVFFlat(faiss.IndexFlatIP(self.text_dimension), self.text_dimension, 100)
            self.mixed_index.train(np.random.rand(1000, self.text_dimension).astype('float32'))
            
            logger.info("FAISS IVF indices initialized and trained successfully")
            
        except Exception as e:
            logger.error(f"Error initializing FAISS indices: {str(e)}")
            raise
    
    def create_text_embedding(self, text: str) -> np.ndarray:
        """
        Create text embedding using Sentence-BERT
        
        Args:
            text: Input text to embed
            
        Returns:
            Normalized embedding vector
        """
        try:
            # Create embedding
            embedding = self.text_model.encode([text], convert_to_tensor=False)
            embedding = embedding[0]  # Get single embedding
            
            # Normalize for cosine similarity
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding.astype('float32')
            
        except Exception as e:
            logger.error(f"Error creating text embedding: {str(e)}")
            return np.zeros(self.text_dimension, dtype='float32')
    
    def create_image_embedding(self, image_data: str) -> np.ndarray:
        """
        Create image embedding using CLIP
        
        Args:
            image_data: Base64 encoded image data
            
        Returns:
            Normalized embedding vector
        """
        try:
            # Decode base64 image
            img_bytes = base64.b64decode(image_data)
            image = Image.open(BytesIO(img_bytes))
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize image to 224x224 for CLIP ViT-B/32
            image = image.resize((224, 224))
            
            # Process image with CLIP
            inputs = self.clip_processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
                # Normalize
                image_features = F.normalize(image_features, p=2, dim=1)
            
            # Convert to numpy
            embedding = image_features.cpu().numpy().flatten()
            
            # Debug: log norm of embedding
            norm = np.linalg.norm(embedding)
            logger.info(f"Image embedding norm: {norm}")
            
            return embedding.astype('float32')
            
        except Exception as e:
            logger.error(f"Error creating image embedding: {str(e)}")
            return np.zeros(self.image_dimension, dtype='float32')
    
    def create_multimodal_embedding(self, text: str, image_data: str = None) -> np.ndarray:
        """
        Create combined text+image embedding for unified search
        
        Args:
            text: Text content
            image_data: Optional base64 image data
            
        Returns:
            Combined embedding vector
        """
        try:
            # Create text embedding
            text_embedding = self.create_text_embedding(text)
            
            if image_data:
                # Create image embedding
                image_embedding = self.create_image_embedding(image_data)
                
                # Normalize embeddings
                text_embedding = text_embedding / np.linalg.norm(text_embedding)
                image_embedding = image_embedding / np.linalg.norm(image_embedding)
                
                # Concatenate embeddings
                combined = np.concatenate((text_embedding, image_embedding))
                
                # Project to fixed dimension (e.g., 512) using a simple linear projection
                # For simplicity, use PCA or random projection here (can be replaced with learned projection)
                # Here, we use random projection matrix initialized once
                
                if not hasattr(self, 'projection_matrix'):
                    np.random.seed(42)
                    self.projection_matrix = np.random.randn(combined.shape[0], self.text_dimension).astype('float32')
                
                projected = np.dot(combined, self.projection_matrix)
                
                # Normalize projected vector
                projected = projected / np.linalg.norm(projected)
                
                return projected.astype('float32')
            else:
                return text_embedding
                
        except Exception as e:
            logger.error(f"Error creating multimodal embedding: {str(e)}")
            return np.zeros(self.text_dimension, dtype='float32')
    
    def process_chunks(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process extracted chunks and create embeddings
        
        Args:
            chunks: List of chunks from document extraction
            
        Returns:
            Processing results and statistics
        """
        logger.info(f"Processing {len(chunks)} chunks for embedding creation")
        
        start_time = datetime.now()
        
        text_embeddings = []
        image_embeddings = []
        mixed_embeddings = []
        
        processed_chunks = 0
        
        for chunk in chunks:
            try:
                chunk_type = chunk.get('type', 'text')
                
                if chunk_type == 'text' or chunk_type == 'table':
                    # Create text embedding
                    text_embedding = self.create_text_embedding(chunk['content'])
                    text_embeddings.append(text_embedding)
                    
                    # Create mixed embedding (text only)
                    mixed_embedding = self.create_multimodal_embedding(chunk['content'])
                    mixed_embeddings.append(mixed_embedding)
                    
                    # Store chunk with embedding info
                    chunk['embedding_type'] = 'text'
                    chunk['embedding_dimension'] = self.text_dimension
                    self.text_chunks.append(chunk)
                    self.mixed_chunks.append(chunk)
                    
                    self.embedding_stats['text_embeddings_created'] += 1
                
                elif chunk_type in ['image', 'embedded_image', 'page_image', 'docx_image']:
                    # Create image embedding
                    image_data = chunk.get('image_data', '')
                    if not image_data and 'image_path' in chunk:
                        # Load image and convert to base64
                        try:
                            with open(chunk['image_path'], 'rb') as img_file:
                                img_bytes = img_file.read()
                                image_data = base64.b64encode(img_bytes).decode('utf-8')
                                chunk['image_data'] = image_data
                        except Exception as e:
                            logger.error(f"Error loading image file {chunk['image_path']}: {str(e)}")
                            image_data = ''
                    if image_data:
                        image_embedding = self.create_image_embedding(image_data)
                        image_embeddings.append(image_embedding)
                        
                        # Create mixed embedding (text + image) - removed to keep dimension consistent
                        # mixed_embedding = self.create_multimodal_embedding(
                        #     chunk.get('content', ''), image_data
                        # )
                        # mixed_embeddings.append(mixed_embedding)
                        
                        # Store chunk with embedding info
                        chunk['embedding_type'] = 'image'
                        chunk['embedding_dimension'] = self.image_dimension
                        self.image_chunks.append(chunk)
                        self.mixed_chunks.append(chunk)
                        
                        self.embedding_stats['image_embeddings_created'] += 1
                
                processed_chunks += 1
                
                if processed_chunks % 10 == 0:
                    logger.info(f"Processed {processed_chunks}/{len(chunks)} chunks")
                
            except Exception as e:
                logger.error(f"Error processing chunk {chunk.get('id', 'unknown')}: {str(e)}")
                continue
        
        # Add embeddings to FAISS indices
        self._add_embeddings_to_indices(text_embeddings, image_embeddings, mixed_embeddings)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        self.embedding_stats['processing_time'] = processing_time
        self.embedding_stats['total_chunks_processed'] = processed_chunks
        
        # Save embeddings and indices
        self._save_embeddings_and_indices()
        
        logger.info(f"Processing complete. Created {len(text_embeddings)} text embeddings and {len(image_embeddings)} image embeddings")
        
        return {
            'text_embeddings': len(text_embeddings),
            'image_embeddings': len(image_embeddings),
            'mixed_embeddings': len(mixed_embeddings),
            'processing_time': processing_time,
            'total_chunks': processed_chunks
        }
    
    def _add_embeddings_to_indices(self, text_embeddings: List[np.ndarray], 
                                  image_embeddings: List[np.ndarray],
                                  mixed_embeddings: List[np.ndarray]):
        """Add embeddings to FAISS indices"""
        try:
            # Add text embeddings
            if text_embeddings:
                text_matrix = np.vstack(text_embeddings)
                self.text_index.add(text_matrix)
                logger.info(f"Added {len(text_embeddings)} text embeddings to index")
            
            # Add image embeddings
            if image_embeddings:
                image_matrix = np.vstack(image_embeddings)
                self.image_index.add(image_matrix)
                logger.info(f"Added {len(image_embeddings)} image embeddings to index")
            
            # Add mixed embeddings
            if mixed_embeddings:
                mixed_matrix = np.vstack(mixed_embeddings)
                self.mixed_index.add(mixed_matrix)
                logger.info(f"Added {len(mixed_embeddings)} mixed embeddings to index")
            
            self.embedding_stats['faiss_indices_created'] = 3
            
        except Exception as e:
            logger.error(f"Error adding embeddings to indices: {str(e)}")
    
    def search_similar_text(self, query: str, k: int = 5, threshold: float = 0.0) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search for similar text chunks
        
        Args:
            query: Search query
            k: Number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of (chunk, similarity_score) tuples
        """
        try:
            # Create query embedding
            query_embedding = self.create_text_embedding(query)
            query_embedding = query_embedding.reshape(1, -1)
            
            # Search in FAISS index
            scores, indices = self.text_index.search(query_embedding, k)
            
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx >= 0 and idx < len(self.text_chunks) and score >= threshold:
                    results.append((self.text_chunks[idx], float(score)))
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching similar text: {str(e)}")
            return []
    
    def search_similar_images(self, image_data: str, k: int = 5, threshold: float = 0.0) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search for similar image chunks
        
        Args:
            image_data: Base64 encoded query image
            k: Number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of (chunk, similarity_score) tuples
        """
        try:
            # Create query embedding
            query_embedding = self.create_image_embedding(image_data)
            
            # Debug: log norm of query embedding
            norm = np.linalg.norm(query_embedding)
            logger.info(f"Query image embedding norm: {norm}")
            
            query_embedding = query_embedding.reshape(1, -1)
            
            # Search in FAISS index
            scores, indices = self.image_index.search(query_embedding, k)
            
            logger.info(f"Image search scores: {scores}")
            logger.info(f"Image search indices: {indices}")
            
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx >= 0 and idx < len(self.image_chunks) and score >= threshold:
                    chunk = self.image_chunks[idx]
                    logger.info(f"Image search result {i}: score={score}, source={chunk.get('source', 'unknown')}")
                    results.append((chunk, float(score)))
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching similar images: {str(e)}")
            return []
    
    def search_multimodal(self, query: str, image_data: str = None, k: int = 5, threshold: float = 0.0) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search across both text and image chunks
        
        Args:
            query: Text query
            image_data: Optional image query
            k: Number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of (chunk, similarity_score) tuples
        """
        try:
            # Create multimodal query embedding
            query_embedding = self.create_multimodal_embedding(query, image_data)
            query_embedding = query_embedding.reshape(1, -1)
            
            # Search in mixed index
            scores, indices = self.mixed_index.search(query_embedding, k)
            
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx >= 0 and idx < len(self.mixed_chunks) and score >= threshold:
                    results.append((self.mixed_chunks[idx], float(score)))
            
            return results
            
        except Exception as e:
            logger.error(f"Error in multimodal search: {str(e)}")
            return []
    
    def _save_embeddings_and_indices(self):
        """Save embeddings and FAISS indices to disk"""
        try:
            # Save FAISS indices
            faiss.write_index(self.text_index, str(self.output_dir / "indices" / "text_index.faiss"))
            faiss.write_index(self.image_index, str(self.output_dir / "indices" / "image_index.faiss"))
            faiss.write_index(self.mixed_index, str(self.output_dir / "indices" / "mixed_index.faiss"))
            
            # Save chunks
            with open(self.output_dir / "embeddings" / "text_chunks.pkl", 'wb') as f:
                pickle.dump(self.text_chunks, f)
            
            with open(self.output_dir / "embeddings" / "image_chunks.pkl", 'wb') as f:
                pickle.dump(self.image_chunks, f)
            
            with open(self.output_dir / "embeddings" / "mixed_chunks.pkl", 'wb') as f:
                pickle.dump(self.mixed_chunks, f)
            
            # Save statistics
            with open(self.output_dir / "embedding_stats.json", 'w') as f:
                json.dump(self.embedding_stats, f, indent=2)
            
            logger.info(f"Embeddings and indices saved to {self.output_dir}")
            
        except Exception as e:
            logger.error(f"Error saving embeddings and indices: {str(e)}")
    
    def load_embeddings_and_indices(self):
        """Load previously saved embeddings and indices"""
        try:
            # Load FAISS indices
            self.text_index = faiss.read_index(str(self.output_dir / "indices" / "text_index.faiss"))
            self.image_index = faiss.read_index(str(self.output_dir / "indices" / "image_index.faiss"))
            self.mixed_index = faiss.read_index(str(self.output_dir / "indices" / "mixed_index.faiss"))
            
            # Load chunks
            with open(self.output_dir / "embeddings" / "text_chunks.pkl", 'rb') as f:
                self.text_chunks = pickle.load(f)
            
            with open(self.output_dir / "embeddings" / "image_chunks.pkl", 'rb') as f:
                self.image_chunks = pickle.load(f)
            
            with open(self.output_dir / "embeddings" / "mixed_chunks.pkl", 'rb') as f:
                self.mixed_chunks = pickle.load(f)
            
            # Load statistics
            with open(self.output_dir / "embedding_stats.json", 'r') as f:
                self.embedding_stats = json.load(f)
            
            logger.info("Embeddings and indices loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading embeddings and indices: {str(e)}")
    
    def visualize_embeddings(self, sample_size: int = 100):
        """
        Create visualizations of the embedding space
        
        Args:
            sample_size: Number of embeddings to visualize
        """
        try:
            from sklearn.manifold import TSNE
            from sklearn.decomposition import PCA
            
            # Sample embeddings for visualization
            if len(self.text_chunks) > 0:
                self._visualize_text_embeddings(sample_size)
            
            if len(self.image_chunks) > 0:
                self._visualize_image_embeddings(sample_size)
            
            # Create combined visualization
            self._visualize_mixed_embeddings(sample_size)
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}")
    
    def _visualize_text_embeddings(self, sample_size: int):
        """Visualize text embeddings"""
        try:
            from sklearn.manifold import TSNE

            # Sample text embeddings
            n_samples = min(sample_size, len(self.text_chunks))
            indices = np.random.choice(len(self.text_chunks), n_samples, replace=False)
            
            # Get embeddings for sampled chunks
            sample_embeddings = []
            sample_labels = []
            
            for idx in indices:
                chunk = self.text_chunks[idx]
                embedding = self.create_text_embedding(chunk['content'])
                sample_embeddings.append(embedding)
                sample_labels.append(chunk['source_file'])
            
            sample_embeddings = np.vstack(sample_embeddings)
            
            # Create t-SNE visualization
            tsne = TSNE(n_components=2, random_state=42)
            embeddings_2d = tsne.fit_transform(sample_embeddings)
            
            # Plot
            plt.figure(figsize=(12, 8))
            unique_files = list(set(sample_labels))
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_files)))
            
            for i, file in enumerate(unique_files):
                mask = [label == file for label in sample_labels]
                plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                           c=[colors[i]], label=file, alpha=0.7)
            
            plt.title('Text Embeddings Visualization (t-SNE)')
            plt.xlabel('t-SNE Component 1')
            plt.ylabel('t-SNE Component 2')
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.output_dir / "visualizations" / "text_embeddings_tsne.png", dpi=300)
            plt.close()
            
            logger.info("Text embeddings visualization saved")
            
        except Exception as e:
            logger.error(f"Error visualizing text embeddings: {str(e)}")
    
    def _visualize_image_embeddings(self, sample_size: int):
        """Visualize image embeddings"""
        try:
            from sklearn.manifold import TSNE
            # Similar to text visualization but for images
            n_samples = min(sample_size, len(self.image_chunks))
            if n_samples == 0:
                return
                
            indices = np.random.choice(len(self.image_chunks), n_samples, replace=False)
            
            sample_embeddings = []
            sample_labels = []
            
            for idx in indices:
                chunk = self.image_chunks[idx]
                embedding = self.create_image_embedding(chunk['image_data'])
                sample_embeddings.append(embedding)
                sample_labels.append(chunk['source_file'])
            
            sample_embeddings = np.vstack(sample_embeddings)
            
            # Create t-SNE visualization
            tsne = TSNE(n_components=2, random_state=42)
            embeddings_2d = tsne.fit_transform(sample_embeddings)
            
            # Plot
            plt.figure(figsize=(12, 8))
            unique_files = list(set(sample_labels))
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_files)))
            
            for i, file in enumerate(unique_files):
                mask = [label == file for label in sample_labels]
                plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                           c=[colors[i]], label=file, alpha=0.7)
            
            plt.title('Image Embeddings Visualization (t-SNE)')
            plt.xlabel('t-SNE Component 1')
            plt.ylabel('t-SNE Component 2')
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.output_dir / "visualizations" / "image_embeddings_tsne.png", dpi=300)
            plt.close()
            
            logger.info("Image embeddings visualization saved")
            
        except Exception as e:
            logger.error(f"Error visualizing image embeddings: {str(e)}")
    
    def _visualize_mixed_embeddings(self, sample_size: int):
        """Visualize mixed embeddings showing text vs image chunks"""
        try:
            from sklearn.manifold import TSNE
            n_samples = min(sample_size, len(self.mixed_chunks))
            if n_samples == 0:
                return
                    
            indices = np.random.choice(len(self.mixed_chunks), n_samples, replace=False)
            
            sample_embeddings = []
            sample_types = []
            sample_files = []
            
            for idx in indices:
                chunk = self.mixed_chunks[idx]
                if chunk['type'] == 'image':
                    embedding = self.create_multimodal_embedding(chunk['content'], chunk['image_data'])
                else:
                    embedding = self.create_multimodal_embedding(chunk['content'])
                
                sample_embeddings.append(embedding)
                sample_types.append(chunk['type'])
                sample_files.append(chunk['source_file'])
            
            sample_embeddings = np.vstack(sample_embeddings)
            
            # Create t-SNE visualization
            tsne = TSNE(n_components=2, random_state=42)
            embeddings_2d = tsne.fit_transform(sample_embeddings)
            
            # Plot with different markers for different types
            plt.figure(figsize=(12, 8))
            
            for chunk_type in ['text', 'image', 'table']:
                mask = [t == chunk_type for t in sample_types]
                if any(mask):
                    marker = 'o' if chunk_type == 'text' else ('s' if chunk_type == 'image' else '^')
                    plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                               label=chunk_type, alpha=0.7, marker=marker)
            
            plt.title('Mixed Embeddings Visualization (t-SNE)')
            plt.xlabel('t-SNE Component 1')
            plt.ylabel('t-SNE Component 2')
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.output_dir / "visualizations" / "mixed_embeddings_tsne.png", dpi=300)
            plt.close()
            
            logger.info("Mixed embeddings visualization saved")
            
        except Exception as e:
            logger.error(f"Error visualizing mixed embeddings: {str(e)}")
    
    def get_embedding_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the embeddings"""
        stats = self.embedding_stats.copy()
        
        # Add current counts
        stats['current_text_chunks'] = len(self.text_chunks)
        stats['current_image_chunks'] = len(self.image_chunks)
        stats['current_mixed_chunks'] = len(self.mixed_chunks)
        
        # Add index information
        stats['text_index_size'] = self.text_index.ntotal if hasattr(self.text_index, 'ntotal') else 0
        stats['image_index_size'] = self.image_index.ntotal if hasattr(self.image_index, 'ntotal') else 0
        stats['mixed_index_size'] = self.mixed_index.ntotal if hasattr(self.mixed_index, 'ntotal') else 0
        
        return stats
    
    def print_embedding_summary(self):
        """Print comprehensive summary of embedding process"""
        stats = self.get_embedding_statistics()
        
        print("\n" + "="*60)
        print("EMBEDDING SYSTEM SUMMARY")
        print("="*60)
        print(f"Text Model: {stats['model_info']['text_model']}")
        print(f"Image Model: {stats['model_info']['image_model']}")
        print(f"Device: {stats['model_info']['device']}")
        print(f"Processing Time: {stats['processing_time']:.2f} seconds")
        print("\nEmbedding Statistics:")
        print(f"  Text Embeddings: {stats['text_embeddings_created']}")
        print(f"  Image Embeddings: {stats['image_embeddings_created']}")
        print(f"  Total Chunks Processed: {stats['total_chunks_processed']}")
        print("\nFAISS Index Sizes:")
        print(f"  Text Index: {stats['text_index_size']} vectors")
        print(f"  Image Index: {stats['image_index_size']} vectors")
        print(f"  Mixed Index: {stats['mixed_index_size']} vectors")
        print(f"\nOutput Directory: {self.output_dir}")
        print("="*60)

def main():
    """Main function to demonstrate the embedding system"""
    
    # Initialize embedding manager
    embedding_manager = MultimodalEmbeddingManager()
    
    # Load extracted chunks (from previous data extraction step)
    try:
        with open("extracted_data/chunks/all_chunks.pkl", 'rb') as f:
            chunks = pickle.load(f)
        
        logger.info(f"Loaded {len(chunks)} chunks from extraction")
        
        # Process chunks to create embeddings
        results = embedding_manager.process_chunks(chunks)
        
        # Print summary
        embedding_manager.print_embedding_summary()
        
        # Create visualizations
        logger.info("Creating embedding visualizations...")
        embedding_manager.visualize_embeddings()
        
        # Test search functionality
        logger.info("Testing search functionality...")
        
        # Test text search
        text_results = embedding_manager.search_similar_text("revenue growth", k=3)
        print(f"\nText Search Results for 'revenue growth':")
        for i, (chunk, score) in enumerate(text_results):
            print(f"{i+1}. Score: {score:.3f} | Source: {chunk['source']}")
            print(f"Content: {chunk['content'][:100]}...\n")
        
        # Test image search (using first image chunk if available)
        if embedding_manager.image_chunks:
            first_image_chunk = embedding_manager.image_chunks[0]
            image_data = first_image_chunk.get('image_data', None)
            if image_data:
                image_results = embedding_manager.search_similar_images(image_data, k=3)
                print(f"\nImage Search Results (based on first image chunk):")
                for i, (chunk, score) in enumerate(image_results):
                    print(f"{i+1}. Score: {score:.3f} | Source: {chunk['source']}")
                    print(f"Content: {chunk.get('content', '')[:100]}...\n")
        
        # Test multimodal search
        multimodal_results = embedding_manager.search_multimodal("financial report", image_data=None, k=3)
        print(f"\nMultimodal Search Results for 'financial report':")
        for i, (chunk, score) in enumerate(multimodal_results):
            print(f"{i+1}. Score: {score:.3f} | Source: {chunk['source']}")
            print(f"Content: {chunk['content'][:100]}...\n")
        
    except FileNotFoundError:
        logger.error("Extracted chunks file not found. Please run the extraction step first.")
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")

if __name__ == "__main__":
    main()
