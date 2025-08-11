"""Embedding service for concept representations."""

import asyncio
from typing import List, Optional, Union
import numpy as np
import hashlib
import logging

from app.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating and managing concept embeddings."""
    
    def __init__(self):
        """Initialize the embedding service."""
        self.vector_dimension = settings.vector_dimension
        self.batch_size = settings.embedding_batch_size
        self._initialized = False
        
    async def initialize(self):
        """Initialize the embedding service asynchronously."""
        if self._initialized:
            return
            
        try:
            logger.info("Initializing simplified embedding service")
            self._initialized = True
            logger.info("Embedding service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize embedding service: {e}")
            raise
    
    def _generate_simple_embedding(self, text: str) -> List[float]:
        """Generate a simple embedding using hash-based approach."""
        try:
            # Create a hash of the text
            text_hash = hashlib.md5(text.encode()).hexdigest()
            
            # Convert hash to numerical values
            hash_values = [int(text_hash[i:i+2], 16) for i in range(0, len(text_hash), 2)]
            
            # Normalize to [-1, 1] range and extend to required dimension
            embedding = []
            for i in range(self.vector_dimension):
                if i < len(hash_values):
                    # Normalize hash value to [-1, 1]
                    normalized = (hash_values[i] / 255.0) * 2 - 1
                    embedding.append(normalized)
                else:
                    # Fill remaining dimensions with small random values
                    embedding.append(np.random.uniform(-0.1, 0.1))
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating simple embedding: {e}")
            # Return zero vector as fallback
            return [0.0] * self.vector_dimension
    
    def _prepare_text(self, text: Union[str, List[str]]) -> str:
        """Prepare text for embedding generation."""
        if isinstance(text, list):
            # Join multiple text elements with proper formatting
            return " | ".join([str(item) for item in text])
        return str(text)
    
    async def generate_embedding(self, text: Union[str, List[str]]) -> List[float]:
        """Generate embedding for a single text input."""
        await self.initialize()
        
        try:
            prepared_text = self._prepare_text(text)
            logger.debug(f"Generating embedding for text: {prepared_text[:100]}...")
            
            # Generate simple embedding
            embedding = self._generate_simple_embedding(prepared_text)
            
            logger.debug(f"Generated embedding with dimension: {len(embedding)}")
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    async def generate_batch_embeddings(self, texts: List[Union[str, List[str]]]) -> List[List[float]]:
        """Generate embeddings for a batch of texts."""
        await self.initialize()
        
        try:
            logger.info(f"Generating batch embeddings for {len(texts)} texts")
            embeddings = []
            
            # Process in batches to avoid memory issues
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                batch_embeddings = []
                
                # Generate embeddings for the current batch
                for text in batch:
                    embedding = await self.generate_embedding(text)
                    batch_embeddings.append(embedding)
                
                embeddings.extend(batch_embeddings)
                logger.debug(f"Processed batch {i//self.batch_size + 1}/{(len(texts) + self.batch_size - 1)//self.batch_size}")
            
            logger.info(f"Successfully generated {len(embeddings)} embeddings")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            raise
    
    async def generate_concept_embedding(self, concept_data: dict) -> List[float]:
        """Generate embedding for a concept based on its key attributes."""
        try:
            # Combine key concept attributes for embedding
            text_elements = [
                concept_data.get("name", ""),
                concept_data.get("description", ""),
                " ".join(concept_data.get("learning_objectives", [])),
                " ".join(concept_data.get("tags", [])),
                f"Grade {concept_data.get('grade_level', '')}",
                f"Difficulty {concept_data.get('difficulty', '')}",
                concept_data.get("concept_type", "")
            ]
            
            # Filter out empty elements and join
            text_elements = [elem for elem in text_elements if elem]
            combined_text = " | ".join(text_elements)
            
            return await self.generate_embedding(combined_text)
            
        except Exception as e:
            logger.error(f"Error generating concept embedding: {e}")
            raise
    
    def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Compute cosine similarity between two embeddings."""
        try:
            # Convert to numpy arrays
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Compute cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error computing similarity: {e}")
            return 0.0
    
    def find_most_similar(self, query_embedding: List[float], candidate_embeddings: List[List[float]], top_k: int = 5) -> List[tuple]:
        """Find the most similar embeddings to a query embedding."""
        try:
            similarities = []
            
            for i, candidate in enumerate(candidate_embeddings):
                similarity = self.compute_similarity(query_embedding, candidate)
                similarities.append((i, similarity))
            
            # Sort by similarity (descending) and return top_k
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Error finding most similar embeddings: {e}")
            return []
    
    async def validate_embedding(self, embedding: List[float]) -> bool:
        """Validate that an embedding meets the expected format and dimension."""
        try:
            if not isinstance(embedding, list):
                return False
            
            if len(embedding) != self.vector_dimension:
                return False
            
            # Check if all elements are numeric
            if not all(isinstance(x, (int, float)) for x in embedding):
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating embedding: {e}")
            return False
    
    def get_model_info(self) -> dict:
        """Get information about the embedding service."""
        return {
            "model_name": "simple_hash_based",
            "vector_dimension": self.vector_dimension,
            "batch_size": self.batch_size,
            "initialized": self._initialized,
            "note": "Using simplified hash-based embeddings (mxbai-embed-large not available)"
        }
    
    async def cleanup(self):
        """Clean up resources."""
        try:
            self._initialized = False
            logger.info("Embedding service cleaned up successfully")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


# Global embedding service instance
embedding_service = EmbeddingService()
