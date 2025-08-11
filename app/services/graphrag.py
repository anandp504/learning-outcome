"""GraphRAG service combining graph relationships with vector embeddings."""

import asyncio
from typing import List, Dict, Any, Optional, Tuple
import networkx as nx
import chromadb
from chromadb.config import Settings
import numpy as np
import json
import logging
from datetime import datetime

from app.config import settings
from app.models.concept import Concept, ConceptSearchResult, ConceptRecommendation
from app.services.embedding import embedding_service

logger = logging.getLogger(__name__)


class GraphRAGService:
    """Service for GraphRAG operations combining graph traversal with vector search."""
    
    def __init__(self):
        """Initialize the GraphRAG service."""
        self.graph = nx.DiGraph()
        self.vector_db = None
        self.concepts_cache = {}
        self.embeddings_cache = {}
        self._initialized = False
        
    async def initialize(self):
        """Initialize the GraphRAG service."""
        if self._initialized:
            return
            
        try:
            # Initialize vector database
            await self._init_vector_db()
            
            # Initialize embedding service
            await embedding_service.initialize()
            
            # Load knowledge graph
            await self._load_knowledge_graph()
            
            self._initialized = True
            logger.info("GraphRAG service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize GraphRAG service: {e}")
            raise
    
    async def _init_vector_db(self):
        """Initialize the ChromaDB vector database."""
        try:
            client_settings = Settings(
                persist_directory=settings.chroma_persist_directory,
                anonymized_telemetry=False
            )
            
            self.vector_db = chromadb.PersistentClient(settings=client_settings)
            
            # Get or create collection
            try:
                self.concept_collection = self.vector_db.get_collection("concepts")
                logger.info("Using existing concepts collection")
            except:
                self.concept_collection = self.vector_db.create_collection(
                    name="concepts",
                    metadata={"description": "Math concepts with embeddings"}
                )
                logger.info("Created new concepts collection")
                
        except Exception as e:
            logger.error(f"Error initializing vector database: {e}")
            raise
    
    async def _load_knowledge_graph(self):
        """Load the knowledge graph from data files."""
        try:
            # Load concepts
            concepts_file = "data/knowledge_graph.json"
            with open(concepts_file, 'r') as f:
                knowledge_graph_data = json.load(f)
            
            # Extract concepts from the knowledge graph structure
            if isinstance(knowledge_graph_data, dict) and "concepts" in knowledge_graph_data:
                concepts_data = knowledge_graph_data["concepts"]
                logger.info(f"Found {len(concepts_data)} concepts in knowledge graph")
            else:
                # Fallback: assume the data is directly a list of concepts
                concepts_data = knowledge_graph_data
                logger.info(f"Using direct concepts data: {len(concepts_data)} concepts")
            
            # Build graph and populate vector database
            await self._build_graph(concepts_data)
            await self._populate_vector_db(concepts_data)
            
            logger.info(f"Loaded {len(concepts_data)} concepts into knowledge graph")
            
        except FileNotFoundError:
            logger.warning("Knowledge graph file not found, starting with empty graph")
            self.graph = nx.DiGraph()
            # Create a minimal concept for testing
            await self._create_minimal_concepts()
        except Exception as e:
            logger.error(f"Error loading knowledge graph: {e}")
            raise
    
    async def _create_minimal_concepts(self):
        """Create minimal concepts for testing when no knowledge graph exists."""
        try:
            minimal_concepts = [
                {
                    "concept_id": "test_concept_1",
                    "name": "Test Concept 1",
                    "grade_level": 6,
                    "description": "A test concept for development",
                    "difficulty": 1,
                    "estimated_hours": 1.0,
                    "concept_type": "test",
                    "learning_objectives": ["Test objective"],
                    "practice_problems": ["Test problem"],
                    "prerequisites": [],
                    "next_concepts": [],
                    "tags": ["test"],
                    "metadata": {}
                }
            ]
            
            await self._build_graph(minimal_concepts)
            await self._populate_vector_db(minimal_concepts)
            logger.info("Created minimal test concepts")
            
        except Exception as e:
            logger.error(f"Error creating minimal concepts: {e}")
    
    async def _build_graph(self, concepts_data: List[Dict]):
        """Build the NetworkX graph from concepts data."""
        try:
            # Clear existing graph
            self.graph.clear()
            
            # Add concept nodes
            for concept in concepts_data:
                concept_id = concept["concept_id"]
                self.graph.add_node(concept_id, **concept)
                self.concepts_cache[concept_id] = concept
            
            # Add edges based on relationships
            for concept in concepts_data:
                concept_id = concept["concept_id"]
                
                # Add prerequisite edges
                for prereq in concept.get("prerequisites", []):
                    if prereq in self.concepts_cache:
                        self.graph.add_edge(prereq, concept_id, 
                                          relationship_type="prerequisite", 
                                          strength=0.9)
                
                # Add next concept edges
                for next_concept in concept.get("next_concepts", []):
                    if next_concept in self.concepts_cache:
                        self.graph.add_edge(concept_id, next_concept, 
                                          relationship_type="next", 
                                          strength=0.8)
            
            logger.info(f"Built graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
            
        except Exception as e:
            logger.error(f"Error building graph: {e}")
            raise
    
    async def _populate_vector_db(self, concepts_data: List[Dict]):
        """Populate the vector database with concept embeddings."""
        try:
            # Generate embeddings for all concepts
            concept_texts = []
            concept_ids = []
            concept_metadatas = []
            
            for concept in concepts_data:
                # Create a text representation of the concept for embedding
                concept_text = f"{concept['name']} {concept['description']} {' '.join(concept.get('tags', []))}"
                
                # Generate embedding
                embedding = await embedding_service.generate_embedding(concept_text)
                
                # Store in cache
                self.embeddings_cache[concept["concept_id"]] = embedding
                
                # Prepare for vector database
                concept_texts.append(concept["name"])
                concept_ids.append(concept["concept_id"])
                concept_metadatas.append({
                    "grade_level": concept["grade_level"],
                    "difficulty": concept["difficulty"],
                    "concept_type": concept["concept_type"],
                    "tags": ",".join(concept.get("tags", []))
                })
            
            # Add to vector database
            if concept_texts:
                self.concept_collection.add(
                    embeddings=[embedding for embedding in self.embeddings_cache.values()],
                    documents=concept_texts,
                    ids=concept_ids,
                    metadatas=concept_metadatas
                )
                
            logger.info(f"Populated vector database with {len(concepts_data)} concepts")
            
        except Exception as e:
            logger.error(f"Error populating vector database: {e}")
            raise
    
    async def search_concepts(self, query: str, top_k: int = 5, 
                            use_graph: bool = True) -> List[ConceptSearchResult]:
        """Search for concepts using vector similarity and optionally graph traversal."""
        await self.initialize()
        
        try:
            # Generate query embedding
            query_embedding = await embedding_service.generate_embedding(query)
            
            # Vector search
            vector_results = await self._vector_search(query_embedding, top_k)
            
            if not use_graph:
                return vector_results
            
            # Graph-enhanced search
            graph_results = await self._graph_enhanced_search(query_embedding, vector_results, top_k)
            
            return graph_results
            
        except Exception as e:
            logger.error(f"Error searching concepts: {e}")
            return []
    
    async def _vector_search(self, query_embedding: List[float], top_k: int) -> List[ConceptSearchResult]:
        """Perform vector similarity search."""
        try:
            # Search in vector database
            results = self.concept_collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            search_results = []
            for i, (doc, metadata, distance) in enumerate(zip(
                results["documents"][0], 
                results["metadatas"][0], 
                results["distances"][0]
            )):
                concept_id = results["ids"][0][i]
                concept = self.concepts_cache.get(concept_id)
                
                if concept:
                    # Convert distance to similarity score (ChromaDB returns distances)
                    similarity_score = 1.0 / (1.0 + distance)
                    
                    search_result = ConceptSearchResult(
                        concept=Concept(**concept),
                        similarity_score=similarity_score,
                        graph_score=0.0,  # Will be computed later if using graph
                        combined_score=similarity_score,
                        reasoning=f"Vector similarity search result with score {similarity_score:.3f}"
                    )
                    search_results.append(search_result)
            
            return search_results
            
        except Exception as e:
            logger.error(f"Error in vector search: {e}")
            return []
    
    async def _graph_enhanced_search(self, query_embedding: List[float], 
                                   vector_results: List[ConceptSearchResult], 
                                   top_k: int) -> List[ConceptSearchResult]:
        """Enhance vector search results using graph relationships."""
        try:
            enhanced_results = []
            
            for result in vector_results:
                concept_id = result.concept.concept_id
                
                # Compute graph score based on relationships
                graph_score = self._compute_graph_score(concept_id, query_embedding)
                
                # Combine scores (weighted average)
                combined_score = 0.7 * result.similarity_score + 0.3 * graph_score
                
                # Update result
                result.graph_score = graph_score
                result.combined_score = combined_score
                result.reasoning = self._generate_reasoning(
                    result.similarity_score, graph_score, concept_id
                )
                
                enhanced_results.append(result)
            
            # Sort by combined score
            enhanced_results.sort(key=lambda x: x.combined_score, reverse=True)
            
            return enhanced_results[:top_k]
            
        except Exception as e:
            logger.error(f"Error in graph-enhanced search: {e}")
            return vector_results
    
    def _compute_graph_score(self, concept_id: str, query_embedding: List[float]) -> float:
        """Compute graph-based score for a concept."""
        try:
            if concept_id not in self.graph:
                return 0.0
            
            # Get concept embedding
            concept_embedding = self.embeddings_cache.get(concept_id)
            if not concept_embedding:
                return 0.0
            
            # Base similarity to query
            base_similarity = embedding_service.compute_similarity(
                query_embedding, concept_embedding
            )
            
            # Graph centrality score
            centrality = nx.pagerank(self.graph, alpha=0.85)
            centrality_score = centrality.get(concept_id, 0.0)
            
            # Relationship strength score
            relationship_score = self._compute_relationship_strength(concept_id)
            
            # Combine scores
            graph_score = 0.4 * base_similarity + 0.3 * centrality_score + 0.3 * relationship_score
            
            return graph_score
            
        except Exception as e:
            logger.error(f"Error computing graph score: {e}")
            return 0.0
    
    def _compute_relationship_strength(self, concept_id: str) -> float:
        """Compute relationship strength score for a concept."""
        try:
            if concept_id not in self.graph:
                return 0.0
            
            # Get incoming and outgoing edges
            in_edges = list(self.graph.in_edges(concept_id, data=True))
            out_edges = list(self.graph.out_edges(concept_id, data=True))
            
            # Compute average edge strength
            total_strength = 0.0
            edge_count = 0
            
            for _, _, data in in_edges + out_edges:
                strength = data.get("strength", 0.5)
                total_strength += strength
                edge_count += 1
            
            if edge_count == 0:
                return 0.5  # Default score
            
            return total_strength / edge_count
            
        except Exception as e:
            logger.error(f"Error computing relationship strength: {e}")
            return 0.5
    
    def _generate_reasoning(self, similarity_score: float, graph_score: float, concept_id: str) -> str:
        """Generate reasoning for search result ranking."""
        try:
            reasoning_parts = []
            
            # Similarity reasoning
            if similarity_score > 0.8:
                reasoning_parts.append("High semantic similarity to query")
            elif similarity_score > 0.6:
                reasoning_parts.append("Good semantic similarity to query")
            else:
                reasoning_parts.append("Moderate semantic similarity to query")
            
            # Graph reasoning
            if graph_score > 0.7:
                reasoning_parts.append("Strong connections in knowledge graph")
            elif graph_score > 0.5:
                reasoning_parts.append("Good connections in knowledge graph")
            
            # Concept-specific reasoning
            concept = self.concepts_cache.get(concept_id)
            if concept:
                if concept.get("difficulty", 3) <= 2:
                    reasoning_parts.append("Appropriate difficulty level")
                if concept.get("grade_level", 6) == 6:
                    reasoning_parts.append("Grade-appropriate content")
            
            return "; ".join(reasoning_parts)
            
        except Exception as e:
            logger.error(f"Error generating reasoning: {e}")
            return "Search result based on semantic similarity and graph relationships"
    
    async def get_concept_recommendations(self, student_id: str, current_concept: str, 
                                       performance_score: float, top_k: int = 5) -> List[ConceptRecommendation]:
        """Get personalized concept recommendations for a student."""
        await self.initialize()
        
        try:
            # Get current concept info
            if current_concept not in self.concepts_cache:
                logger.warning(f"Current concept {current_concept} not found in cache")
                return []
            
            current_concept_data = self.concepts_cache[current_concept]
            
            # Get next concepts from graph
            next_concepts = current_concept_data.get("next_concepts", [])
            
            # Filter and rank recommendations
            recommendations = []
            for next_concept_id in next_concepts:
                if next_concept_id in self.concepts_cache:
                    recommendation = await self._create_recommendation(
                        next_concept_id, student_id, performance_score
                    )
                    if recommendation:
                        recommendations.append(recommendation)
            
            # Sort by confidence score and return top_k
            recommendations.sort(key=lambda x: x.confidence_score, reverse=True)
            return recommendations[:top_k]
            
        except Exception as e:
            logger.error(f"Error getting concept recommendations: {e}")
            return []
    
    async def _create_recommendation(self, concept_id: str, student_id: str, 
                                   performance_score: float) -> Optional[ConceptRecommendation]:
        """Create a recommendation for a specific concept."""
        try:
            concept_data = self.concepts_cache.get(concept_id)
            if not concept_data:
                return None
            
            # Check prerequisites
            prerequisites_met = self._check_prerequisites(concept_id, student_id)
            
            # Compute confidence score
            confidence_score = self._compute_confidence_score(
                concept_data, performance_score, prerequisites_met
            )
            
            # Generate reasoning
            reasoning = self._generate_recommendation_reasoning(
                concept_data, performance_score, prerequisites_met
            )
            
            # Estimate time to mastery
            time_to_mastery = concept_data.get("estimated_hours", 3.0)
            if performance_score < 0.5:
                time_to_mastery *= 1.5  # More time if struggling
            
            recommendation = ConceptRecommendation(
                concept_id=concept_id,
                name=concept_data["name"],
                description=concept_data["description"],
                difficulty=concept_data["difficulty"],
                estimated_hours=concept_data["estimated_hours"],
                reasoning=reasoning,
                confidence_score=confidence_score,
                prerequisites_met=prerequisites_met,
                time_to_mastery=time_to_mastery
            )
            
            return recommendation
            
        except Exception as e:
            logger.error(f"Error creating recommendation: {e}")
            return None
    
    def _check_prerequisites(self, concept_id: str, student_id: str) -> bool:
        """Check if prerequisites are met for a concept."""
        try:
            if concept_id not in self.graph:
                return False
            
            # Get prerequisites
            prerequisites = list(self.graph.predecessors(concept_id))
            
            # For now, assume prerequisites are met (would check student progress in real implementation)
            # TODO: Implement actual prerequisite checking based on student progress
            return True
            
        except Exception as e:
            logger.error(f"Error checking prerequisites: {e}")
            return False
    
    def _compute_confidence_score(self, concept_data: Dict, performance_score: float, 
                                prerequisites_met: bool) -> float:
        """Compute confidence score for a recommendation."""
        try:
            base_score = 0.5
            
            # Adjust based on performance
            if performance_score > 0.8:
                base_score += 0.2
            elif performance_score > 0.6:
                base_score += 0.1
            elif performance_score < 0.4:
                base_score -= 0.1
            
            # Adjust based on prerequisites
            if prerequisites_met:
                base_score += 0.2
            else:
                base_score -= 0.3
            
            # Adjust based on difficulty progression
            if concept_data.get("difficulty", 3) <= 3:
                base_score += 0.1
            
            return max(0.0, min(1.0, base_score))
            
        except Exception as e:
            logger.error(f"Error computing confidence score: {e}")
            return 0.5
    
    def _generate_recommendation_reasoning(self, concept_data: Dict, performance_score: float, 
                                         prerequisites_met: bool) -> str:
        """Generate reasoning for a recommendation."""
        try:
            reasoning_parts = []
            
            if prerequisites_met:
                reasoning_parts.append("Prerequisites satisfied")
            else:
                reasoning_parts.append("Prerequisites not yet met")
            
            if performance_score > 0.8:
                reasoning_parts.append("High performance suggests readiness for next concept")
            elif performance_score > 0.6:
                reasoning_parts.append("Good performance indicates readiness to advance")
            else:
                reasoning_parts.append("Consider reviewing current concept before advancing")
            
            reasoning_parts.append(f"Natural progression from current concept")
            
            return "; ".join(reasoning_parts)
            
        except Exception as e:
            logger.error(f"Error generating recommendation reasoning: {e}")
            return "Recommended based on learning progression and current performance"
    
    def get_graph_info(self) -> Dict[str, Any]:
        """Get information about the knowledge graph."""
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "concepts_loaded": len(self.concepts_cache),
            "embeddings_loaded": len(self.embeddings_cache),
            "initialized": self._initialized
        }
    
    async def cleanup(self):
        """Clean up resources."""
        try:
            if self.vector_db:
                self.vector_db.persist()
                logger.info("Vector database persisted successfully")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


# Global GraphRAG service instance
graphrag_service = GraphRAGService()
