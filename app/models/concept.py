"""Concept data models for the knowledge graph."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class ConceptBase(BaseModel):
    """Base concept model with common fields."""
    
    name: str = Field(..., description="Name of the math concept")
    grade_level: int = Field(..., ge=1, le=12, description="Grade level for the concept")
    description: str = Field(..., description="Detailed description of the concept")
    difficulty: int = Field(..., ge=1, le=5, description="Difficulty level (1=easy, 5=hard)")
    estimated_hours: float = Field(..., gt=0, description="Estimated time to master in hours")
    concept_type: str = Field(..., description="Type of concept (e.g., number_theory, fractions)")
    
    learning_objectives: List[str] = Field(..., description="List of learning objectives")
    practice_problems: List[str] = Field(..., description="Sample practice problems")
    
    prerequisites: List[str] = Field(default=[], description="List of prerequisite concept IDs")
    next_concepts: List[str] = Field(default=[], description="List of next concept IDs")
    
    tags: List[str] = Field(default=[], description="Tags for categorization")
    metadata: Dict[str, Any] = Field(default={}, description="Additional metadata")


class ConceptCreate(ConceptBase):
    """Model for creating a new concept."""
    pass


class ConceptUpdate(BaseModel):
    """Model for updating an existing concept."""
    
    name: Optional[str] = None
    grade_level: Optional[int] = Field(None, ge=1, le=12)
    description: Optional[str] = None
    difficulty: Optional[int] = Field(None, ge=1, le=5)
    estimated_hours: Optional[float] = Field(None, gt=0)
    concept_type: Optional[str] = None
    
    learning_objectives: Optional[List[str]] = None
    practice_problems: Optional[List[str]] = None
    
    prerequisites: Optional[List[str]] = None
    next_concepts: Optional[List[str]] = None
    
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class Concept(ConceptBase):
    """Complete concept model with all fields."""
    
    concept_id: str = Field(..., description="Unique identifier for the concept")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding of the concept")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        from_attributes = True


class ConceptRelationship(BaseModel):
    """Model for concept relationships."""
    
    relationship_id: str = Field(..., description="Unique identifier for the relationship")
    from_concept: str = Field(..., description="Source concept ID")
    to_concept: str = Field(..., description="Target concept ID")
    relationship_type: str = Field(..., description="Type of relationship (prerequisite, next, related)")
    strength: float = Field(..., ge=0, le=1, description="Strength of the relationship (0-1)")
    description: str = Field(..., description="Description of the relationship")
    
    class Config:
        from_attributes = True


class ConceptSearchResult(BaseModel):
    """Model for concept search results."""
    
    concept: Concept
    similarity_score: float = Field(..., description="Similarity score from vector search")
    graph_score: float = Field(..., description="Score from graph traversal")
    combined_score: float = Field(..., description="Combined ranking score")
    reasoning: str = Field(..., description="Explanation for the recommendation")


class ConceptRecommendation(BaseModel):
    """Model for concept recommendations."""
    
    concept_id: str
    name: str
    description: str
    difficulty: int
    estimated_hours: float
    reasoning: str = Field(..., description="Why this concept was recommended")
    confidence_score: float = Field(..., ge=0, le=1, description="Confidence in the recommendation")
    prerequisites_met: bool = Field(..., description="Whether prerequisites are satisfied")
    time_to_mastery: float = Field(..., description="Estimated time considering current progress")
