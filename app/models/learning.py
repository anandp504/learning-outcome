"""Learning journey models for tracking student learning paths."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class LearningJourneyBase(BaseModel):
    """Base learning journey model with common fields."""
    
    student_id: str = Field(..., description="Student identifier")
    current_concept: str = Field(..., description="Currently studying concept")
    
    learning_path: List[str] = Field(..., description="Sequence of concepts in learning path")
    completed_concepts: List[str] = Field(default=[], description="List of completed concepts")
    
    next_recommendations: List[str] = Field(default=[], description="Next recommended concepts")
    alternative_paths: List[List[str]] = Field(default=[], description="Alternative learning paths")
    
    learning_goals: List[str] = Field(default=[], description="Specific learning goals")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion date")
    
    metadata: Dict[str, Any] = Field(default={}, description="Additional journey information")


class LearningJourneyCreate(LearningJourneyBase):
    """Model for creating a new learning journey."""
    pass


class LearningJourneyUpdate(BaseModel):
    """Model for updating an existing learning journey."""
    
    current_concept: Optional[str] = None
    learning_path: Optional[List[str]] = None
    completed_concepts: Optional[List[str]] = None
    
    next_recommendations: Optional[List[str]] = None
    alternative_paths: Optional[List[List[str]]] = None
    
    learning_goals: Optional[List[str]] = None
    estimated_completion: Optional[datetime] = None
    
    metadata: Optional[Dict[str, Any]] = None


class LearningJourney(LearningJourneyBase):
    """Complete learning journey model with all fields."""
    
    journey_id: str = Field(..., description="Unique identifier for the learning journey")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        from_attributes = True


class LearningProgress(BaseModel):
    """Model for tracking learning progress on a concept."""
    
    student_id: str = Field(..., description="Student identifier")
    concept_id: str = Field(..., description="Concept identifier")
    
    progress_score: float = Field(..., ge=0, le=1, description="Progress score (0-1)")
    time_spent: float = Field(..., ge=0, description="Time spent in hours")
    attempts: int = Field(..., ge=0, description="Number of attempts")
    
    current_stage: str = Field(..., description="Current learning stage")
    next_milestone: str = Field(..., description="Next learning milestone")
    
    last_updated: datetime = Field(..., description="Last progress update")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion")
    
    class Config:
        from_attributes = True


class LearningRecommendation(BaseModel):
    """Model for learning recommendations."""
    
    concept_id: str = Field(..., description="Recommended concept ID")
    name: str = Field(..., description="Concept name")
    description: str = Field(..., description="Concept description")
    
    reasoning: str = Field(..., description="Why this concept was recommended")
    confidence_score: float = Field(..., ge=0, le=1, description="Confidence in recommendation")
    
    difficulty: int = Field(..., ge=1, le=5, description="Concept difficulty level")
    estimated_time: float = Field(..., ge=0, description="Estimated time to master")
    
    prerequisites_met: bool = Field(..., description="Whether prerequisites are satisfied")
    learning_style_match: float = Field(..., ge=0, le=1, description="Match with learning style")
    
    alternative_concepts: List[str] = Field(default=[], description="Alternative concept options")


class LearningPath(BaseModel):
    """Model for a complete learning path."""
    
    path_id: str = Field(..., description="Unique path identifier")
    name: str = Field(..., description="Path name")
    description: str = Field(..., description="Path description")
    
    concepts: List[str] = Field(..., description="Ordered list of concept IDs")
    total_estimated_time: float = Field(..., ge=0, description="Total estimated time")
    difficulty_progression: List[int] = Field(..., description="Difficulty progression")
    
    prerequisites: List[str] = Field(default=[], description="Path prerequisites")
    learning_outcomes: List[str] = Field(..., description="Expected learning outcomes")
    
    tags: List[str] = Field(default=[], description="Path tags")
    metadata: Dict[str, Any] = Field(default={}, description="Additional path information")
    
    class Config:
        from_attributes = True


class LearningSession(BaseModel):
    """Model for tracking individual learning sessions."""
    
    session_id: str = Field(..., description="Unique session identifier")
    student_id: str = Field(..., description="Student identifier")
    concept_id: str = Field(..., description="Concept being studied")
    
    start_time: datetime = Field(..., description="Session start time")
    end_time: Optional[datetime] = Field(None, description="Session end time")
    
    duration: float = Field(..., ge=0, description="Session duration in minutes")
    activities_completed: List[str] = Field(default=[], description="Completed activities")
    
    performance_metrics: Dict[str, Any] = Field(default={}, description="Performance data")
    notes: str = Field(default="", description="Session notes")
    
    class Config:
        from_attributes = True
