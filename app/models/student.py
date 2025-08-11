"""Student data models for tracking learning progress."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class StudentBase(BaseModel):
    """Base student model with common fields."""
    
    name: str = Field(..., description="Student's full name")
    grade_level: int = Field(..., ge=1, le=12, description="Current grade level")
    age: int = Field(..., ge=5, le=18, description="Student's age")
    
    learning_style: str = Field(default="visual", description="Preferred learning style")
    math_aptitude: str = Field(default="average", description="General math aptitude level")
    
    interests: List[str] = Field(default=[], description="Student's interests and hobbies")
    goals: List[str] = Field(default=[], description="Learning goals and objectives")
    
    metadata: Dict[str, Any] = Field(default={}, description="Additional student information")


class StudentCreate(StudentBase):
    """Model for creating a new student."""
    pass


class StudentUpdate(BaseModel):
    """Model for updating an existing student."""
    
    name: Optional[str] = None
    grade_level: Optional[int] = Field(None, ge=1, le=12)
    age: Optional[int] = Field(None, ge=5, le=18)
    
    learning_style: Optional[str] = None
    math_aptitude: Optional[str] = None
    
    interests: Optional[List[str]] = None
    goals: Optional[List[str]] = None
    
    metadata: Optional[Dict[str, Any]] = None


class Student(StudentBase):
    """Complete student model with all fields."""
    
    student_id: str = Field(..., description="Unique identifier for the student")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        from_attributes = True


class StudentPerformance(BaseModel):
    """Model for tracking student performance on specific concepts."""
    
    student_id: str = Field(..., description="Student identifier")
    concept_id: str = Field(..., description="Concept identifier")
    
    performance_score: float = Field(..., ge=0, le=1, description="Performance score (0-1)")
    attempts: int = Field(..., ge=0, description="Number of attempts")
    time_spent: float = Field(..., ge=0, description="Time spent in hours")
    
    strengths: List[str] = Field(default=[], description="Areas of strength")
    weaknesses: List[str] = Field(default=[], description="Areas needing improvement")
    
    last_attempt: datetime = Field(..., description="Last attempt timestamp")
    mastery_level: str = Field(..., description="Current mastery level")
    
    feedback: str = Field(default="", description="Teacher or system feedback")
    
    class Config:
        from_attributes = True


class StudentProfile(BaseModel):
    """Comprehensive student profile with performance data."""
    
    student: Student
    current_concept: Optional[str] = Field(None, description="Currently studying concept")
    overall_progress: float = Field(..., ge=0, le=1, description="Overall learning progress")
    
    completed_concepts: List[str] = Field(default=[], description="List of mastered concepts")
    in_progress_concepts: List[str] = Field(default=[], description="Concepts currently being studied")
    
    total_study_time: float = Field(..., ge=0, description="Total study time in hours")
    average_performance: float = Field(..., ge=0, le=1, description="Average performance score")
    
    learning_pace: str = Field(..., description="Learning pace (slow, average, fast)")
    preferred_difficulty: int = Field(..., ge=1, le=5, description="Preferred difficulty level")
    
    last_activity: datetime = Field(..., description="Last learning activity timestamp")
    
    class Config:
        from_attributes = True
