"""Recommendation service for personalized learning suggestions."""

import logging
from typing import List, Dict, Any, Optional
from app.models.concept import ConceptRecommendation
from app.services.graphrag import graphrag_service
from app.services.llm_service import llm_service

logger = logging.getLogger(__name__)


class RecommendationService:
    """Service for generating personalized learning recommendations."""
    
    def __init__(self):
        """Initialize the recommendation service."""
        self._initialized = False
        
    async def initialize(self):
        """Initialize the recommendation service."""
        if self._initialized:
            return
            
        try:
            # Initialize dependencies
            await graphrag_service.initialize()
            await llm_service.initialize()
            
            self._initialized = True
            logger.info("Recommendation service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize recommendation service: {e}")
            raise
    
    async def get_next_concept_recommendations(self, student_id: str, current_concept: str, 
                                            performance_score: float, limit: int = 5) -> List[ConceptRecommendation]:
        """Get next concept recommendations for a student."""
        await self.initialize()
        
        try:
            # Get basic recommendations from GraphRAG
            recommendations = await graphrag_service.get_concept_recommendations(
                student_id, current_concept, performance_score, limit
            )
            
            # Enhance with LLM insights if available
            if recommendations and llm_service._initialized and llm_service.current_provider != "none":
                enhanced_recommendations = await self._enhance_recommendations_with_llm(
                    recommendations, student_id, current_concept, performance_score
                )
                return enhanced_recommendations
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting concept recommendations: {e}")
            return []
    
    async def _enhance_recommendations_with_llm(self, recommendations: List[ConceptRecommendation], 
                                              student_id: str, current_concept: str, 
                                              performance_score: float) -> List[ConceptRecommendation]:
        """Enhance recommendations using LLM insights."""
        try:
            enhanced_recommendations = []
            
            for recommendation in recommendations:
                # Generate enhanced reasoning using LLM
                enhanced_reasoning = await self._generate_enhanced_reasoning(
                    recommendation, student_id, current_concept, performance_score
                )
                
                # Update the recommendation with enhanced reasoning
                recommendation.reasoning = enhanced_reasoning
                enhanced_recommendations.append(recommendation)
            
            return enhanced_recommendations
            
        except Exception as e:
            logger.error(f"Error enhancing recommendations with LLM: {e}")
            return recommendations
    
    async def _generate_enhanced_reasoning(self, recommendation: ConceptRecommendation, 
                                        student_id: str, current_concept: str, 
                                        performance_score: float) -> str:
        """Generate enhanced reasoning for a recommendation using LLM."""
        try:
            prompt = f"""
            Analyze this learning recommendation and provide enhanced reasoning:
            
            Current Concept: {current_concept}
            Recommended Concept: {recommendation.name}
            Student Performance: {performance_score:.2f}
            Current Reasoning: {recommendation.reasoning}
            
            Please provide:
            1. Why this concept is the best next step
            2. How it builds on current knowledge
            3. Specific benefits for the student
            4. Any preparation needed
            
            Keep the response concise and encouraging.
            """
            
            enhanced_reasoning = await llm_service.generate_response(
                prompt, 
                system_prompt="You are an expert math teacher providing personalized learning guidance.",
                max_tokens=200,
                temperature=0.3
            )
            
            return enhanced_reasoning.strip()
            
        except Exception as e:
            logger.error(f"Error generating enhanced reasoning: {e}")
            return recommendation.reasoning
    
    async def get_learning_path_recommendations(self, student_id: str, 
                                              completed_concepts: List[str], 
                                              learning_goals: List[str]) -> List[Dict[str, Any]]:
        """Get recommendations for optimal learning paths."""
        await self.initialize()
        
        try:
            # Analyze completed concepts to understand student's learning pattern
            learning_pattern = await self._analyze_learning_pattern(completed_concepts)
            
            # Generate path recommendations based on goals and pattern
            path_recommendations = await self._generate_path_recommendations(
                learning_pattern, learning_goals
            )
            
            return path_recommendations
            
        except Exception as e:
            logger.error(f"Error getting learning path recommendations: {e}")
            return []
    
    async def _analyze_learning_pattern(self, completed_concepts: List[str]) -> Dict[str, Any]:
        """Analyze the student's learning pattern from completed concepts."""
        try:
            # This would analyze the sequence and types of concepts completed
            # For now, return a basic pattern
            return {
                "concept_count": len(completed_concepts),
                "learning_pace": "average",
                "strengths": ["foundational_math"],
                "preferred_difficulty": 2
            }
            
        except Exception as e:
            logger.error(f"Error analyzing learning pattern: {e}")
            return {}
    
    async def _generate_path_recommendations(self, learning_pattern: Dict[str, Any], 
                                           learning_goals: List[str]) -> List[Dict[str, Any]]:
        """Generate learning path recommendations."""
        try:
            # This would generate optimal learning paths based on pattern and goals
            # For now, return basic recommendations
            return [
                {
                    "path_id": "accelerated_path",
                    "name": "Accelerated Learning Path",
                    "description": "Fast-paced path for quick learners",
                    "estimated_time": 15.0,
                    "difficulty_progression": [1, 2, 3, 4, 5]
                },
                {
                    "path_id": "steady_path",
                    "name": "Steady Progress Path",
                    "description": "Balanced pace with extra practice",
                    "estimated_time": 20.0,
                    "difficulty_progression": [1, 1, 2, 2, 3, 3, 4, 5]
                }
            ]
            
        except Exception as e:
            logger.error(f"Error generating path recommendations: {e}")
            return []
    
    async def get_practice_recommendations(self, student_id: str, concept_id: str, 
                                         performance_score: float) -> Dict[str, Any]:
        """Get personalized practice recommendations."""
        await self.initialize()
        
        try:
            # Analyze performance to determine practice needs
            practice_focus = await self._determine_practice_focus(performance_score)
            
            # Generate practice recommendations
            practice_recommendations = {
                "concept_id": concept_id,
                "practice_focus": practice_focus,
                "recommended_practice_time": self._calculate_practice_time(performance_score),
                "practice_types": self._get_practice_types(performance_score),
                "difficulty_adjustment": self._get_difficulty_adjustment(performance_score)
            }
            
            return practice_recommendations
            
        except Exception as e:
            logger.error(f"Error getting practice recommendations: {e}")
            return {}
    
    def _determine_practice_focus(self, performance_score: float) -> str:
        """Determine what the student should focus on in practice."""
        if performance_score < 0.4:
            return "fundamentals"
        elif performance_score < 0.7:
            return "application"
        else:
            return "enrichment"
    
    def _calculate_practice_time(self, performance_score: float) -> float:
        """Calculate recommended practice time in hours."""
        if performance_score < 0.4:
            return 2.0  # More practice for struggling students
        elif performance_score < 0.7:
            return 1.5  # Moderate practice
        else:
            return 1.0  # Less practice for high performers
    
    def _get_practice_types(self, performance_score: float) -> List[str]:
        """Get recommended practice types based on performance."""
        if performance_score < 0.4:
            return ["drill_practice", "visual_aids", "step_by_step"]
        elif performance_score < 0.7:
            return ["word_problems", "real_world_applications", "peer_teaching"]
        else:
            return ["challenge_problems", "teaching_others", "creative_applications"]
    
    def _get_difficulty_adjustment(self, performance_score: float) -> str:
        """Get difficulty adjustment recommendation."""
        if performance_score < 0.4:
            return "decrease"
        elif performance_score > 0.8:
            return "increase"
        else:
            return "maintain"
    
    async def cleanup(self):
        """Clean up resources."""
        try:
            self._initialized = False
            logger.info("Recommendation service cleaned up successfully")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


# Global recommendation service instance
recommendation_service = RecommendationService()
