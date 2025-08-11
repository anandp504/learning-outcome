"""Analytics service for learning insights and performance analysis."""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)


class AnalyticsService:
    """Service for analyzing student learning patterns and performance."""
    
    def __init__(self):
        """Initialize the analytics service."""
        self._initialized = False
        
    async def initialize(self):
        """Initialize the analytics service."""
        if self._initialized:
            return
            
        try:
            self._initialized = True
            logger.info("Analytics service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize analytics service: {e}")
            raise
    
    async def analyze_student_performance(self, student_id: str, 
                                       performance_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze overall student performance patterns."""
        await self.initialize()
        
        try:
            if not performance_data:
                return {"error": "No performance data available"}
            
            # Calculate basic statistics
            total_concepts = len(performance_data)
            avg_score = sum(p.get("performance_score", 0) for p in performance_data) / total_concepts
            total_time = sum(p.get("time_spent", 0) for p in performance_data)
            avg_attempts = sum(p.get("attempts", 0) for p in performance_data) / total_concepts
            
            # Analyze performance trends
            performance_trends = self._analyze_performance_trends(performance_data)
            
            # Identify strengths and weaknesses
            strengths_weaknesses = self._identify_strengths_weaknesses(performance_data)
            
            # Learning pace analysis
            learning_pace = self._analyze_learning_pace(performance_data)
            
            # Generate insights
            insights = self._generate_performance_insights(avg_score, performance_trends, learning_pace)
            
            analysis = {
                "student_id": student_id,
                "summary": {
                    "total_concepts_studied": total_concepts,
                    "average_performance_score": round(avg_score, 3),
                    "total_study_time_hours": round(total_time, 2),
                    "average_attempts_per_concept": round(avg_attempts, 2)
                },
                "performance_trends": performance_trends,
                "strengths_weaknesses": strengths_weaknesses,
                "learning_pace": learning_pace,
                "insights": insights,
                "recommendations": self._generate_recommendations(avg_score, performance_trends),
                "generated_at": datetime.now().isoformat()
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing student performance: {e}")
            return {"error": f"Analysis failed: {str(e)}"}
    
    def _analyze_performance_trends(self, performance_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance trends over time."""
        try:
            # Sort by last_attempt timestamp
            sorted_data = sorted(performance_data, 
                               key=lambda x: x.get("last_attempt", ""), 
                               reverse=True)
            
            if len(sorted_data) < 2:
                return {"trend": "insufficient_data", "description": "Need more data for trend analysis"}
            
            # Calculate recent vs earlier performance
            recent_count = min(3, len(sorted_data))
            recent_scores = [p.get("performance_score", 0) for p in sorted_data[:recent_count]]
            earlier_scores = [p.get("performance_score", 0) for p in sorted_data[-recent_count:]]
            
            recent_avg = sum(recent_scores) / len(recent_scores)
            earlier_avg = sum(earlier_scores) / len(earlier_scores)
            
            if recent_avg > earlier_avg + 0.1:
                trend = "improving"
                description = "Performance is showing positive trends"
            elif recent_avg < earlier_avg - 0.1:
                trend = "declining"
                description = "Performance may need attention"
            else:
                trend = "stable"
                description = "Performance is consistent"
            
            return {
                "trend": trend,
                "description": description,
                "recent_average": round(recent_avg, 3),
                "earlier_average": round(earlier_avg, 3),
                "change": round(recent_avg - earlier_avg, 3)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing performance trends: {e}")
            return {"trend": "error", "description": "Unable to analyze trends"}
    
    def _identify_strengths_weaknesses(self, performance_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Identify student strengths and areas for improvement."""
        try:
            # Group by concept type or difficulty
            concept_performance = {}
            for p in performance_data:
                concept_id = p.get("concept_id", "unknown")
                score = p.get("performance_score", 0)
                if concept_id not in concept_performance:
                    concept_performance[concept_id] = []
                concept_performance[concept_id].append(score)
            
            # Calculate average scores per concept
            concept_averages = {}
            for concept_id, scores in concept_performance.items():
                concept_averages[concept_id] = sum(scores) / len(scores)
            
            # Identify strengths (concepts with high performance)
            strengths = []
            for concept_id, avg_score in concept_averages.items():
                if avg_score >= 0.8:
                    strengths.append({
                        "concept_id": concept_id,
                        "average_score": round(avg_score, 3),
                        "strength_level": "strong"
                    })
            
            # Identify weaknesses (concepts with low performance)
            weaknesses = []
            for concept_id, avg_score in concept_averages.items():
                if avg_score < 0.6:
                    weaknesses.append({
                        "concept_id": concept_id,
                        "average_score": round(avg_score, 3),
                        "improvement_needed": "high" if avg_score < 0.4 else "moderate"
                    })
            
            return {
                "strengths": strengths,
                "weaknesses": weaknesses,
                "total_concepts_analyzed": len(concept_averages)
            }
            
        except Exception as e:
            logger.error(f"Error identifying strengths and weaknesses: {e}")
            return {"strengths": [], "weaknesses": [], "error": str(e)}
    
    def _analyze_learning_pace(self, performance_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the student's learning pace."""
        try:
            if len(performance_data) < 2:
                return {"pace": "insufficient_data", "description": "Need more data for pace analysis"}
            
            # Calculate average time per concept
            total_time = sum(p.get("time_spent", 0) for p in performance_data)
            avg_time_per_concept = total_time / len(performance_data)
            
            # Calculate average attempts per concept
            total_attempts = sum(p.get("attempts", 0) for p in performance_data)
            avg_attempts_per_concept = total_attempts / len(performance_data)
            
            # Determine pace based on time and attempts
            if avg_time_per_concept < 2.0 and avg_attempts_per_concept < 2.0:
                pace = "fast"
                description = "Student learns concepts quickly with minimal attempts"
            elif avg_time_per_concept > 4.0 or avg_attempts_per_concept > 4.0:
                pace = "slow"
                description = "Student takes more time and attempts to master concepts"
            else:
                pace = "average"
                description = "Student has a balanced learning pace"
            
            return {
                "pace": pace,
                "description": description,
                "average_time_per_concept_hours": round(avg_time_per_concept, 2),
                "average_attempts_per_concept": round(avg_attempts_per_concept, 2),
                "total_study_time_hours": round(total_time, 2)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing learning pace: {e}")
            return {"pace": "error", "description": "Unable to analyze learning pace"}
    
    def _generate_performance_insights(self, avg_score: float, 
                                     performance_trends: Dict[str, Any], 
                                     learning_pace: Dict[str, Any]) -> List[str]:
        """Generate actionable insights from performance data."""
        insights = []
        
        try:
            # Performance level insights
            if avg_score >= 0.8:
                insights.append("Excellent performance! You're mastering concepts effectively.")
            elif avg_score >= 0.6:
                insights.append("Good progress! Focus on areas that need improvement.")
            else:
                insights.append("Keep practicing! Consider reviewing foundational concepts.")
            
            # Trend insights
            trend = performance_trends.get("trend", "")
            if trend == "improving":
                insights.append("Your performance is improving - great work!")
            elif trend == "declining":
                insights.append("Consider reviewing recent concepts to maintain progress.")
            
            # Pace insights
            pace = learning_pace.get("pace", "")
            if pace == "fast":
                insights.append("You learn quickly - consider challenging yourself with advanced topics.")
            elif pace == "slow":
                insights.append("Take your time to ensure solid understanding - quality over speed.")
            
            # Study time insights
            avg_time = learning_pace.get("average_time_per_concept_hours", 0)
            if avg_time > 4.0:
                insights.append("You spend significant time on concepts - consider breaking them into smaller parts.")
            elif avg_time < 1.0:
                insights.append("Quick learning is great, but ensure you're not missing important details.")
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            insights.append("Unable to generate specific insights at this time.")
        
        return insights
    
    def _generate_recommendations(self, avg_score: float, 
                                performance_trends: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on performance."""
        recommendations = []
        
        try:
            # Performance-based recommendations
            if avg_score < 0.6:
                recommendations.append("Review foundational concepts before moving to advanced topics")
                recommendations.append("Practice with simpler problems to build confidence")
            elif avg_score < 0.8:
                recommendations.append("Focus on areas where you scored below 70%")
                recommendations.append("Try more challenging problems to push your limits")
            else:
                recommendations.append("You're ready for more advanced concepts")
                recommendations.append("Consider helping peers to reinforce your understanding")
            
            # Trend-based recommendations
            trend = performance_trends.get("trend", "")
            if trend == "declining":
                recommendations.append("Schedule regular review sessions to maintain knowledge")
                recommendations.append("Identify what changed in your study routine")
            elif trend == "improving":
                recommendations.append("Maintain your current study strategies")
                recommendations.append("Set higher goals for continued improvement")
            
            # General recommendations
            recommendations.append("Practice regularly, even with mastered concepts")
            recommendations.append("Use different study methods to reinforce learning")
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            recommendations.append("Focus on consistent practice and review")
        
        return recommendations
    
    async def get_class_overview(self, all_students_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get overview analytics for the entire class."""
        await self.initialize()
        
        try:
            if not all_students_data:
                return {"error": "No student data available"}
            
            # Calculate class-wide statistics
            total_students = len(all_students_data)
            class_performance_scores = []
            class_study_times = []
            
            for student_data in all_students_data:
                performance_data = student_data.get("performance_data", [])
                if performance_data:
                    avg_score = sum(p.get("performance_score", 0) for p in performance_data) / len(performance_data)
                    class_performance_scores.append(avg_score)
                    
                    total_time = sum(p.get("time_spent", 0) for p in performance_data)
                    class_study_times.append(total_time)
            
            if class_performance_scores:
                class_avg_performance = sum(class_performance_scores) / len(class_performance_scores)
                class_avg_study_time = sum(class_study_times) / len(class_study_times)
            else:
                class_avg_performance = 0
                class_avg_study_time = 0
            
            # Performance distribution
            performance_distribution = {
                "excellent": len([s for s in class_performance_scores if s >= 0.8]),
                "good": len([s for s in class_performance_scores if 0.6 <= s < 0.8]),
                "developing": len([s for s in class_performance_scores if 0.4 <= s < 0.6]),
                "needs_help": len([s for s in class_performance_scores if s < 0.4])
            }
            
            class_overview = {
                "total_students": total_students,
                "class_average_performance": round(class_avg_performance, 3),
                "class_average_study_time_hours": round(class_avg_study_time, 2),
                "performance_distribution": performance_distribution,
                "top_performers": self._identify_top_performers(all_students_data),
                "concepts_needing_attention": self._identify_concepts_needing_attention(all_students_data),
                "generated_at": datetime.now().isoformat()
            }
            
            return class_overview
            
        except Exception as e:
            logger.error(f"Error getting class overview: {e}")
            return {"error": f"Class overview failed: {str(e)}"}
    
    def _identify_top_performers(self, all_students_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify top performing students."""
        try:
            student_performances = []
            
            for student_data in all_students_data:
                student_id = student_data.get("student_id", "unknown")
                student_name = student_data.get("name", "Unknown")
                performance_data = student_data.get("performance_data", [])
                
                if performance_data:
                    avg_score = sum(p.get("performance_score", 0) for p in performance_data) / len(performance_data)
                    student_performances.append({
                        "student_id": student_id,
                        "name": student_name,
                        "average_score": round(avg_score, 3)
                    })
            
            # Sort by performance and return top 3
            student_performances.sort(key=lambda x: x["average_score"], reverse=True)
            return student_performances[:3]
            
        except Exception as e:
            logger.error(f"Error identifying top performers: {e}")
            return []
    
    def _identify_concepts_needing_attention(self, all_students_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify concepts that many students struggle with."""
        try:
            concept_difficulties = {}
            
            for student_data in all_students_data:
                performance_data = student_data.get("performance_data", [])
                for p in performance_data:
                    concept_id = p.get("concept_id", "unknown")
                    score = p.get("performance_score", 0)
                    
                    if concept_id not in concept_difficulties:
                        concept_difficulties[concept_id] = {"scores": [], "count": 0}
                    
                    concept_difficulties[concept_id]["scores"].append(score)
                    concept_difficulties[concept_id]["count"] += 1
            
            # Calculate average difficulty per concept
            concept_analysis = []
            for concept_id, data in concept_difficulties.items():
                if data["count"] >= 3:  # Only consider concepts with sufficient data
                    avg_score = sum(data["scores"]) / len(data["scores"])
                    concept_analysis.append({
                        "concept_id": concept_id,
                        "average_score": round(avg_score, 3),
                        "student_count": data["count"],
                        "difficulty_level": "high" if avg_score < 0.5 else "moderate" if avg_score < 0.7 else "low"
                    })
            
            # Sort by difficulty (lowest scores first)
            concept_analysis.sort(key=lambda x: x["average_score"])
            return concept_analysis[:5]  # Return top 5 most difficult concepts
            
        except Exception as e:
            logger.error(f"Error identifying concepts needing attention: {e}")
            return []
    
    async def cleanup(self):
        """Clean up resources."""
        try:
            self._initialized = False
            logger.info("Analytics service cleaned up successfully")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


# Global analytics service instance
analytics_service = AnalyticsService()
