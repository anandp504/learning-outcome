"""LLM service for integrating with Ollama and OpenAI."""

import asyncio
from typing import List, Dict, Any, Optional, Union
import ollama
import openai
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings

logger = logging.getLogger(__name__)


class LLMService:
    """Service for LLM operations using Ollama (local) and OpenAI (fallback)."""
    
    def __init__(self):
        """Initialize the LLM service."""
        self.ollama_client = None
        self.openai_client = None
        self.current_provider = "ollama"
        self._initialized = False
        
        # Initialize clients
        self._init_ollama()
        self._init_openai()
    
    def _init_ollama(self):
        """Initialize Ollama client."""
        try:
            self.ollama_client = ollama.Client(host=settings.ollama_base_url)
            # Test connection
            models = self.ollama_client.list()
            logger.info(f"Ollama initialized successfully. Available models: {[m['name'] for m in models['models']]}")
        except Exception as e:
            logger.warning(f"Failed to initialize Ollama: {e}")
            self.ollama_client = None
    
    def _init_openai(self):
        """Initialize OpenAI client."""
        try:
            if settings.openai_api_key:
                self.openai_client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
                logger.info("OpenAI client initialized successfully")
            else:
                logger.warning("OpenAI API key not provided")
                self.openai_client = None
        except Exception as e:
            logger.warning(f"Failed to initialize OpenAI: {e}")
            self.openai_client = None
    
    async def initialize(self):
        """Initialize the LLM service asynchronously."""
        if self._initialized:
            return
        
        # Check which provider is available
        if self.ollama_client:
            self.current_provider = "ollama"
            logger.info("Using Ollama as primary LLM provider")
        elif self.openai_client:
            self.current_provider = "openai"
            logger.info("Using OpenAI as primary LLM provider")
        else:
            # Instead of failing, log a warning and continue with limited functionality
            logger.warning("No LLM provider available - LLM features will be disabled")
            self.current_provider = "none"
        
        self._initialized = True
        logger.info("LLM service initialized successfully")
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def generate_response(self, prompt: str, system_prompt: str = "", 
                              max_tokens: int = 1000, temperature: float = 0.7) -> str:
        """Generate response using the available LLM provider."""
        await self.initialize()
        
        # Check if any provider is available
        if self.current_provider == "none":
            return "LLM service is not available. Please configure Ollama or OpenAI API key."
        
        try:
            if self.current_provider == "ollama" and self.ollama_client:
                return await self._generate_ollama_response(prompt, system_prompt, max_tokens, temperature)
            elif self.openai_client:
                return await self._generate_openai_response(prompt, system_prompt, max_tokens, temperature)
            else:
                raise RuntimeError("No LLM provider available")
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            # Try to fallback to OpenAI if using Ollama
            if self.current_provider == "ollama" and self.openai_client:
                logger.info("Falling back to OpenAI")
                self.current_provider = "openai"
                return await self._generate_openai_response(prompt, system_prompt, max_tokens, temperature)
            raise
    
    async def _generate_ollama_response(self, prompt: str, system_prompt: str = "", 
                                      max_tokens: int = 1000, temperature: float = 0.7) -> str:
        """Generate response using Ollama."""
        try:
            # Prepare messages
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            # Generate response
            response = await asyncio.to_thread(
                self.ollama_client.chat,
                model=settings.ollama_model,
                messages=messages,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            )
            
            return response["message"]["content"]
            
        except Exception as e:
            logger.error(f"Error generating Ollama response: {e}")
            raise
    
    async def _generate_openai_response(self, prompt: str, system_prompt: str = "", 
                                      max_tokens: int = 1000, temperature: float = 0.7) -> str:
        """Generate response using OpenAI."""
        try:
            # Prepare messages
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            # Generate response
            response = await self.openai_client.chat.completions.create(
                model=settings.openai_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating OpenAI response: {e}")
            raise
    
    async def analyze_student_performance(self, student_data: Dict[str, Any], 
                                        concept_data: Dict[str, Any], 
                                        performance_metrics: Dict[str, Any]) -> str:
        """Analyze student performance and provide insights."""
        try:
            system_prompt = """You are an expert math teacher analyzing a Grade 6 student's performance. 
            Provide constructive feedback and specific recommendations for improvement. 
            Be encouraging and focus on actionable steps."""
            
            prompt = f"""
            Analyze this student's performance and provide insights:
            
            Student: {student_data.get('name', 'Unknown')}, Grade {student_data.get('grade_level', 6)}
            Current Concept: {concept_data.get('name', 'Unknown')}
            
            Performance Metrics:
            - Score: {performance_metrics.get('performance_score', 0):.2f}
            - Time Spent: {performance_metrics.get('time_spent', 0):.1f} hours
            - Attempts: {performance_metrics.get('attempts', 0)}
            
            Strengths: {', '.join(performance_metrics.get('strengths', []))}
            Weaknesses: {', '.join(performance_metrics.get('weaknesses', []))}
            
            Please provide:
            1. Analysis of current performance
            2. Specific areas for improvement
            3. Recommended study strategies
            4. Encouragement and motivation
            """
            
            response = await self.generate_response(prompt, system_prompt, max_tokens=500, temperature=0.3)
            return response
            
        except Exception as e:
            logger.error(f"Error analyzing student performance: {e}")
            return "Unable to analyze performance at this time. Please try again later."
    
    async def generate_practice_problems(self, concept_data: Dict[str, Any], 
                                       difficulty_level: int, 
                                       count: int = 3) -> List[Dict[str, Any]]:
        """Generate personalized practice problems for a concept."""
        try:
            system_prompt = """You are an expert math teacher creating practice problems for Grade 6 students. 
            Generate problems that are appropriate for the specified difficulty level and concept. 
            Provide clear, step-by-step solutions."""
            
            prompt = f"""
            Generate {count} practice problems for this concept:
            
            Concept: {concept_data.get('name', 'Unknown')}
            Description: {concept_data.get('description', '')}
            Difficulty Level: {difficulty_level}/5
            Grade Level: {concept_data.get('grade_level', 6)}
            
            Please provide problems in this format:
            Problem 1: [Question]
            Solution 1: [Step-by-step solution]
            
            Problem 2: [Question]
            Solution 2: [Step-by-step solution]
            
            And so on...
            
            Make sure the problems are:
            - Age-appropriate for Grade 6
            - Match the specified difficulty level
            - Include clear, educational solutions
            - Build understanding progressively
            """
            
            response = await self.generate_response(prompt, system_prompt, max_tokens=800, temperature=0.4)
            
            # Parse the response into structured format
            problems = self._parse_practice_problems(response)
            return problems
            
        except Exception as e:
            logger.error(f"Error generating practice problems: {e}")
            return []
    
    def _parse_practice_problems(self, response: str) -> List[Dict[str, Any]]:
        """Parse LLM response into structured practice problems."""
        try:
            problems = []
            lines = response.split('\n')
            current_problem = None
            
            for line in lines:
                line = line.strip()
                if line.startswith('Problem'):
                    if current_problem:
                        problems.append(current_problem)
                    current_problem = {'question': '', 'solution': ''}
                elif line.startswith('Solution') and current_problem:
                    continue
                elif current_problem and line:
                    if not current_problem['question']:
                        current_problem['question'] = line
                    else:
                        current_problem['solution'] += line + '\n'
            
            if current_problem:
                problems.append(current_problem)
            
            return problems
            
        except Exception as e:
            logger.error(f"Error parsing practice problems: {e}")
            return []
    
    async def explain_concept(self, concept_data: Dict[str, Any], 
                            student_level: str = "beginner") -> str:
        """Generate a student-friendly explanation of a concept."""
        try:
            system_prompt = f"""You are an expert math teacher explaining concepts to Grade 6 students. 
            Adapt your explanation to a {student_level} level. Use clear language, examples, and analogies 
            that students can relate to."""
            
            prompt = f"""
            Explain this math concept in a way that a Grade 6 student can understand:
            
            Concept: {concept_data.get('name', 'Unknown')}
            Description: {concept_data.get('description', '')}
            Learning Objectives: {', '.join(concept_data.get('learning_objectives', []))}
            
            Please provide:
            1. A simple, clear explanation
            2. Real-world examples or analogies
            3. Step-by-step approach if applicable
            4. Common mistakes to avoid
            5. Tips for remembering the concept
            
            Make it engaging and easy to follow for a {student_level} level student.
            """
            
            response = await self.generate_response(prompt, system_prompt, max_tokens=600, temperature=0.3)
            return response
            
        except Exception as e:
            logger.error(f"Error explaining concept: {e}")
            return "Unable to generate explanation at this time. Please try again later."
    
    async def optimize_learning_path(self, student_data: Dict[str, Any], 
                                   current_concept: str, 
                                   available_concepts: List[Dict[str, Any]]) -> str:
        """Optimize learning path based on student profile and performance."""
        try:
            system_prompt = """You are an expert educational psychologist specializing in math learning paths. 
            Analyze the student's profile and recommend the optimal sequence of concepts to study. 
            Consider learning style, current performance, and educational best practices."""
            
            prompt = f"""
            Optimize the learning path for this student:
            
            Student Profile:
            - Name: {student_data.get('name', 'Unknown')}
            - Grade: {student_data.get('grade_level', 6)}
            - Learning Style: {student_data.get('learning_style', 'Unknown')}
            - Math Aptitude: {student_data.get('math_aptitude', 'Unknown')}
            - Current Concept: {current_concept}
            
            Available Next Concepts:
            {self._format_concepts_for_prompt(available_concepts)}
            
            Please recommend:
            1. The optimal sequence of concepts to study next
            2. Reasoning for the recommended order
            3. Estimated time for each concept
            4. Alternative paths if the student struggles
            5. Strategies to maximize learning efficiency
            
            Consider the student's learning style and current performance level.
            """
            
            response = await self.generate_response(prompt, system_prompt, max_tokens=700, temperature=0.4)
            return response
            
        except Exception as e:
            logger.error(f"Error optimizing learning path: {e}")
            return "Unable to optimize learning path at this time. Please try again later."
    
    def _format_concepts_for_prompt(self, concepts: List[Dict[str, Any]]) -> str:
        """Format concepts for inclusion in LLM prompts."""
        formatted = []
        for concept in concepts:
            formatted.append(f"- {concept.get('name', 'Unknown')}: {concept.get('description', '')[:100]}...")
        return '\n'.join(formatted)
    
    async def get_learning_insights(self, student_data: Dict[str, Any], 
                                  learning_history: List[Dict[str, Any]]) -> str:
        """Generate insights about student learning patterns."""
        try:
            system_prompt = """You are an expert educational analyst. Analyze the student's learning history 
            and provide insights about their learning patterns, strengths, and areas for improvement. 
            Be encouraging and provide actionable recommendations."""
            
            prompt = f"""
            Analyze this student's learning history and provide insights:
            
            Student: {student_data.get('name', 'Unknown')}, Grade {student_data.get('grade_level', 6)}
            
            Learning History Summary:
            - Total Concepts Studied: {len(learning_history)}
            - Average Performance: {self._calculate_average_performance(learning_history):.2f}
            - Learning Pace: {self._assess_learning_pace(learning_history)}
            
            Recent Performance:
            {self._format_recent_performance(learning_history)}
            
            Please provide:
            1. Analysis of learning patterns
            2. Identified strengths and learning preferences
            3. Areas that need attention
            4. Recommendations for improvement
            5. Encouragement and motivation
            
            Focus on actionable insights and positive reinforcement.
            """
            
            response = await self.generate_response(prompt, system_prompt, max_tokens=600, temperature=0.3)
            return response
            
        except Exception as e:
            logger.error(f"Error getting learning insights: {e}")
            return "Unable to generate learning insights at this time. Please try again later."
    
    def _calculate_average_performance(self, learning_history: List[Dict[str, Any]]) -> float:
        """Calculate average performance from learning history."""
        if not learning_history:
            return 0.0
        
        total_score = sum(item.get('performance_score', 0) for item in learning_history)
        return total_score / len(learning_history)
    
    def _assess_learning_pace(self, learning_history: List[Dict[str, Any]]) -> str:
        """Assess the student's learning pace."""
        if len(learning_history) < 2:
            return "insufficient data"
        
        # Simple heuristic based on time spent and performance
        avg_time = sum(item.get('time_spent', 0) for item in learning_history) / len(learning_history)
        avg_performance = self._calculate_average_performance(learning_history)
        
        if avg_time < 2.0 and avg_performance > 0.7:
            return "fast"
        elif avg_time > 4.0 and avg_performance < 0.6:
            return "slow"
        else:
            return "average"
    
    def _format_recent_performance(self, learning_history: List[Dict[str, Any]]) -> str:
        """Format recent performance for prompt inclusion."""
        if not learning_history:
            return "No learning history available"
        
        recent = learning_history[-3:]  # Last 3 concepts
        formatted = []
        for item in recent:
            formatted.append(f"- {item.get('concept_name', 'Unknown')}: Score {item.get('performance_score', 0):.2f}")
        
        return '\n'.join(formatted)
    
    def get_service_info(self) -> Dict[str, Any]:
        """Get information about the LLM service."""
        return {
            "current_provider": self.current_provider,
            "ollama_available": self.ollama_client is not None,
            "openai_available": self.openai_client is not None,
            "initialized": self._initialized
        }
    
    async def cleanup(self):
        """Clean up resources."""
        try:
            # Clean up any resources if needed
            logger.info("LLM service cleaned up successfully")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


# Global LLM service instance
llm_service = LLMService()
