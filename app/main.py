"""Main FastAPI application for the Grade 6 Math Learning Prototype."""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from pydantic import BaseModel

from app.config import settings
from app.services.graphrag import graphrag_service
from app.services.llm_service import llm_service
from app.services.embedding import embedding_service

# Request models
class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    use_graph: bool = True

class RecommendationRequest(BaseModel):
    student_id: str
    current_concept: str
    performance_score: float
    limit: int = 5

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Grade 6 Math Learning Prototype...")
    
    try:
        # Initialize services
        await graphrag_service.initialize()
        await llm_service.initialize()
        await embedding_service.initialize()
        logger.info("All services initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down services...")
    try:
        await graphrag_service.cleanup()
        await llm_service.cleanup()
        await embedding_service.cleanup()
        logger.info("All services cleaned up successfully")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="A GraphRAG-based learning recommendation system for Grade 6 math students",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Welcome to Grade 6 Math Learning Prototype",
        "version": settings.app_version,
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check service status
        graphrag_info = graphrag_service.get_graph_info()
        llm_info = llm_service.get_service_info()
        embedding_info = embedding_service.get_model_info()
        
        return {
            "status": "healthy",
            "services": {
                "graphrag": graphrag_info,
                "llm": llm_info,
                "embedding": embedding_info
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service unavailable: {str(e)}"
        )


@app.get("/concepts/{concept_id}")
async def get_concept(concept_id: str):
    """Get concept details by ID."""
    try:
        concept = graphrag_service.concepts_cache.get(concept_id)
        if not concept:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Concept {concept_id} not found"
            )
        return concept
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving concept {concept_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@app.get("/concepts/{concept_id}/next")
async def get_next_concepts(concept_id: str, limit: int = 5):
    """Get next recommended concepts for a given concept."""
    try:
        concept = graphrag_service.concepts_cache.get(concept_id)
        if not concept:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Concept {concept_id} not found"
            )
        
        next_concepts = []
        for next_id in concept.get("next_concepts", [])[:limit]:
            next_concept = graphrag_service.concepts_cache.get(next_id)
            if next_concept:
                next_concepts.append(next_concept)
        
        return {
            "current_concept": concept_id,
            "next_concepts": next_concepts,
            "total_count": len(next_concepts)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving next concepts for {concept_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@app.get("/concepts/{concept_id}/prerequisites")
async def get_prerequisites(concept_id: str):
    """Get prerequisite concepts for a given concept."""
    try:
        concept = graphrag_service.concepts_cache.get(concept_id)
        if not concept:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Concept {concept_id} not found"
            )
        
        prerequisites = []
        for prereq_id in concept.get("prerequisites", []):
            prereq_concept = graphrag_service.concepts_cache.get(prereq_id)
            if prereq_concept:
                prerequisites.append(prereq_concept)
        
        return {
            "concept": concept_id,
            "prerequisites": prerequisites,
            "total_count": len(prerequisites)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving prerequisites for {concept_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@app.post("/search")
async def search_concepts(request: SearchRequest):
    """Search for concepts using GraphRAG."""
    try:
        if not request.query.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query cannot be empty"
            )
        
        results = await graphrag_service.search_concepts(request.query, request.top_k, request.use_graph)
        
        return {
            "query": request.query,
            "results": [result.dict() for result in results],
            "total_count": len(results),
            "search_method": "graphrag" if request.use_graph else "vector_only"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching concepts: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@app.post("/recommendations")
async def get_recommendations(request: RecommendationRequest):
    """Get personalized concept recommendations for a student."""
    try:
        if not 0 <= request.performance_score <= 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Performance score must be between 0 and 1"
            )
        
        recommendations = await graphrag_service.get_concept_recommendations(
            request.student_id, request.current_concept, request.performance_score, request.limit
        )
        
        return {
            "student_id": request.student_id,
            "current_concept": request.current_concept,
            "performance_score": request.performance_score,
            "recommendations": [rec.dict() for rec in recommendations],
            "total_count": len(recommendations)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@app.post("/analyze-performance")
async def analyze_performance(student_data: dict, concept_data: dict, 
                            performance_metrics: dict):
    """Analyze student performance using LLM."""
    try:
        analysis = await llm_service.analyze_student_performance(
            student_data, concept_data, performance_metrics
        )
        
        return {
            "analysis": analysis,
            "student_id": student_data.get("student_id"),
            "concept_id": concept_data.get("concept_id"),
            "timestamp": "2024-01-15T10:30:00Z"  # Would be actual timestamp in real app
        }
    except Exception as e:
        logger.error(f"Error analyzing performance: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@app.post("/generate-practice")
async def generate_practice_problems(concept_data: dict, difficulty_level: int = 3, 
                                   count: int = 3):
    """Generate personalized practice problems using LLM."""
    try:
        if not 1 <= difficulty_level <= 5:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Difficulty level must be between 1 and 5"
            )
        
        if not 1 <= count <= 10:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Count must be between 1 and 10"
            )
        
        problems = await llm_service.generate_practice_problems(
            concept_data, difficulty_level, count
        )
        
        return {
            "concept_id": concept_data.get("concept_id"),
            "difficulty_level": difficulty_level,
            "problems": problems,
            "total_count": len(problems)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating practice problems: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@app.post("/explain-concept")
async def explain_concept(concept_data: dict, student_level: str = "beginner"):
    """Get LLM explanation of a concept."""
    try:
        if student_level not in ["beginner", "intermediate", "advanced"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Student level must be beginner, intermediate, or advanced"
            )
        
        explanation = await llm_service.explain_concept(concept_data, student_level)
        
        return {
            "concept_id": concept_data.get("concept_id"),
            "student_level": student_level,
            "explanation": explanation
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error explaining concept: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@app.get("/graph-info")
async def get_graph_info():
    """Get information about the knowledge graph."""
    try:
        info = graphrag_service.get_graph_info()
        return info
    except Exception as e:
        logger.error(f"Error getting graph info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@app.get("/llm-info")
async def get_llm_info():
    """Get information about the LLM service."""
    try:
        info = llm_service.get_service_info()
        return info
    except Exception as e:
        logger.error(f"Error getting LLM info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@app.get("/embedding-info")
async def get_embedding_info():
    """Get information about the embedding service."""
    try:
        info = embedding_service.get_model_info()
        return info
    except Exception as e:
        logger.error(f"Error getting embedding info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"}
    )


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )
