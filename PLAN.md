# Grade 6 Math Learning Prototype - Implementation Plan

## 1. Project Overview

Build a GraphRAG-based learning recommendation system that helps Grade 6 students navigate their math learning journey, starting with LCM (Least Common Multiple) and expanding to related concepts. The system will use intelligent concept retrieval and AI-powered recommendations to create personalized learning paths.

## 2. Architecture Components

### 2.1 Knowledge Graph Structure
- **JSON-based knowledge graph** representing math concepts and their relationships
- **Concept nodes**: Each math concept (LCM, GCF, fractions, etc.)
- **Relationship edges**: Prerequisites, next steps, difficulty levels, estimated time
- **Metadata**: Grade level, concept type, learning objectives, practice problems

### 2.2 Core Technologies
- **FastAPI**: RESTful API endpoints for concept retrieval and learning journey
- **GraphRAG**: Vector database with graph relationships for intelligent retrieval
- **LLM Integration**: 
  - Primary: Llama 3.2 via Ollama (local)
  - Fallback: OpenAI GPT models
- **Vector Database**: ChromaDB or FAISS for embedding storage
- **Graph Database**: NetworkX for concept relationships
- **Embedding Model**: mxbai-embed-large for high-quality concept embeddings

## 3. Knowledge Graph Schema

### 3.1 Concept Node Structure
```json
{
  "concept_id": "lcm_basic",
  "name": "Least Common Multiple (Basic)",
  "grade_level": 6,
  "description": "Finding the smallest common multiple of two numbers",
  "prerequisites": ["multiplication_basic", "divisibility_rules"],
  "next_concepts": ["lcm_advanced", "fraction_addition"],
  "difficulty": 2,
  "estimated_hours": 3,
  "learning_objectives": [
    "Understand what LCM means",
    "Find LCM of two numbers using listing method",
    "Apply LCM to solve simple word problems"
  ],
  "practice_problems": [
    "Find LCM of 6 and 8",
    "Find LCM of 12 and 18",
    "A bus comes every 15 minutes, another every 20 minutes. When will they arrive together?"
  ],
  "embedding": "[vector representation from mxbai-embed-large]"
}
```

### 3.2 Learning Journey Structure
```json
{
  "student_id": "student_001",
  "current_concept": "lcm_basic",
  "performance_score": 0.75,
  "learning_path": ["concept_1", "concept_2", "lcm_basic"],
  "next_recommendations": ["lcm_advanced", "fraction_addition"],
  "strengths": ["multiplication", "divisibility"],
  "weaknesses": ["word_problems", "larger_numbers"],
  "time_spent": 2.5,
  "attempts": 3,
  "last_updated": "2024-01-15T10:30:00Z"
}
```

### 3.3 Concept Relationships
```json
{
  "relationship_id": "rel_001",
  "from_concept": "multiplication_basic",
  "to_concept": "lcm_basic",
  "relationship_type": "prerequisite",
  "strength": 0.9,
  "description": "Strong prerequisite - multiplication is essential for understanding LCM"
}
```

## 4. API Endpoints (FastAPI)

### 4.1 Core Endpoints
- `GET /concepts/{concept_id}`: Retrieve concept details
- `GET /concepts/{concept_id}/next`: Get next recommended concepts
- `GET /learning-journey/{student_id}`: Get student's learning journey
- `POST /learning-journey/{student_id}/update`: Update student progress
- `POST /recommendations`: Get AI-powered next concept recommendations
- `GET /concepts/{concept_id}/prerequisites`: Get prerequisite concepts
- `GET /concepts/{concept_id}/related`: Get related concepts

### 4.2 LLM Integration Endpoints
- `POST /analyze-performance`: Analyze student performance and suggest improvements
- `POST /generate-practice`: Generate personalized practice problems
- `POST /explain-concept`: Get LLM explanation of concepts
- `POST /optimize-path`: Optimize learning path based on performance

### 4.3 Analytics Endpoints
- `GET /analytics/student/{student_id}`: Get detailed student analytics
- `GET /analytics/concept/{concept_id}`: Get concept performance analytics
- `GET /analytics/class-overview`: Get class-wide learning insights

## 5. GraphRAG Implementation

### 5.1 Vector Database Setup
- Store concept embeddings using **mxbai-embed-large** model
- Index concepts by grade level, difficulty, and topic
- Support semantic search across concept descriptions and learning objectives
- Implement hybrid search combining semantic similarity with graph traversal

### 5.2 Graph Relationships
- Build concept dependency graph using NetworkX
- Implement pathfinding algorithms for learning sequences
- Weight edges based on concept difficulty and prerequisites
- Support bidirectional relationships (prerequisites and next steps)

### 5.3 Retrieval Strategy
- **Hybrid Search**: Combine semantic similarity with graph traversal
- **Context-Aware Retrieval**: Consider student's current level and performance
- **Personalized Ranking**: Factor in student's learning history and preferences
- **Difficulty Progression**: Ensure logical difficulty progression in recommendations

## 6. LLM Integration Strategy

### 6.1 Ollama (Llama 3.2) Setup
- Local model serving for concept explanations
- Performance analysis and personalized recommendations
- Practice problem generation
- Learning path optimization

### 6.2 OpenAI Fallback
- API key configuration for cloud-based inference
- Consistent interface for both local and cloud models
- Cost optimization and rate limiting
- Fallback when local model is unavailable

### 6.3 Prompt Engineering
- Structured prompts for concept recommendations
- Performance analysis templates
- Learning path optimization prompts
- Practice problem generation prompts

## 7. Data Generation & Simulation

### 7.1 Knowledge Graph Simulator
- Generate realistic Grade 6 math concepts
- Create meaningful concept relationships
- Generate sample practice problems and learning objectives
- Create difficulty progression matrices

### 7.2 Student Performance Simulator
- Simulate various learning scenarios
- Generate performance data for testing
- Create diverse student profiles
- Simulate learning pace variations

### 7.3 Concept Coverage Areas
- **Number Theory**: LCM, GCF, divisibility, prime factorization
- **Fractions**: Addition, subtraction, multiplication, division
- **Decimals**: Operations, conversions, applications
- **Algebra**: Basic expressions, equations, patterns
- **Geometry**: Area, perimeter, volume, basic shapes
- **Data & Probability**: Basic statistics, simple probability

## 8. Implementation Phases

### Phase 1: Foundation (Week 1)
- Set up project structure and dependencies
- Implement basic FastAPI server
- Create knowledge graph schema and data structures
- Set up mxbai-embed-large embedding pipeline

### Phase 2: Core Functionality (Week 2)
- Implement GraphRAG retrieval system
- Build concept relationship graph
- Create basic API endpoints
- Implement vector database integration

### Phase 3: LLM Integration (Week 3)
- Integrate Ollama (Llama 3.2)
- Implement OpenAI fallback
- Build recommendation engine
- Implement performance analysis

### Phase 4: Data & Testing (Week 4)
- Generate comprehensive knowledge graph data
- Create simulation scripts
- Test end-to-end functionality
- Performance optimization and bug fixes

## 9. File Structure
```
learning-outcome/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application
│   ├── config.py               # Configuration settings
│   ├── models/                 # Pydantic models
│   │   ├── __init__.py
│   │   ├── concept.py          # Concept data models
│   │   ├── student.py          # Student data models
│   │   └── learning.py         # Learning journey models
│   ├── services/               # Business logic
│   │   ├── __init__.py
│   │   ├── graphrag.py        # GraphRAG implementation
│   │   ├── llm_service.py     # LLM integration
│   │   ├── recommendation.py  # Recommendation engine
│   │   ├── embedding.py       # mxbai-embed-large integration
│   │   └── analytics.py       # Performance analytics
│   ├── api/                   # API routes
│   │   ├── __init__.py
│   │   ├── concepts.py        # Concept endpoints
│   │   ├── students.py        # Student endpoints
│   │   ├── recommendations.py # Recommendation endpoints
│   │   └── analytics.py       # Analytics endpoints
│   ├── database/              # Database models
│   │   ├── __init__.py
│   │   ├── vector_db.py       # Vector database operations
│   │   └── graph_db.py        # Graph database operations
│   └── utils/                 # Utility functions
│       ├── __init__.py
│       ├── validators.py      # Data validation
│       └── helpers.py         # Helper functions
├── data/
│   ├── knowledge_graph.json   # Concept knowledge graph
│   ├── sample_students.json  # Sample student data
│   └── embeddings/            # Pre-computed embeddings
├── scripts/
│   ├── generate_knowledge_graph.py  # Knowledge graph generator
│   ├── simulate_students.py        # Student performance simulator
│   └── generate_embeddings.py      # Generate concept embeddings
├── tests/                     # Test files
│   ├── __init__.py
│   ├── test_api.py            # API endpoint tests
│   ├── test_services.py       # Service layer tests
│   └── test_models.py         # Model tests
├── requirements.txt            # Dependencies
├── .env.example               # Environment variables template
├── docker-compose.yml          # Docker setup (optional)
└── README.md                  # Documentation
```

## 10. Key Features

### 10.1 Core Learning Features
- **Intelligent Recommendations**: AI-powered next concept suggestions
- **Personalized Learning Paths**: Adaptive learning journeys based on performance
- **Performance Analytics**: Detailed analysis of student progress
- **Concept Relationships**: Clear prerequisite and progression mapping

### 10.2 Technical Features
- **High-Quality Embeddings**: mxbai-embed-large for superior semantic understanding
- **Flexible LLM Integration**: Support for both local and cloud models
- **Scalable Architecture**: Easy to extend with new concepts and features
- **Real-time Updates**: Dynamic learning path adjustments

### 10.3 User Experience Features
- **Progress Tracking**: Visual learning journey representation
- **Difficulty Adaptation**: Automatic difficulty adjustment based on performance
- **Practice Generation**: Personalized practice problems
- **Performance Insights**: Detailed feedback and improvement suggestions

## 11. Technical Considerations

### 11.1 Performance
- **Efficient Vector Search**: Optimized mxbai-embed-large embeddings
- **Graph Traversal**: Fast concept relationship navigation
- **Caching Strategy**: Cache frequently accessed concepts and embeddings
- **Async Processing**: Non-blocking API responses

### 11.2 Scalability
- **Modular Design**: Easy to add new concepts and features
- **Database Optimization**: Efficient indexing and querying
- **Load Balancing**: Handle multiple concurrent students
- **Horizontal Scaling**: Support for multiple instances

### 11.3 Security & Monitoring
- **API Authentication**: Secure endpoint access
- **Rate Limiting**: Prevent API abuse
- **Logging**: Comprehensive request and error logging
- **Performance Metrics**: Monitor system health and response times

### 11.4 Testing Strategy
- **Unit Tests**: Test individual components and functions
- **Integration Tests**: Test API endpoints and service interactions
- **Performance Tests**: Test embedding generation and retrieval speed
- **End-to-End Tests**: Test complete learning recommendation flow

## 12. Dependencies & Requirements

### 12.1 Core Dependencies
- **FastAPI**: Web framework for API development
- **mxbai-embed-large**: High-quality embedding model
- **ChromaDB/FAISS**: Vector database for embeddings
- **NetworkX**: Graph operations and algorithms
- **Pydantic**: Data validation and serialization

### 12.2 LLM Dependencies
- **Ollama**: Local LLM serving
- **OpenAI**: Cloud LLM API integration
- **LangChain**: LLM orchestration (optional)

### 12.3 Utility Dependencies
- **NumPy**: Numerical operations
- **Pandas**: Data manipulation
- **Pytest**: Testing framework
- **Uvicorn**: ASGI server

## 13. Success Metrics

### 13.1 Technical Metrics
- **API Response Time**: < 200ms for concept retrieval
- **Embedding Generation**: < 1s per concept
- **Recommendation Accuracy**: > 85% relevant suggestions
- **System Uptime**: > 99.5%

### 13.2 Learning Metrics
- **Concept Mastery**: Improved student performance scores
- **Learning Efficiency**: Reduced time to concept mastery
- **Student Engagement**: Increased practice problem completion
- **Path Optimization**: Shorter learning paths to mastery

## 14. Future Enhancements

### 14.1 Advanced Features
- **Multi-modal Learning**: Support for images, videos, and interactive content
- **Collaborative Learning**: Group study recommendations
- **Adaptive Testing**: Dynamic assessment based on performance
- **Parent/Teacher Dashboard**: Progress monitoring and insights

### 14.2 Technical Improvements
- **Real-time Updates**: WebSocket support for live progress updates
- **Mobile App**: Native mobile application
- **Offline Support**: Local caching for offline learning
- **AI Tutoring**: Conversational AI for concept explanations

This comprehensive plan provides a solid foundation for building a sophisticated learning recommendation system using GraphRAG and mxbai-embed-large embeddings. The system will deliver intelligent, personalized learning experiences for Grade 6 math students.
