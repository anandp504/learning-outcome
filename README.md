# Grade 6 Math Learning Prototype

A GraphRAG-based learning recommendation system for Grade 6 math students, combining knowledge graphs with vector embeddings and LLM-powered insights.

## 🚀 Features

- **GraphRAG Architecture**: Combines graph relationships with vector similarity for intelligent concept retrieval
- **Personalized Learning**: Recommends next concepts based on student performance and learning patterns
- **Knowledge Graph**: JSON-based representation of math concepts with prerequisites and relationships
- **Vector Search**: Semantic search using concept embeddings
- **LLM Integration**: Support for Ollama (local) and OpenAI (fallback) for enhanced explanations
- **FastAPI Backend**: RESTful APIs for all learning operations
- **Analytics**: Student performance analysis and learning insights

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI App  │    │  GraphRAG      │    │   Vector DB     │
│                 │◄──►│   Service      │◄──►│   (ChromaDB)    │
│   - REST APIs  │    │                 │    │                 │
│   - Auth       │    │   - NetworkX    │    │   - Embeddings  │
│   - CORS       │    │   - Graph       │    │   - Similarity  │
└─────────────────┘    │   - Traversal  │    └─────────────────┘
                       └─────────────────┘
                                │
                       ┌─────────────────┐
                       │   LLM Service   │
                       │                 │
                       │   - Ollama      │
                       │   - OpenAI      │
                       │   - Fallback    │
                       └─────────────────┘
```

## 📋 Prerequisites

- Python 3.12+
- `uv` package manager (recommended) or `pip`
- Ollama (optional, for local LLM)
- OpenAI API key (optional, for fallback LLM)

## 🛠️ Installation & Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd learning-outcome
```

### 2. Create Virtual Environment

```bash
# Using uv (recommended)
uv venv
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate     # On Windows

# Using venv
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate     # On Windows
```

### 3. Install Dependencies

```bash
# Using uv (recommended)
uv pip install .

# Using pip
pip install -e .
```

### 4. Environment Configuration

Create a `.env` file based on `.env.example`:

```bash
cp .env.example .env
```

Edit `.env` with your configuration:

```ini
# Application Settings
APP_NAME=Grade 6 Math Learning Prototype
APP_VERSION=0.1.0
DEBUG=true
LOG_LEVEL=INFO

# Server Settings
HOST=0.0.0.0
PORT=8000

# Database Settings
CHROMA_PERSIST_DIRECTORY=./chroma
VECTOR_DIMENSION=1024

# LLM Settings
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4

# Embedding Settings
EMBEDDING_MODEL=mxbai-embed-large
EMBEDDING_BATCH_SIZE=32
```

### 5. Generate Sample Data

```bash
# Generate knowledge graph
python scripts/generate_knowledge_graph.py

# Generate sample student data
python scripts/simulate_students.py
```

### 6. Data Loading Process

The application automatically loads the knowledge graph data into ChromaDB during startup. Here's how the process works:

#### 6.1 Knowledge Graph Structure

The `knowledge_graph.json` file contains a structured representation of math concepts:

```json
{
  "concepts": [
    {
      "concept_id": "concept_lcm_basic",
      "name": "Least Common Multiple (Basic)",
      "description": "Finding the smallest common multiple of two or more numbers",
      "grade_level": 6,
      "difficulty": 2,
      "prerequisites": ["concept_multiplication_basic"],
      "next_concepts": ["concept_lcm_advanced"],
      "tags": ["lcm", "multiples", "number_theory"],
      "learning_objectives": [...],
      "estimated_hours": 3.0
    }
  ]
}
```

#### 6.2 Vector Database Initialization

When the application starts, the GraphRAG service performs these steps:

1. **Load Knowledge Graph**: Reads `data/knowledge_graph.json`
2. **Build NetworkX Graph**: Creates a directed graph with concept relationships
3. **Generate Embeddings**: Creates vector representations for each concept
4. **Populate ChromaDB**: Stores concepts and embeddings in the vector database

#### 6.3 Embedding Generation Process

For each concept, the system:

1. **Creates Concept Text**: Combines name, description, and tags
   ```
   "Least Common Multiple (Basic) Finding the smallest common multiple of two or more numbers lcm multiples number_theory"
   ```

2. **Generates Embedding**: Uses the configured embedding model (currently simplified hash-based)
3. **Stores in ChromaDB**: Saves concept metadata and vector embedding

#### 6.4 ChromaDB Collection Structure

The vector database creates a `concepts` collection with:

- **Document IDs**: `concept_lcm_basic`, `concept_multiplication_basic`, etc.
- **Metadata**: Full concept information (name, description, difficulty, etc.)
- **Embeddings**: 1024-dimensional vectors for similarity search
- **Indexing**: Optimized for fast similarity queries

#### 6.5 Graph Relationship Building

Simultaneously, the system builds a NetworkX graph:

- **Nodes**: Each concept becomes a graph node
- **Edges**: Prerequisites and next_concepts create directed edges
- **Traversal**: Enables pathfinding between concepts

#### 6.6 Startup Logs

During initialization, you'll see logs like:

```
INFO - Found 16 concepts in knowledge graph
INFO - Built graph with 16 nodes and 14 edges
INFO - Populated vector database with 16 concepts
INFO - Loaded 16 concepts into knowledge graph
```

#### 6.7 Data Persistence

- **ChromaDB**: Automatically persists to `./chroma/` directory
- **Knowledge Graph**: Loaded from `data/knowledge_graph.json` on each startup
- **Embeddings**: Regenerated if ChromaDB is empty or corrupted

#### 6.8 Data Flow Summary

```
knowledge_graph.json → GraphRAG Service → ChromaDB + NetworkX Graph
       ↓                        ↓              ↓
   Concept Data         Embedding Gen    Vector Storage
   Relationships        Graph Building    Graph Traversal
   Metadata            Vector Storage    Similarity Search
```

### 7. Start the Application

```bash
python -m app.main
```

The server will start at `http://localhost:8000`

## 📚 API Endpoints

### Base URL
```
http://localhost:8000
```

### 1. Health Check

**GET** `/health`

Check the health status of all services.

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "services": {
    "graphrag": {
      "nodes": 16,
      "edges": 14,
      "concepts_loaded": 16,
      "embeddings_loaded": 16,
      "initialized": true
    },
    "llm": {
      "current_provider": "none",
      "ollama_available": false,
      "openai_available": false,
      "initialized": true
    },
    "embedding": {
      "model_name": "simple_hash_based",
      "vector_dimension": 1024,
      "batch_size": 32,
      "initialized": true
    }
  }
}
```

### 2. Concept Search

**POST** `/search`

Search for concepts using semantic similarity and graph relationships.

```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "lcm",
    "top_k": 3,
    "use_graph": true
  }'
```

**Request Body:**
```json
{
  "query": "lcm",
  "top_k": 3,
  "use_graph": true
}
```

**Response:**
```json
{
  "query": "lcm",
  "results": [
    {
      "concept": {
        "concept_id": "concept_lcm_advanced",
        "name": "Least Common Multiple (Advanced)",
        "description": "Advanced LCM techniques and applications...",
        "difficulty": 3,
        "grade_level": 6
      },
      "similarity_score": 0.067,
      "graph_score": 0.0,
      "combined_score": 0.047,
      "reasoning": "Moderate semantic similarity to query..."
    }
  ],
  "total_count": 1,
  "search_method": "graphrag"
}
```

### 3. Learning Recommendations

**POST** `/recommendations`

Get personalized concept recommendations for a student.

```bash
curl -X POST "http://localhost:8000/recommendations" \
  -H "Content-Type: application/json" \
  -d '{
    "student_id": "student_001",
    "current_concept": "concept_lcm_basic",
    "performance_score": 0.7,
    "limit": 3
  }'
```

**Request Body:**
```json
{
  "student_id": "student_001",
  "current_concept": "concept_lcm_basic",
  "performance_score": 0.7,
  "limit": 3
}
```

**Response:**
```json
{
  "student_id": "student_001",
  "current_concept": "concept_lcm_basic",
  "performance_score": 0.7,
  "recommendations": [
    {
      "concept_id": "concept_lcm_advanced",
      "name": "Least Common Multiple (Advanced)",
      "description": "Advanced LCM techniques and applications...",
      "difficulty": 3,
      "estimated_hours": 2.5,
      "reasoning": "Prerequisites satisfied; Good performance indicates readiness...",
      "confidence_score": 0.9,
      "prerequisites_met": true,
      "time_to_mastery": 2.5
    }
  ],
  "total_count": 1
}
```

### 4. Concept Information

**GET** `/concepts/{concept_id}`

Get detailed information about a specific concept.

```bash
curl http://localhost:8000/concepts/concept_lcm_basic
```

**Response:**
```json
{
  "concept_id": "concept_lcm_basic",
  "name": "Least Common Multiple (Basic)",
  "description": "Finding the smallest common multiple of two or more numbers",
  "grade_level": 6,
  "difficulty": 2,
  "estimated_hours": 3.0,
  "concept_type": "number_theory",
  "learning_objectives": [
    "Understand what LCM means and why it's useful",
    "Find LCM using listing method",
    "Find LCM using prime factorization"
  ],
  "prerequisites": [
    "concept_multiplication_basic",
    "concept_divisibility_rules"
  ],
  "next_concepts": [
    "concept_lcm_advanced",
    "concept_fraction_addition"
  ]
}
```

### 5. Next Concepts

**GET** `/concepts/{concept_id}/next?limit=5`

Get the next recommended concepts after completing a current concept.

```bash
curl "http://localhost:8000/concepts/concept_lcm_basic/next?limit=3"
```

### 6. Prerequisites

**GET** `/concepts/{concept_id}/prerequisites`

Get all prerequisite concepts for a given concept.

```bash
curl "http://localhost:8000/concepts/concept_lcm_basic/prerequisites"
```

### 7. Learning Path

**GET** `/concepts/{concept_id}/learning-path?max_depth={depth}`

Get the complete learning path from current concept to all reachable concepts.

**Parameters:**
- `concept_id`: The starting concept identifier
- `max_depth` (optional): Maximum depth to explore (1-10, default: 5)

```bash
curl "http://localhost:8000/concepts/concept_lcm_basic/learning-path?max_depth=3"
```

**Response:**
```json
{
  "current_concept": "concept_lcm_basic",
  "total_concepts": 8,
  "estimated_total_hours": 18.5,
  "path_levels": [
    ["concept_lcm_basic"],
    ["concept_lcm_advanced", "concept_fraction_addition"],
    ["concept_fraction_multiplication", "concept_algebraic_expressions"]
  ],
  "concept_details": {
    "concept_lcm_basic": {
      "name": "Least Common Multiple (Basic)",
      "description": "Finding the smallest common multiple of two or more numbers",
      "difficulty": 2,
      "estimated_hours": 2.5,
      "concept_type": "number_theory",
      "prerequisites": ["concept_multiplication_basic"],
      "next_concepts": ["concept_lcm_basic", "concept_fraction_addition"],
      "depth": 0
    },
    "concept_lcm_advanced": {
      "name": "Least Common Multiple (Advanced)",
      "description": "Advanced LCM techniques and applications",
      "difficulty": 3,
      "estimated_hours": 2.5,
      "concept_type": "number_theory",
      "prerequisites": ["concept_lcm_basic"],
      "next_concepts": ["concept_fraction_multiplication"],
      "depth": 1
    }
  }
}
```

**Use Cases:**
- **Learning Roadmap**: Show students their complete learning journey
- **Progress Planning**: Help teachers plan curriculum and pacing
- **Goal Setting**: Allow students to see what's ahead and set milestones
- **Time Estimation**: Provide realistic time expectations for learning goals

### 8. Graph Information

**GET** `/graph-info`

Get information about the knowledge graph structure.

```bash
curl http://localhost:8000/graph-info
```

**Response:**
```json
{
  "nodes": 16,
  "edges": 14,
  "concepts_loaded": 16,
  "embeddings_loaded": 16,
  "initialized": true
}
```

### 9. LLM Service Info

**GET** `/llm-info`

Get information about the LLM service status.

```bash
curl http://localhost:8000/llm-info
```

### 9. Embedding Service Info

**GET** `/embedding-info`

Get information about the embedding service.

```bash
curl http://localhost:8000/embedding-info
```

### 10. Performance Analysis

**POST** `/analyze-performance`

Analyze student performance on a specific concept using LLM insights.

```bash
curl -X POST "http://localhost:8000/analyze-performance" \
  -H "Content-Type: application/json" \
  -d '{
    "student_data": {
      "student_id": "student_001",
      "name": "Alex Johnson",
      "grade_level": 6,
      "age": 12,
      "learning_style": "visual",
      "math_aptitude": "above_average",
      "interests": ["puzzles", "games", "art"],
      "goals": ["master fractions", "improve problem solving"]
    },
    "concept_data": {
      "concept_id": "concept_lcm_basic",
      "name": "Least Common Multiple (Basic)",
      "description": "Finding the smallest common multiple of two or more numbers",
      "difficulty": 2,
      "grade_level": 6,
      "concept_type": "number_theory"
    },
    "performance_metrics": {
      "performance_score": 0.75,
      "attempts": 3,
      "time_spent": 2.5,
      "strengths": ["understanding prime factors", "listing multiples"],
      "weaknesses": ["finding LCM of more than 2 numbers", "word problems"],
      "mastery_level": "developing",
      "feedback": "Good progress on basic concepts, needs practice with complex scenarios"
    }
  }'
```

**Request Body:**
```json
{
  "student_data": {
    "student_id": "student_001",
    "name": "Alex Johnson",
    "grade_level": 6,
    "age": 12,
    "learning_style": "visual",
    "math_aptitude": "above_average",
    "interests": ["puzzles", "games", "art"],
    "goals": ["master fractions", "improve problem solving"]
  },
  "concept_data": {
    "concept_id": "concept_lcm_basic",
    "name": "Least Common Multiple (Basic)",
    "description": "Finding the smallest common multiple of two or more numbers",
    "difficulty": 2,
    "grade_level": 6,
    "concept_type": "number_theory"
  },
  "performance_metrics": {
    "performance_score": 0.75,
    "attempts": 3,
    "time_spent": 2.5,
    "strengths": ["understanding prime factors", "listing multiples"],
    "weaknesses": ["finding LCM of more than 2 numbers", "word problems"],
    "mastery_level": "developing",
    "feedback": "Good progress on basic concepts, needs practice with complex scenarios"
    }
}
```

**Response:**
```json
{
  "analysis": "Based on Alex's performance analysis, they show strong foundational understanding of LCM concepts with a 75% performance score. Their strengths in prime factorization and listing multiples indicate solid number theory skills. However, they struggle with complex scenarios involving more than 2 numbers and word problems, suggesting a need for more practice with real-world applications. Given their visual learning style and above-average aptitude, I recommend focusing on visual representations of complex LCM problems and gradually increasing difficulty. Their current mastery level of 'developing' suggests they're ready for intermediate challenges while reinforcing basic concepts.",
  "student_id": "student_001",
  "concept_id": "concept_lcm_basic",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

**Use Cases:**
- **Teacher Assessment**: Get AI-powered insights into student performance
- **Personalized Feedback**: Generate specific recommendations based on learning patterns
- **Progress Tracking**: Analyze learning trajectory and identify improvement areas
- **Intervention Planning**: Determine when additional support is needed

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `APP_NAME` | Application name | Grade 6 Math Learning Prototype |
| `APP_VERSION` | Application version | 0.1.0 |
| `HOST` | Server host | 0.0.0.0 |
| `PORT` | Server port | 8000 |
| `DEBUG` | Debug mode | true |
| `LOG_LEVEL` | Logging level | INFO |
| `CHROMA_PERSIST_DIRECTORY` | ChromaDB persistence directory | ./chroma |
| `VECTOR_DIMENSION` | Vector embedding dimension | 1024 |
| `OLLAMA_BASE_URL` | Ollama server URL | http://localhost:11434 |
| `OLLAMA_MODEL` | Ollama model name | llama3.2 |
| `OPENAI_API_KEY` | OpenAI API key | (required for OpenAI) |
| `OPENAI_MODEL` | OpenAI model name | gpt-4 |
| `EMBEDDING_MODEL` | Embedding model name | mxbai-embed-large |
| `EMBEDDING_BATCH_SIZE` | Embedding batch size | 32 |

### LLM Configuration

#### Ollama (Local)
1. Install Ollama: https://ollama.ai/
2. Pull the model: `ollama pull llama3.2`
3. Start Ollama service: `ollama serve`
4. Set `OLLAMA_BASE_URL` in your `.env` file

#### OpenAI (Fallback)
1. Get an API key from https://platform.openai.com/
2. Set `OPENAI_API_KEY` in your `.env` file
3. The service will automatically fallback to OpenAI if Ollama is unavailable

## 📊 Data Structure

### Knowledge Graph

The knowledge graph is stored in `data/knowledge_graph.json` with the following structure:

```json
{
  "metadata": {
    "generated_at": "2025-08-11T13:22:06.211380",
    "version": "1.0",
    "grade_level": 6,
    "total_concepts": 16,
    "description": "Grade 6 Math Knowledge Graph focusing on LCM and related concepts"
  },
  "concepts": [
    {
      "concept_id": "concept_lcm_basic",
      "name": "Least Common Multiple (Basic)",
      "grade_level": 6,
      "description": "Finding the smallest common multiple of two or more numbers",
      "difficulty": 2,
      "estimated_hours": 3.0,
      "concept_type": "number_theory",
      "learning_objectives": [...],
      "practice_problems": [...],
      "prerequisites": ["concept_multiplication_basic", "concept_divisibility_rules"],
      "next_concepts": ["concept_lcm_advanced", "concept_fraction_addition"],
      "tags": ["lcm", "number_theory", "multiples"],
      "metadata": {...}
    }
  ]
}
```

### Student Data

Sample student data is generated in `data/sample_students.json` with:

- Student profiles
- Performance data
- Learning journeys
- Progress tracking

## 🧪 Testing

### Run Tests

```bash
# Install test dependencies
uv pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=app
```

### Manual Testing

1. **Start the server**: `python -m app.main`
2. **Test health**: `curl http://localhost:8000/health`
3. **Test search**: Use the search endpoint examples above
4. **Test recommendations**: Use the recommendations endpoint examples above

## 🚀 Development

### Project Structure

```
learning-outcome/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration management
│   ├── models/              # Pydantic models
│   ├── services/            # Business logic services
│   ├── api/                 # API routes
│   ├── database/            # Database operations
│   └── utils/               # Utility functions
├── data/                    # Data files
├── chroma/                  # ChromaDB vector database
├── scripts/                 # Data generation scripts
├── tests/                   # Test files
├── pyproject.toml          # Project configuration
├── .env.example            # Environment template
└── README.md               # This file
```

### Adding New Concepts

1. Edit `scripts/generate_knowledge_graph.py`
2. Add new concept definitions
3. Update prerequisites and next_concepts relationships
4. Regenerate the knowledge graph: `python scripts/generate_knowledge_graph.py`
5. Restart the server

### Adding New Services

1. Create service file in `app/services/`
2. Implement required methods
3. Add to `app/services/__init__.py`
4. Update `app/main.py` if needed

## 🔍 Troubleshooting

### Common Issues

#### 1. "No LLM provider available"
- **Cause**: Neither Ollama nor OpenAI is configured
- **Solution**: Configure at least one LLM provider or accept limited functionality

#### 2. "Address already in use"
- **Cause**: Port 8000 is already occupied
- **Solution**: Stop existing processes or change port in `.env`

#### 3. "Failed to initialize Ollama"
- **Cause**: Ollama is not running or model not available
- **Solution**: Start Ollama service and pull required models

#### 4. "Graph has 0 edges"
- **Cause**: Concept IDs in prerequisites/next_concepts don't match actual concept IDs
- **Solution**: Regenerate knowledge graph with correct concept IDs

### Debug Mode

Enable debug mode in `.env`:

```ini
DEBUG=true
LOG_LEVEL=DEBUG
```

### Logs

Check application logs for detailed error information:

```bash
tail -f logs/app.log
```

## 📈 Performance

### Current Metrics

- **Knowledge Graph**: 16 concepts, 14 relationships
- **Vector Search**: ~100ms response time
- **Recommendations**: ~200ms response time
- **Memory Usage**: ~50MB base + embeddings

### Optimization Tips

1. **Vector Database**: Use SSD storage for better I/O performance
2. **Embeddings**: Consider using GPU-accelerated embedding models
3. **Caching**: Implement Redis for frequently accessed data
4. **Batch Processing**: Process multiple requests in batches

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:

1. Check the troubleshooting section
2. Review the logs
3. Open an issue on GitHub
4. Check the API documentation at `http://localhost:8000/docs` when running

## 🔮 Roadmap

- [ ] Web-based frontend interface
- [ ] Student progress tracking dashboard
- [ ] Advanced analytics and insights
- [ ] Integration with learning management systems
- [ ] Mobile app support
- [ ] Multi-language support
- [ ] Advanced LLM features (practice problem generation, explanations)
- [ ] Real-time collaborative learning features
