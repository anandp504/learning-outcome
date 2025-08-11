"""Configuration settings for the Grade 6 Math Learning Prototype."""

from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Application Settings
    app_name: str = "Grade 6 Math Learning Prototype"
    app_version: str = "0.1.0"
    debug: bool = True
    log_level: str = "INFO"
    
    # Server Settings
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Database Settings
    chroma_persist_directory: str = "./chroma"
    vector_dimension: int = 1024
    
    # LLM Settings
    # Ollama (Local)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    
    # OpenAI (Fallback)
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4"
    openai_max_tokens: int = 1000
    
    # Embedding Settings
    embedding_model: str = "mxbai-embed-large"
    embedding_batch_size: int = 32
    
    # Graph Settings
    graph_cache_size: int = 1000
    max_recommendations: int = 5
    
    # Security
    api_key_header: str = "X-API-Key"
    rate_limit_per_minute: int = 100
    
    # Logging
    log_file: str = "./logs/app.log"
    log_max_size: str = "10MB"
    log_backup_count: int = 5
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()
