"""
Модуль для работы с векторной базой данных Qdrant
"""

from .qdrant_manager import QdrantManager
from .ollama_embeddings import OllamaEmbeddings
from .reranker import BGEReranker  # Добавляем импорт нового класса

__all__ = ['QdrantManager', 'OllamaEmbeddings', 'BGEReranker']