"""
Модуль для семантического разделения документов ППЭЭ
с использованием библиотеки docling.
"""

from .chunker import SemanticChunker, SemanticDocumentSplitter

__all__ = ['SemanticChunker', 'SemanticDocumentSplitter']