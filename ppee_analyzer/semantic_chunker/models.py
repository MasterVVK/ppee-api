"""
Модели данных для хранения результатов семантического разделения документов.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class SemanticChunk:
    """Класс для представления семантического фрагмента документа"""
    content: str
    type: str
    page: Optional[int] = None
    heading: Optional[str] = None
    table_id: Optional[str] = None
    pages: Optional[List[int]] = None
    section_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Преобразует чанк в словарь для сериализации"""
        return {
            'content': self.content,
            'type': self.type,
            'page': self.page,
            'heading': self.heading,
            'table_id': self.table_id,
            'pages': self.pages,
            'section_path': self.section_path
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SemanticChunk':
        """Создает чанк из словаря"""
        return cls(
            content=data['content'],
            type=data['type'],
            page=data.get('page'),
            heading=data.get('heading'),
            table_id=data.get('table_id'),
            pages=data.get('pages'),
            section_path=data.get('section_path')
        )


@dataclass
class DocumentAnalysisResult:
    """Класс для представления результатов анализа документа"""
    chunks: List[SemanticChunk]
    document_path: str
    statistics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Преобразует результат анализа в словарь для сериализации"""
        return {
            'document_path': self.document_path,
            'statistics': self.statistics,
            'chunks': [chunk.to_dict() for chunk in self.chunks]
        }