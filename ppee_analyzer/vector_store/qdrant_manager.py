"""
Класс для работы с векторной базой данных Qdrant
"""

import logging
from typing import List, Dict, Any, Optional, Union
import os

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore  # Импортируем из langchain_qdrant

from qdrant_client import QdrantClient
from qdrant_client.http import models

from .ollama_embeddings import OllamaEmbeddings

# Настройка логирования
logger = logging.getLogger(__name__)


class QdrantManager:
    """Класс для работы с векторной базой данных Qdrant"""

    def __init__(
        self,
        collection_name: str = "ppee_applications",
        host: str = "localhost",
        port: int = 6333,
        embeddings_type: str = "huggingface",  # Новый параметр для выбора типа эмбеддингов
        model_name: str = "BAAI/bge-m3",
        vector_size: int = 1024,
        device: str = "cuda",  # использовать "cpu" если нет GPU
        ollama_url: str = "http://localhost:11434",  # URL для Ollama
        create_collection: bool = True,
        check_availability: bool = True,
        ollama_options: Dict[str, Any] = None  # Опции для Ollama API
    ):
        """
        Инициализирует менеджер Qdrant.

        Args:
            collection_name: Имя коллекции в Qdrant
            host: Хост Qdrant
            port: Порт Qdrant
            embeddings_type: Тип эмбеддингов ("huggingface" или "ollama")
            model_name: Название модели для эмбеддингов
            vector_size: Размерность векторов
            device: Устройство для вычислений (cuda/cpu)
            ollama_url: URL для Ollama API
            create_collection: Создавать коллекцию, если она не существует
            check_availability: Проверять ли доступность модели при инициализации
            ollama_options: Опции для Ollama API
        """
        self.collection_name = collection_name
        self.host = host
        self.port = port
        self.model_name = model_name
        self.vector_size = vector_size
        self.device = device
        self.embeddings_type = embeddings_type
        self.ollama_url = ollama_url

        # Инициализация клиента Qdrant
        self.client = QdrantClient(host=host, port=port)

        # Инициализация модели эмбеддингов
        if embeddings_type.lower() == "ollama":
            logger.info(f"Используем эмбеддинги Ollama с моделью {model_name}")

            # Получаем опции из OllamaEmbeddings
            options = OllamaEmbeddings.get_default_options()

            # Обновляем опции, если они переданы
            if ollama_options:
                options.update(ollama_options)

            self.embeddings = OllamaEmbeddings(
                model_name=model_name,
                base_url=ollama_url,
                normalize_embeddings=True,
                check_availability=check_availability,
                options=options
            )
        else:  # По умолчанию используем HuggingFace
            logger.info(f"Используем эмбеддинги HuggingFace с моделью {model_name}")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': device},
                encode_kwargs={'normalize_embeddings': True}
            )

        # Проверяем существование коллекции и создаем при необходимости
        if create_collection:
            self._ensure_collection_exists()

        # Инициализация векторного хранилища
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings  # Изменяем embeddings на embedding
        )

    def _ensure_collection_exists(self):
        """Проверяет существование коллекции и создает при необходимости"""
        collections = self.client.get_collections().collections
        collection_names = [collection.name for collection in collections]

        if self.collection_name not in collection_names:
            logger.info(f"Создание коллекции {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE
                )
            )
            logger.info(f"Коллекция {self.collection_name} успешно создана")

            # Создаем индексы для ускорения фильтрации
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="metadata.application_id",
                field_schema="keyword"
            )
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="metadata.content_type",
                field_schema="keyword"
            )
        else:
            logger.info(f"Коллекция {self.collection_name} уже существует")

    def add_documents(self, documents: List[Document], batch_size: int = 32) -> int:
        """
        Добавляет документы в векторное хранилище.

        Args:
            documents: Список документов для добавления
            batch_size: Размер партии для индексации

        Returns:
            int: Количество добавленных документов
        """
        total_documents = len(documents)
        logger.info(f"Индексация {total_documents} документов")

        # Добавляем документы пакетами для оптимизации
        for i in range(0, total_documents, batch_size):
            end_idx = min(i + batch_size, total_documents)
            batch = documents[i:end_idx]
            self.vector_store.add_documents(batch)
            logger.info(f"Проиндексирована партия {i+1}-{end_idx} из {total_documents}")

        logger.info(f"Индексация завершена. Добавлено {total_documents} документов")
        return total_documents

    def search(
            self,
            query: str,
            filter_dict: Optional[Dict[str, Any]] = None,
            k: int = 3
    ) -> List[Document]:
        """
        Выполняет семантический поиск в векторном хранилище.

        Args:
            query: Текст запроса
            filter_dict: Словарь для фильтрации
            k: Количество результатов

        Returns:
            List[Document]: Список найденных документов
        """
        # Предпроцессинг запроса для модели bge
        processed_query = f"query: {query}" if "bge" in self.model_name.lower() else query

        # Форматирование фильтра для Qdrant
        filter_obj = None
        if filter_dict:
            conditions = []
            for key, value in filter_dict.items():
                conditions.append(
                    models.FieldCondition(
                        key=f"metadata.{key}",
                        match=models.MatchValue(value=value)
                    )
                )
            if conditions:
                filter_obj = models.Filter(must=conditions)

        # Выполнение поиска
        search_results = self.vector_store.similarity_search_with_score(
            query=processed_query,
            filter=filter_obj,
            k=k
        )

        # Преобразование результатов с сохранением scores
        documents = []
        for doc, score in search_results:
            # Добавляем score в метаданные
            doc.metadata['score'] = float(score)
            documents.append(doc)

        return documents

    def get_application_ids(self) -> List[str]:
        """
        Получает список ID заявок в хранилище.

        Returns:
            List[str]: Список ID заявок
        """
        response = self.client.scroll(
            collection_name=self.collection_name,
            limit=1000,
            with_payload=["metadata.application_id"],
            with_vectors=False
        )

        application_ids = set()
        for point in response[0]:
            if "metadata" in point.payload and "application_id" in point.payload["metadata"]:
                application_ids.add(point.payload["metadata"]["application_id"])

        return list(application_ids)

    def delete_application(self, application_id: str) -> int:
        """
        Удаляет заявку из хранилища.

        Args:
            application_id: ID заявки

        Returns:
            int: Количество удаленных точек
        """
        # Находим все точки, связанные с данной заявкой
        response = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.application_id",
                        match=models.MatchValue(value=application_id)
                    )
                ]
            ),
            limit=10000,
            with_payload=False,
            with_vectors=False
        )

        # Получаем ID точек для удаления
        points_to_delete = [point.id for point in response[0]]

        if points_to_delete:
            # Удаляем точки
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(
                    points=points_to_delete
                )
            )
            logger.info(f"Удалено {len(points_to_delete)} точек для заявки {application_id}")
            return len(points_to_delete)

        logger.info(f"Не найдено точек для заявки {application_id}")
        return 0

    def get_stats(self, application_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Получает статистику по хранилищу.

        Args:
            application_id: ID заявки для фильтрации (None - по всем заявкам)

        Returns:
            Dict[str, Any]: Статистика
        """
        # Настраиваем фильтр, если указан application_id
        scroll_filter = None
        if application_id:
            scroll_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.application_id",
                        match=models.MatchValue(value=application_id)
                    )
                ]
            )

        # Запрашиваем данные из Qdrant
        response = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=scroll_filter,
            limit=10000,  # Ограничение на количество результатов
            with_payload=["metadata"],
            with_vectors=False
        )

        # Анализируем результаты
        points = response[0]

        # Статистика по типам документов
        content_types = {}
        applications = set()
        documents = set()
        sections = set()

        for point in points:
            if "metadata" in point.payload:
                metadata = point.payload["metadata"]

                # Тип контента
                content_type = metadata.get("content_type", "unknown")
                content_types[content_type] = content_types.get(content_type, 0) + 1

                # Приложения и документы
                if "application_id" in metadata:
                    applications.add(metadata["application_id"])
                if "document_id" in metadata:
                    documents.add(metadata["document_id"])
                if "section" in metadata:
                    sections.add(metadata["section"])

        return {
            "total_points": len(points),
            "content_types": content_types,
            "applications_count": len(applications),
            "documents_count": len(documents),
            "sections_count": len(sections),
            "applications": list(applications),
            "documents": list(documents)
        }